-- Additive foundation. Does not publish new ratings or replay any result.
-- The TypeScript replay is authoritative; admin submissions are committed atomically.
alter table public.players add column initial_mmr double precision;
update public.players p set initial_mmr = coalesce(
  (select h.mmr_before from public.mmr_history h where h.player_id=p.id and h.league_id=p.league_id order by h.date,h.match_id,h.id limit 1),
  p.mmr,1000);
alter table public.players alter column initial_mmr set not null;
alter table public.matches add column selection_source text not null default 'unknown'
  check (selection_source in ('unknown','committee','app','adjusted_app'));

create table public.rating_adjustments (
  id bigint generated always as identity primary key,
  league_id bigint not null references public.leagues(id) on delete cascade,
  player_id bigint not null references public.players(id) on delete cascade,
  after_date text not null default '',
  after_match_id bigint not null default 0,
  delta double precision not null check (delta > '-Infinity'::float8 and delta < 'Infinity'::float8),
  reason text not null,
  created_at timestamptz not null default now(),
  created_by uuid default auth.uid()
);
create index rating_adjustments_league_order on public.rating_adjustments(league_id,after_date,after_match_id,id);
create index rating_adjustments_player on public.rating_adjustments(player_id);

-- Preserve discontinuities already present in history as dated manual adjustments.
insert into public.rating_adjustments(league_id,player_id,after_date,after_match_id,delta,reason)
select league_id,player_id,coalesce(previous_date,''),coalesce(previous_match,0),mmr_before-previous_after,'Preserved historical adjustment'
from (select h.*,lag(mmr_after) over w previous_after,lag(date) over w previous_date,lag(match_id) over w previous_match
      from public.mmr_history h window w as (partition by league_id,player_id order by date,match_id,id)) h
where previous_after is not null and abs(mmr_before-previous_after)>0.000001;
insert into public.rating_adjustments(league_id,player_id,after_date,after_match_id,delta,reason)
select p.league_id,p.id,coalesce(m.date,''),coalesce(m.id,0),p.mmr-h.mmr_after,'Preserved current manual adjustment'
from public.players p
join lateral (select mmr_after from public.mmr_history where player_id=p.id and league_id=p.league_id order by date desc,match_id desc,id desc limit 1) h on true
left join lateral (select id,date from public.matches where league_id=p.league_id and processed=1 order by date desc,id desc limit 1) m on true
where abs(p.mmr-h.mmr_after)>0.000001;

create table public.rating_runs (
  id uuid primary key,
  league_id bigint not null references public.leagues(id) on delete cascade,
  model_version text not null,
  source_token text not null,
  change jsonb not null,
  before_snapshot jsonb not null,
  summary jsonb not null,
  created_at timestamptz not null default now(),
  created_by uuid default auth.uid()
);
create index rating_runs_league_date on public.rating_runs(league_id,created_at desc);
create table public.balance_model_states (
  league_id bigint not null references public.leagues(id) on delete cascade,
  model_version text not null,
  run_id uuid not null references public.rating_runs(id),
  state jsonb not null,
  ratings jsonb not null,
  replay_predictions jsonb not null,
  config jsonb not null,
  updated_at timestamptz not null default now(),
  primary key(league_id,model_version)
);
create index balance_model_states_run on public.balance_model_states(run_id);
create table public.team_selection_snapshots (
  id uuid primary key,
  league_id bigint not null references public.leagues(id) on delete cascade,
  model_version text not null,
  source_token text not null,
  match_date date not null,
  snapshot jsonb not null,
  created_at timestamptz not null default now(),
  created_by uuid default auth.uid()
);
create index team_selection_snapshots_league_date on public.team_selection_snapshots(league_id,match_date,created_at);

do $policies$
declare t text;
begin
  foreach t in array array['rating_adjustments','rating_runs','balance_model_states','team_selection_snapshots'] loop
    execute format('alter table public.%I enable row level security',t);
    execute format('create policy admin_access on public.%I for all to authenticated using (exists (select 1 from public.league_members lm where lm.league_id=%I.league_id and lm.user_id=(select auth.uid()) and lm.status=''active'' and lm.role in (''owner'',''admin''))) with check (exists (select 1 from public.league_members lm where lm.league_id=%I.league_id and lm.user_id=(select auth.uid()) and lm.status=''active'' and lm.role in (''owner'',''admin'')))',t,t,t);
    execute format('revoke all on public.%I from anon',t);
  end loop;
end $policies$;
grant select,insert on public.rating_adjustments,public.rating_runs,public.team_selection_snapshots to authenticated;
grant select,insert,update on public.balance_model_states to authenticated;
grant usage on sequence public.rating_adjustments_id_seq to authenticated;

-- Align the existing owner role with the UI and the match/history policies.
alter policy players_league_admin_write on public.players
using (exists(select 1 from public.league_members lm where lm.league_id=players.league_id and lm.user_id=(select auth.uid()) and lm.status='active' and lm.role in ('owner','admin')))
with check (exists(select 1 from public.league_members lm where lm.league_id=players.league_id and lm.user_id=(select auth.uid()) and lm.status='active' and lm.role in ('owner','admin')));

create function public.rating_write_guard() returns trigger language plpgsql security invoker set search_path='' as $fn$
declare lid bigint; anchor record; goals text[]; names text[];
begin
  lid := case when TG_OP='DELETE' then OLD.league_id else NEW.league_id end;
  if TG_OP='UPDATE' and NEW.league_id is distinct from OLD.league_id then raise exception 'Move players and matches through an explicit league migration.'; end if;
  perform pg_advisory_xact_lock(hashtextextended('lovefive-ratings:'||lid::text,0));
  -- After this league starts the new writer, old clients cannot partially rebuild it.
  if TG_TABLE_NAME in ('matches','mmr_history') and coalesce(current_setting('lovefive.rating_rebuild',true),'') <> 'on'
     and exists(select 1 from public.balance_model_states where league_id=lid) then
    raise exception 'Use the updated website to save or rebuild results for this league.';
  end if;
  if TG_TABLE_NAME='players' then
    if TG_OP='INSERT' then NEW.initial_mmr := coalesce(NEW.initial_mmr,NEW.mmr,1000); end if;
    if TG_OP='UPDATE' and NEW.mmr is distinct from OLD.mmr and coalesce(current_setting('lovefive.rating_rebuild',true),'') <> 'on' then
      if NEW.mmr is null or not (NEW.mmr > '-Infinity'::float8 and NEW.mmr < 'Infinity'::float8) then raise exception 'Invalid rating'; end if;
      select id,date into anchor from public.matches where league_id=lid and processed=1 order by date desc,id desc limit 1;
      insert into public.rating_adjustments(league_id,player_id,after_date,after_match_id,delta,reason)
      values(lid,NEW.id,coalesce(anchor.date,''),coalesce(anchor.id,0),NEW.mmr-coalesce(OLD.mmr,1000),'Admin rating adjustment');
    end if;
  end if;
  if TG_TABLE_NAME='matches' and TG_OP <> 'DELETE' then
   if NEW.processed=1 then
    if NEW.score is null or NEW.score !~ '^\s*[0-9]{1,3}\s*[-:]\s*[0-9]{1,3}\s*$' then raise exception 'Enter a complete score, such as 10-8.'; end if;
    if NEW.date is null or NEW.date !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' then raise exception 'Invalid match date'; end if;
    perform NEW.date::date;
    goals:=regexp_split_to_array(regexp_replace(NEW.score,'\s','','g'),'[-:]');
    NEW.score:=(goals[1]::integer)::text||'-'||(goals[2]::integer)::text;
    NEW.result:=case when goals[1]::integer>goals[2]::integer then 'A' when goals[1]::integer<goals[2]::integer then 'B' else 'Draw' end;
    if cardinality(string_to_array(NEW.team_a,',')) is distinct from 5 or cardinality(string_to_array(NEW.team_b,',')) is distinct from 5 then raise exception 'Choose five players per side'; end if;
    select array_agg(lower(btrim(n))) into names from unnest(string_to_array(NEW.team_a||','||NEW.team_b,',')) n;
    if (select count(distinct n) from unnest(names) n) <> 10 then raise exception 'Choose ten distinct players'; end if;
    if (select count(*) from public.players p where p.league_id=lid and lower(btrim(p.name))=any(names)) <> 10 then raise exception 'A selected player is missing from this league'; end if;
   end if;
  end if;
  if TG_OP='DELETE' then return OLD; end if;
  return NEW;
end $fn$;
revoke all on function public.rating_write_guard() from public,anon,authenticated;
create trigger rating_write_guard before insert or update or delete on public.players for each row execute function public.rating_write_guard();
create trigger rating_write_guard before insert or update or delete on public.matches for each row execute function public.rating_write_guard();
create trigger rating_write_guard before insert or update or delete on public.mmr_history for each row execute function public.rating_write_guard();
create trigger rating_write_guard before insert or update or delete on public.rating_adjustments for each row execute function public.rating_write_guard();

create function public.rating_workspace(p_league_id bigint) returns jsonb language plpgsql security invoker set search_path='' as $fn$
declare snapshot jsonb;
begin
  if auth.uid() is null or not exists(select 1 from public.league_members where league_id=p_league_id and user_id=auth.uid() and status='active' and role in ('owner','admin')) then
    raise exception 'League admin access required' using errcode='42501';
  end if;
  select jsonb_build_object(
    'players',coalesce((select jsonb_agg(to_jsonb(p) order by id) from public.players p where league_id=p_league_id),'[]'::jsonb),
    'matches',coalesce((select jsonb_agg(to_jsonb(m) order by date,id) from public.matches m where league_id=p_league_id),'[]'::jsonb),
    'history',coalesce((select jsonb_agg(to_jsonb(h) order by date,match_id,id) from public.mmr_history h where league_id=p_league_id),'[]'::jsonb),
    'adjustments',coalesce((select jsonb_agg(to_jsonb(a) order by after_date,after_match_id,id) from public.rating_adjustments a where league_id=p_league_id),'[]'::jsonb)
  ) into snapshot;
  return snapshot || jsonb_build_object('token',md5(snapshot::text));
end $fn$;
revoke all on function public.rating_workspace(bigint) from public,anon;
grant execute on function public.rating_workspace(bigint) to authenticated;

create function public.commit_rating_run(p_league_id bigint,p_expected_token text,p_request_id uuid,p_change jsonb,p_calculated jsonb,p_model_version text)
returns jsonb language plpgsql security invoker set search_path='' as $fn$
declare before_data jsonb; previous_run public.rating_runs; action text; m jsonb; new_match_id bigint; row_count integer;
        item jsonb; pid bigint; mid bigint; expected_players integer; expected_matches integer; saved_summary jsonb; shadow jsonb;
begin
  -- Every old and new writer uses the same league lock through the triggers.
  perform pg_advisory_xact_lock(hashtextextended('lovefive-ratings:'||p_league_id::text,0));
  before_data := public.rating_workspace(p_league_id);
  select * into previous_run from public.rating_runs where id=p_request_id;
  if found then
    if previous_run.league_id <> p_league_id or previous_run.change <> p_change then raise exception 'Request ID already used'; end if;
    return previous_run.summary;
  end if;
  if before_data->>'token' is distinct from p_expected_token then raise exception 'League changed during calculation' using errcode='40001'; end if;
  if p_model_version <> 'margin-v1' or p_calculated->'shadow'->'state'->>'version' is distinct from p_model_version then raise exception 'Unknown rating version'; end if;
  action := p_change->>'action'; m := p_change->'match';
  if action not in ('add','edit','delete','rebuild') or action is null then raise exception 'Unknown action'; end if;
  perform set_config('lovefive.rating_rebuild','on',true);
  if action='add' then
    insert into public.matches(league_id,date,team_a,team_b,score,result,processed,selection_source)
    values(p_league_id,m->>'date',m->>'team_a',m->>'team_b',m->>'score',m->>'result',1,coalesce(m->>'selection_source','unknown')) returning id into new_match_id;
  elsif action='edit' then
    update public.matches set date=m->>'date',team_a=m->>'team_a',team_b=m->>'team_b',score=m->>'score',result=m->>'result',processed=1,
      selection_source=coalesce(m->>'selection_source',selection_source) where id=(m->>'id')::bigint and league_id=p_league_id;
    get diagnostics row_count = row_count;
    if row_count <> 1 then raise exception 'Match not found'; end if;
  elsif action='delete' then
    delete from public.matches where id=(m->>'id')::bigint and league_id=p_league_id;
    get diagnostics row_count = row_count;
    if row_count <> 1 then raise exception 'Match not found'; end if;
  end if;
  select count(*) into expected_players from public.players where league_id=p_league_id;
  select count(*) into expected_matches from public.matches where league_id=p_league_id and processed=1;
  if jsonb_array_length(p_calculated->'players') <> expected_players or jsonb_array_length(p_calculated->'history') <> expected_matches*10 then raise exception 'Incomplete rating calculation'; end if;
  if (select count(distinct (p->>'id')::bigint) from jsonb_array_elements(p_calculated->'players') p) <> expected_players then raise exception 'Duplicate players'; end if;
  -- Data validation precedes replacement; any later failure rolls the entire call back.
  for item in select value from jsonb_array_elements(p_calculated->'players') loop
    pid := (item->>'id')::bigint;
    if not exists(select 1 from public.players where id=pid and league_id=p_league_id) then raise exception 'Player belongs to another league'; end if;
    if not ((item->>'mmr')::float8 > '-Infinity'::float8 and (item->>'mmr')::float8 < 'Infinity'::float8) or item->>'mmr' is null then raise exception 'Invalid calculated rating'; end if;
  end loop;
  perform set_config('lovefive.rating_rebuild','on',true);
  delete from public.mmr_history where league_id=p_league_id;
  for item in select value from jsonb_array_elements(p_calculated->'history') loop
    pid := (item->>'player_id')::bigint;
    mid := (item->>'match_id')::bigint;
    if mid=9007199254740991 then mid:=new_match_id; end if;
    if not exists(select 1 from public.players where id=pid and league_id=p_league_id) or not exists(select 1 from public.matches where id=mid and league_id=p_league_id and processed=1) then raise exception 'Invalid history reference'; end if;
    insert into public.mmr_history(league_id,player_id,match_id,date,mmr_before,mmr_after)
    values(p_league_id,pid,mid,item->>'date',(item->>'mmr_before')::float8,(item->>'mmr_after')::float8);
  end loop;
  if exists(select 1 from public.mmr_history where league_id=p_league_id group by match_id having count(*)<>10 or count(distinct player_id)<>10) then raise exception 'Incomplete match history'; end if;
  for item in select value from jsonb_array_elements(p_calculated->'players') loop
    update public.players set mmr=(item->>'mmr')::float8,matches_played=(item->>'matches_played')::integer,
      wins=(item->>'wins')::integer,draws=(item->>'draws')::integer,losses=(item->>'losses')::integer,
      win_streak=(item->>'win_streak')::integer,lose_streak=(item->>'lose_streak')::integer
    where id=(item->>'id')::bigint and league_id=p_league_id;
  end loop;
  perform set_config('lovefive.rating_rebuild','off',true);
  saved_summary := jsonb_build_object('processedMatches',expected_matches,'playersUpdated',expected_players,'historyRows',expected_matches*10);
  insert into public.rating_runs(id,league_id,model_version,source_token,change,before_snapshot,summary)
  values(p_request_id,p_league_id,p_model_version,p_expected_token,p_change,before_data,saved_summary);
  shadow := p_calculated->'shadow';
  if new_match_id is not null then
    select jsonb_set(shadow,'{records}',jsonb_agg(case when (r->>'matchId')::bigint=9007199254740991 then jsonb_set(r,'{matchId}',to_jsonb(new_match_id)) else r end order by ord))
      into shadow from jsonb_array_elements(shadow->'records') with ordinality as rows(r,ord);
  end if;
  insert into public.balance_model_states(league_id,model_version,run_id,state,ratings,replay_predictions,config)
  values(p_league_id,p_model_version,p_request_id,shadow->'state',shadow->'ratings',shadow->'records',shadow->'config')
  on conflict(league_id,model_version) do update set run_id=excluded.run_id,state=excluded.state,ratings=excluded.ratings,
    replay_predictions=excluded.replay_predictions,config=excluded.config,updated_at=now();
  return saved_summary;
end $fn$;
revoke all on function public.commit_rating_run(bigint,text,uuid,jsonb,jsonb,text) from public,anon;
grant execute on function public.commit_rating_run(bigint,text,uuid,jsonb,jsonb,text) to authenticated;

create function public.save_team_selection(p_league_id bigint,p_expected_token text,p_id uuid,p_match_date date,p_snapshot jsonb)
returns uuid language plpgsql security invoker set search_path='' as $fn$
declare workspace jsonb;
begin
  perform pg_advisory_xact_lock(hashtextextended('lovefive-ratings:'||p_league_id::text,0));
  workspace:=public.rating_workspace(p_league_id);
  if workspace->>'token' is distinct from p_expected_token then raise exception 'League changed during selection' using errcode='40001'; end if;
  if p_match_date < current_date then raise exception 'Pre-match forecasts must be recorded before the match date ends'; end if;
  if p_snapshot->>'version' is distinct from 'margin-v1' then raise exception 'Unknown rating version'; end if;
  insert into public.team_selection_snapshots(id,league_id,model_version,source_token,match_date,snapshot)
  values(p_id,p_league_id,'margin-v1',p_expected_token,p_match_date,p_snapshot);
  return p_id;
end $fn$;
revoke all on function public.save_team_selection(bigint,text,uuid,date,jsonb) from public,anon;
grant execute on function public.save_team_selection(bigint,text,uuid,date,jsonb) to authenticated;
