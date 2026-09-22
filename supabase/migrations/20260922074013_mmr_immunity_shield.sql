-- Match classification is additive: historical matches retain league treatment.
-- Deploy with the competition-aware website. This migration never replays ratings.
alter table public.matches add column competition text not null default 'league'
  constraint matches_competition_check check (competition in ('league','immunity_shield'));
comment on column public.matches.competition is 'League rates public MMR; Immunity Shield trains balancing only. Draws award no Shield.';

create or replace function public.commit_competition_rating_run(p_league_id bigint,p_expected_token text,p_request_id uuid,p_change jsonb,p_calculated jsonb,p_model_version text)
returns jsonb language plpgsql security invoker set search_path='' as $fn$
declare before_data jsonb; previous_run public.rating_runs; action text; m jsonb; new_match_id bigint; row_count integer;
        item jsonb; pid bigint; mid bigint; expected_players integer; expected_matches integer; rated_matches integer; saved_summary jsonb; shadow jsonb;
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
  if p_calculated->>'competitionPolicy' is distinct from 'league-shield-v1' then raise exception 'Refresh the website before saving competition results'; end if;
  action := p_change->>'action'; m := p_change->'match';
  if action not in ('add','edit','delete','rebuild') or action is null then raise exception 'Unknown action'; end if;
  if action in ('add','edit') and (m->>'competition' is null or m->>'competition' not in ('league','immunity_shield')) then raise exception 'Choose a valid competition'; end if;
  perform set_config('lovefive.rating_rebuild','on',true);
  if action='add' then
    insert into public.matches(league_id,date,team_a,team_b,score,result,processed,selection_source,review_note,competition)
    values(p_league_id,m->>'date',m->>'team_a',m->>'team_b',m->>'score',m->>'result',1,coalesce(m->>'selection_source','unknown'),btrim(coalesce(m->>'review_note','')),m->>'competition') returning id into new_match_id;
  elsif action='edit' then
    update public.matches set date=m->>'date',team_a=m->>'team_a',team_b=m->>'team_b',score=m->>'score',result=m->>'result',processed=1,
      competition=m->>'competition',
      selection_source=coalesce(m->>'selection_source',selection_source),
      review_note=case when m ? 'review_note' then btrim(coalesce(m->>'review_note','')) else review_note end where id=(m->>'id')::bigint and league_id=p_league_id;
    get diagnostics row_count = row_count;
    if row_count <> 1 then raise exception 'Match not found'; end if;
  elsif action='delete' then
    delete from public.matches where id=(m->>'id')::bigint and league_id=p_league_id;
    get diagnostics row_count = row_count;
    if row_count <> 1 then raise exception 'Match not found'; end if;
  end if;
  select count(*) into expected_players from public.players where league_id=p_league_id;
  select count(*) into expected_matches from public.matches where league_id=p_league_id and processed=1;
  select count(*) into rated_matches from public.matches where league_id=p_league_id and processed=1 and competition='league';
  if jsonb_typeof(p_calculated->'players') is distinct from 'array' or jsonb_typeof(p_calculated->'history') is distinct from 'array' then raise exception 'Incomplete rating calculation'; end if;
  if jsonb_array_length(p_calculated->'players') <> expected_players or jsonb_array_length(p_calculated->'history') <> rated_matches*10 then raise exception 'Incomplete rating calculation'; end if;
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
    if not exists(select 1 from public.players where id=pid and league_id=p_league_id) or not exists(select 1 from public.matches where id=mid and league_id=p_league_id and processed=1 and competition='league') then raise exception 'Invalid history reference'; end if;
    if not exists (
      select 1 from public.matches game join public.players player on player.id=pid and player.league_id=game.league_id
      where game.id=mid and game.league_id=p_league_id and game.date=item->>'date'
      and lower(btrim(player.name)) in (select lower(btrim(n)) from unnest(string_to_array(game.team_a||','||game.team_b,',')) n)
    ) then raise exception 'History player is not a participant or history date is incorrect'; end if;
    if item->>'mmr_before' is null or item->>'mmr_after' is null or
       not ((item->>'mmr_before')::float8 > '-Infinity'::float8 and (item->>'mmr_before')::float8 < 'Infinity'::float8 and
            (item->>'mmr_after')::float8 > '-Infinity'::float8 and (item->>'mmr_after')::float8 < 'Infinity'::float8)
       then raise exception 'Invalid history rating'; end if;
    insert into public.mmr_history(league_id,player_id,match_id,date,mmr_before,mmr_after)
    values(p_league_id,pid,mid,item->>'date',(item->>'mmr_before')::float8,(item->>'mmr_after')::float8);
  end loop;
  if exists(select 1 from public.mmr_history where league_id=p_league_id group by match_id having count(*)<>10 or count(distinct player_id)<>10) then raise exception 'Incomplete match history'; end if;
  if exists(select 1 from public.matches game where game.league_id=p_league_id and game.processed=1 and game.competition='league'
    and (select count(*) from public.mmr_history h where h.league_id=p_league_id and h.match_id=game.id) <> 10)
    then raise exception 'Missing league match history'; end if;
  for item in select value from jsonb_array_elements(p_calculated->'players') loop
    update public.players set mmr=(item->>'mmr')::float8,matches_played=(item->>'matches_played')::integer,
      wins=(item->>'wins')::integer,draws=(item->>'draws')::integer,losses=(item->>'losses')::integer,
      win_streak=(item->>'win_streak')::integer,lose_streak=(item->>'lose_streak')::integer
    where id=(item->>'id')::bigint and league_id=p_league_id;
  end loop;
  perform set_config('lovefive.rating_rebuild','off',true);
  saved_summary := jsonb_build_object('processedMatches',expected_matches,'ratedMatches',rated_matches,'playersUpdated',expected_players,'historyRows',rated_matches*10);
  insert into public.rating_runs(id,league_id,model_version,source_token,change,before_snapshot,summary)
  values(p_request_id,p_league_id,p_model_version,p_expected_token,p_change,before_data,saved_summary);
  shadow := p_calculated->'shadow';
  if jsonb_typeof(shadow->'records') is distinct from 'array' or jsonb_array_length(shadow->'records') <> expected_matches then
    raise exception 'Incomplete balancing history'; end if;
  if new_match_id is not null then
    select jsonb_set(shadow,'{records}',jsonb_agg(case when (r->>'matchId')::bigint=9007199254740991 then jsonb_set(r,'{matchId}',to_jsonb(new_match_id)) else r end order by ord))
      into shadow from jsonb_array_elements(shadow->'records') with ordinality as rows(r,ord);
  end if;
  if (select count(distinct (r->>'matchId')::bigint) from jsonb_array_elements(shadow->'records') r) <> expected_matches
    or exists(select 1 from jsonb_array_elements(shadow->'records') r where not exists(
      select 1 from public.matches game where game.league_id=p_league_id and game.processed=1 and game.id=(r->>'matchId')::bigint))
    then raise exception 'Invalid balancing history'; end if;
  insert into public.balance_model_states(league_id,model_version,run_id,state,ratings,replay_predictions,config)
  values(p_league_id,p_model_version,p_request_id,shadow->'state',shadow->'ratings',shadow->'records',shadow->'config')
  on conflict(league_id,model_version) do update set run_id=excluded.run_id,state=excluded.state,ratings=excluded.ratings,
    replay_predictions=excluded.replay_predictions,config=excluded.config,updated_at=now();
  return saved_summary;
end $fn$;
revoke all on function public.commit_competition_rating_run(bigint,text,uuid,jsonb,jsonb,text) from public,anon;
grant execute on function public.commit_competition_rating_run(bigint,text,uuid,jsonb,jsonb,text) to authenticated;



-- An old client must not submit a league-only replay after Shield fixtures exist.
create or replace function public.commit_rating_run(p_league_id bigint,p_expected_token text,p_request_id uuid,p_change jsonb,p_calculated jsonb,p_model_version text)
returns jsonb language plpgsql security invoker set search_path='' as $fn$
begin
  raise exception 'Refresh the website: result saving now requires competition support';
end $fn$;
revoke all on function public.commit_rating_run(bigint,text,uuid,jsonb,jsonb,text) from public,anon;
grant execute on function public.commit_rating_run(bigint,text,uuid,jsonb,jsonb,text) to authenticated;

-- Enforce immunity even on direct history writes. Existing member/admin RLS stays in place.
create function public.competition_write_guard() returns trigger language plpgsql security invoker set search_path='' as $fn$
begin
  if TG_TABLE_NAME='matches' then
    if NEW.competition='immunity_shield' and coalesce(current_setting('lovefive.rating_rebuild',true),'') <> 'on' then
      raise exception 'Use competition result saving for Shield fixtures';
    end if;
  elsif exists(select 1 from public.matches game where game.id=NEW.match_id and game.league_id=NEW.league_id and game.competition='immunity_shield') then
    raise exception 'Shield matches cannot have public MMR history';
  end if;
  return NEW;
end $fn$;
revoke all on function public.competition_write_guard() from public,anon,authenticated;
create trigger competition_write_guard before insert or update on public.matches for each row execute function public.competition_write_guard();
create trigger competition_write_guard before insert or update on public.mmr_history for each row execute function public.competition_write_guard();

notify pgrst, 'reload schema';
