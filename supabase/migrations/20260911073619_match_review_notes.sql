-- Optional editorial context. No rating replay and no existing result changes.
alter table public.matches add column review_note text not null default ''
  constraint matches_review_note_length check (char_length(review_note) <= 1500);
comment on column public.matches.review_note is 'Optional league-admin match context, displayed as plain text.';

create or replace function public.commit_rating_run(p_league_id bigint,p_expected_token text,p_request_id uuid,p_change jsonb,p_calculated jsonb,p_model_version text)
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
    insert into public.matches(league_id,date,team_a,team_b,score,result,processed,selection_source,review_note)
    values(p_league_id,m->>'date',m->>'team_a',m->>'team_b',m->>'score',m->>'result',1,coalesce(m->>'selection_source','unknown'),btrim(coalesce(m->>'review_note',''))) returning id into new_match_id;
  elsif action='edit' then
    update public.matches set date=m->>'date',team_a=m->>'team_a',team_b=m->>'team_b',score=m->>'score',result=m->>'result',processed=1,
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


-- Editing prose alone must not recalculate ratings. Use the same league lock and
-- member policies as result saving, plus a compare-and-swap to avoid lost notes.
create function public.save_match_review_note(p_league_id bigint,p_match_id bigint,p_expected_note text,p_note text)
returns text language plpgsql security invoker set search_path='' as $fn$
declare old_note text; clean_note text := btrim(coalesce(p_note,'')); old_guard text;
begin
  if auth.uid() is null or not exists(select 1 from public.league_members where league_id=p_league_id
      and user_id=(select auth.uid()) and status='active' and role in ('owner','admin')) then
    raise exception 'League admin access required' using errcode='42501';
  end if;
  if char_length(coalesce(p_note,'')) > 1500 then raise exception 'Keep the note to 1500 characters.' using errcode='22001'; end if;
  perform pg_advisory_xact_lock(hashtextextended('lovefive-ratings:'||p_league_id::text,0));
  select review_note into old_note from public.matches where id=p_match_id and league_id=p_league_id for update;
  if not found then raise exception 'Match not found' using errcode='P0002'; end if;
  -- A repeated request after a lost response is harmless.
  if old_note = clean_note then return old_note; end if;
  if old_note is distinct from coalesce(p_expected_note,'') then
    raise exception 'Match note changed while editing' using errcode='40001';
  end if;
  old_guard := coalesce(current_setting('lovefive.rating_rebuild',true),'');
  perform set_config('lovefive.rating_rebuild','on',true);
  update public.matches set review_note=clean_note where id=p_match_id and league_id=p_league_id;
  perform set_config('lovefive.rating_rebuild',old_guard,true);
  return clean_note;
end $fn$;
revoke all on function public.save_match_review_note(bigint,bigint,text,text) from public,anon;
grant execute on function public.save_match_review_note(bigint,bigint,text,text) to authenticated;
notify pgrst, 'reload schema';
