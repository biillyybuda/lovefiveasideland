-- Collect baseline readings as optional snapshots attached to a historical match.
-- No fitness formulas or changes to ratings, awards, or team selection.
alter table public.match_player_activity
  add column resting_heart_rate integer check (resting_heart_rate between 1 and 300),
  add column personal_max_heart_rate integer check (personal_max_heart_rate between 1 and 300),
  drop constraint activity_has_reading,
  add constraint activity_has_reading check (
    num_nonnulls(distance_km,duration_minutes,average_heart_rate,max_heart_rate,resting_heart_rate,personal_max_heart_rate) > 0),
  add constraint activity_baseline_heart_rate_order check (resting_heart_rate < personal_max_heart_rate);

comment on column public.match_player_activity.max_heart_rate is 'Peak heart rate recorded during this match, not a personal HRmax baseline.';
comment on column public.match_player_activity.resting_heart_rate is 'Optional resting heart rate known around this match date; never inferred or carried between matches.';
comment on column public.match_player_activity.personal_max_heart_rate is 'Optional known personal HRmax around this match date, entered separately from the match peak.';

create or replace function public.save_match_player_activity(p_league_id bigint,p_match_id bigint,p_player_id bigint,
  p_expected_version uuid,p_values jsonb,p_remove boolean default false)
returns jsonb language plpgsql security invoker set search_path='' as $fn$
declare existing public.match_player_activity; saved public.match_player_activity; desired jsonb;
begin
  if auth.uid() is null or not exists(select 1 from public.league_members where league_id=p_league_id
    and user_id=(select auth.uid()) and status='active' and role in ('owner','admin')) then
    raise exception 'League admin access required' using errcode='42501';
  end if;
  perform pg_advisory_xact_lock(hashtextextended('lovefive-ratings:'||p_league_id::text,0));
  if not exists(select 1 from public.matches where id=p_match_id and league_id=p_league_id and processed=1) then
    raise exception 'Match not found' using errcode='P0002';
  end if;
  select * into existing from public.match_player_activity where league_id=p_league_id and match_id=p_match_id and player_id=p_player_id for update;
  if p_remove then
    if existing.version is null then return null; end if;
    if existing.version is distinct from p_expected_version then raise exception 'Activity changed while editing' using errcode='40001'; end if;
    delete from public.match_player_activity where league_id=p_league_id and match_id=p_match_id and player_id=p_player_id;
    return null;
  end if;
  if p_values is null or jsonb_typeof(p_values)<>'object' then raise exception 'Enter activity readings' using errcode='22023'; end if;
  -- Omitted NEW fields preserve saved values for older clients. Explicit null clears them.
  desired:=jsonb_build_object('distance_km',(p_values->>'distance_km')::numeric,
    'duration_minutes',(p_values->>'duration_minutes')::numeric,
    'average_heart_rate',(p_values->>'average_heart_rate')::integer,
    'max_heart_rate',(p_values->>'max_heart_rate')::integer,
    'resting_heart_rate',case when p_values ? 'resting_heart_rate' then (p_values->>'resting_heart_rate')::integer else existing.resting_heart_rate end,
    'personal_max_heart_rate',case when p_values ? 'personal_max_heart_rate' then (p_values->>'personal_max_heart_rate')::integer else existing.personal_max_heart_rate end,
    'source',btrim(coalesce(p_values->>'source','')));
  if existing.version is not null and desired=jsonb_build_object('distance_km',existing.distance_km,
    'duration_minutes',existing.duration_minutes,'average_heart_rate',existing.average_heart_rate,
    'max_heart_rate',existing.max_heart_rate,'resting_heart_rate',existing.resting_heart_rate,
    'personal_max_heart_rate',existing.personal_max_heart_rate,'source',existing.source) then return to_jsonb(existing); end if;
  if existing.version is distinct from p_expected_version then raise exception 'Activity changed while editing' using errcode='40001'; end if;
  if existing.version is null then
    insert into public.match_player_activity(league_id,match_id,player_id,distance_km,duration_minutes,average_heart_rate,max_heart_rate,resting_heart_rate,personal_max_heart_rate,source)
      values(p_league_id,p_match_id,p_player_id,(desired->>'distance_km')::numeric,(desired->>'duration_minutes')::numeric,
        (desired->>'average_heart_rate')::integer,(desired->>'max_heart_rate')::integer,
        (desired->>'resting_heart_rate')::integer,(desired->>'personal_max_heart_rate')::integer,desired->>'source') returning * into saved;
  else
    update public.match_player_activity set distance_km=(desired->>'distance_km')::numeric,duration_minutes=(desired->>'duration_minutes')::numeric,
      average_heart_rate=(desired->>'average_heart_rate')::integer,max_heart_rate=(desired->>'max_heart_rate')::integer,
      resting_heart_rate=(desired->>'resting_heart_rate')::integer,personal_max_heart_rate=(desired->>'personal_max_heart_rate')::integer,source=desired->>'source'
      where league_id=p_league_id and match_id=p_match_id and player_id=p_player_id returning * into saved;
  end if;
  return to_jsonb(saved);
end $fn$;
revoke all on function public.save_match_player_activity(bigint,bigint,bigint,uuid,jsonb,boolean) from public,anon;
grant execute on function public.save_match_player_activity(bigint,bigint,bigint,uuid,jsonb,boolean) to authenticated;
notify pgrst, 'reload schema';
