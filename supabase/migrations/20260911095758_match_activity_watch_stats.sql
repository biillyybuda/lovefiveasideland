-- Optional watch-reported figures. No estimates are inferred and no ratings logic is changed.
alter table public.match_player_activity
  add column average_pace_seconds integer check (average_pace_seconds between 1 and 7200),
  add column moving_pace_seconds integer check (moving_pace_seconds between 1 and 7200),
  add column best_pace_seconds integer check (best_pace_seconds between 1 and 7200),
  add column average_speed_kmh numeric check (average_speed_kmh between 0 and 60),
  add column moving_speed_kmh numeric check (moving_speed_kmh between 0 and 60),
  add column max_speed_kmh numeric check (max_speed_kmh between 0 and 60),
  add column moving_minutes numeric check (moving_minutes between 0 and 600),
  add column elapsed_minutes numeric check (elapsed_minutes between 0.01 and 600),
  add column calories integer check (calories between 0 and 10000),
  add column average_respiration integer check (average_respiration between 1 and 120),
  add column min_respiration integer check (min_respiration between 1 and 120),
  add column max_respiration integer check (max_respiration between 1 and 120),
  drop constraint activity_has_reading,
  add constraint activity_has_reading check (num_nonnulls(distance_km,duration_minutes,average_heart_rate,max_heart_rate,resting_heart_rate,personal_max_heart_rate,average_pace_seconds,moving_pace_seconds,best_pace_seconds,average_speed_kmh,moving_speed_kmh,max_speed_kmh,moving_minutes,elapsed_minutes,calories,average_respiration,min_respiration,max_respiration) > 0),
  add constraint activity_speed_order check (average_speed_kmh <= max_speed_kmh and moving_speed_kmh <= max_speed_kmh),
  add constraint activity_respiration_order check (min_respiration <= average_respiration and average_respiration <= max_respiration and min_respiration <= max_respiration);

comment on column public.match_player_activity.duration_minutes is 'Total activity timer / playing time in minutes. May contain fractional minutes to preserve watch seconds.';
comment on column public.match_player_activity.moving_minutes is 'Watch-reported moving time, separate from the activity timer and elapsed time.';
comment on column public.match_player_activity.elapsed_minutes is 'Watch-reported elapsed time including pauses, in minutes.';
comment on column public.match_player_activity.calories is 'Watch estimate in kcal, as supplied. May be influenced by the configured wearer profile; not a fitness score.';
comment on column public.match_player_activity.average_pace_seconds is 'Watch-reported seconds per kilometre, not inferred from rounded speed or distance.';
comment on column public.match_player_activity.max_speed_kmh is 'Watch-reported peak km/h, not a validated sprint test.';
comment on column public.match_player_activity.average_respiration is 'Watch-reported average breaths per minute during this activity; not resting respiration.';

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
    'average_pace_seconds',case when p_values ? 'average_pace_seconds' then (p_values->>'average_pace_seconds')::integer else existing.average_pace_seconds end,
    'moving_pace_seconds',case when p_values ? 'moving_pace_seconds' then (p_values->>'moving_pace_seconds')::integer else existing.moving_pace_seconds end,
    'best_pace_seconds',case when p_values ? 'best_pace_seconds' then (p_values->>'best_pace_seconds')::integer else existing.best_pace_seconds end,
    'average_speed_kmh',case when p_values ? 'average_speed_kmh' then (p_values->>'average_speed_kmh')::numeric else existing.average_speed_kmh end,
    'moving_speed_kmh',case when p_values ? 'moving_speed_kmh' then (p_values->>'moving_speed_kmh')::numeric else existing.moving_speed_kmh end,
    'max_speed_kmh',case when p_values ? 'max_speed_kmh' then (p_values->>'max_speed_kmh')::numeric else existing.max_speed_kmh end,
    'moving_minutes',case when p_values ? 'moving_minutes' then (p_values->>'moving_minutes')::numeric else existing.moving_minutes end,
    'elapsed_minutes',case when p_values ? 'elapsed_minutes' then (p_values->>'elapsed_minutes')::numeric else existing.elapsed_minutes end,
    'calories',case when p_values ? 'calories' then (p_values->>'calories')::integer else existing.calories end,
    'average_respiration',case when p_values ? 'average_respiration' then (p_values->>'average_respiration')::integer else existing.average_respiration end,
    'min_respiration',case when p_values ? 'min_respiration' then (p_values->>'min_respiration')::integer else existing.min_respiration end,
    'max_respiration',case when p_values ? 'max_respiration' then (p_values->>'max_respiration')::integer else existing.max_respiration end,
    'source',btrim(coalesce(p_values->>'source','')));
  if existing.version is not null and desired=jsonb_build_object('distance_km',existing.distance_km,
    'duration_minutes',existing.duration_minutes,'average_heart_rate',existing.average_heart_rate,
    'max_heart_rate',existing.max_heart_rate,'resting_heart_rate',existing.resting_heart_rate,
    'personal_max_heart_rate',existing.personal_max_heart_rate,'average_pace_seconds',existing.average_pace_seconds,'moving_pace_seconds',existing.moving_pace_seconds,'best_pace_seconds',existing.best_pace_seconds,'average_speed_kmh',existing.average_speed_kmh,'moving_speed_kmh',existing.moving_speed_kmh,'max_speed_kmh',existing.max_speed_kmh,'moving_minutes',existing.moving_minutes,'elapsed_minutes',existing.elapsed_minutes,'calories',existing.calories,'average_respiration',existing.average_respiration,'min_respiration',existing.min_respiration,'max_respiration',existing.max_respiration,'source',existing.source) then return to_jsonb(existing); end if;
  if existing.version is distinct from p_expected_version then raise exception 'Activity changed while editing' using errcode='40001'; end if;
  if existing.version is null then
    insert into public.match_player_activity(league_id,match_id,player_id,distance_km,duration_minutes,average_heart_rate,max_heart_rate,resting_heart_rate,personal_max_heart_rate,average_pace_seconds,moving_pace_seconds,best_pace_seconds,average_speed_kmh,moving_speed_kmh,max_speed_kmh,moving_minutes,elapsed_minutes,calories,average_respiration,min_respiration,max_respiration,source)
      values(p_league_id,p_match_id,p_player_id,(desired->>'distance_km')::numeric,(desired->>'duration_minutes')::numeric,
        (desired->>'average_heart_rate')::integer,(desired->>'max_heart_rate')::integer,
        (desired->>'resting_heart_rate')::integer,(desired->>'personal_max_heart_rate')::integer,(desired->>'average_pace_seconds')::integer,(desired->>'moving_pace_seconds')::integer,(desired->>'best_pace_seconds')::integer,(desired->>'average_speed_kmh')::numeric,(desired->>'moving_speed_kmh')::numeric,(desired->>'max_speed_kmh')::numeric,(desired->>'moving_minutes')::numeric,(desired->>'elapsed_minutes')::numeric,(desired->>'calories')::integer,(desired->>'average_respiration')::integer,(desired->>'min_respiration')::integer,(desired->>'max_respiration')::integer,desired->>'source') returning * into saved;
  else
    update public.match_player_activity set distance_km=(desired->>'distance_km')::numeric,duration_minutes=(desired->>'duration_minutes')::numeric,
      average_heart_rate=(desired->>'average_heart_rate')::integer,max_heart_rate=(desired->>'max_heart_rate')::integer,
      resting_heart_rate=(desired->>'resting_heart_rate')::integer,personal_max_heart_rate=(desired->>'personal_max_heart_rate')::integer,
      average_pace_seconds=(desired->>'average_pace_seconds')::integer,
      moving_pace_seconds=(desired->>'moving_pace_seconds')::integer,
      best_pace_seconds=(desired->>'best_pace_seconds')::integer,
      average_speed_kmh=(desired->>'average_speed_kmh')::numeric,
      moving_speed_kmh=(desired->>'moving_speed_kmh')::numeric,
      max_speed_kmh=(desired->>'max_speed_kmh')::numeric,
      moving_minutes=(desired->>'moving_minutes')::numeric,
      elapsed_minutes=(desired->>'elapsed_minutes')::numeric,
      calories=(desired->>'calories')::integer,
      average_respiration=(desired->>'average_respiration')::integer,
      min_respiration=(desired->>'min_respiration')::integer,
      max_respiration=(desired->>'max_respiration')::integer,source=desired->>'source'
      where league_id=p_league_id and match_id=p_match_id and player_id=p_player_id returning * into saved;
  end if;
  return to_jsonb(saved);
end $fn$;
revoke all on function public.save_match_player_activity(bigint,bigint,bigint,uuid,jsonb,boolean) from public,anon;
grant execute on function public.save_match_player_activity(bigint,bigint,bigint,uuid,jsonb,boolean) to authenticated;
notify pgrst, 'reload schema';
