-- Optional watch/app readings, separate from result and rating calculations.
create table public.match_player_activity (
  league_id bigint not null references public.leagues(id) on delete cascade,
  match_id bigint not null references public.matches(id) on delete cascade,
  player_id bigint not null references public.players(id) on delete cascade,
  distance_km numeric check (distance_km >= 0 and distance_km <= 100),
  duration_minutes numeric check (duration_minutes >= 0.01 and duration_minutes <= 600),
  average_heart_rate integer check (average_heart_rate between 1 and 300),
  max_heart_rate integer check (max_heart_rate between 1 and 300),
  source text not null default '' check (char_length(source) <= 80),
  version uuid not null default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  created_by uuid default auth.uid(),
  updated_by uuid default auth.uid(),
  primary key (match_id,player_id),
  constraint activity_has_reading check (num_nonnulls(distance_km,duration_minutes,average_heart_rate,max_heart_rate) > 0),
  constraint activity_heart_rate_order check (average_heart_rate <= max_heart_rate)
);
create index match_player_activity_league on public.match_player_activity(league_id,match_id);
create index match_player_activity_player on public.match_player_activity(player_id);
alter table public.match_player_activity enable row level security;
revoke all on public.match_player_activity from anon;
grant select,insert,update,delete on public.match_player_activity to authenticated;
create policy activity_member_read on public.match_player_activity for select to authenticated
  using (exists(select 1 from public.league_members lm where lm.league_id=match_player_activity.league_id
    and lm.user_id=(select auth.uid()) and lm.status='active'));
create policy activity_admin_write on public.match_player_activity for all to authenticated
  using (exists(select 1 from public.league_members lm where lm.league_id=match_player_activity.league_id
    and lm.user_id=(select auth.uid()) and lm.status='active' and lm.role in ('owner','admin')))
  with check (exists(select 1 from public.league_members lm where lm.league_id=match_player_activity.league_id
    and lm.user_id=(select auth.uid()) and lm.status='active' and lm.role in ('owner','admin')));

create function public.validate_match_player_activity() returns trigger
language plpgsql security invoker set search_path='' as $fn$
declare game public.matches; participant public.players;
begin
  perform pg_advisory_xact_lock(hashtextextended('lovefive-ratings:'||NEW.league_id::text,0));
  if TG_OP='UPDATE' and (NEW.league_id,NEW.match_id,NEW.player_id) is distinct from (OLD.league_id,OLD.match_id,OLD.player_id) then
    raise exception 'Activity cannot be moved to another match or player' using errcode='23514';
  end if;
  select * into game from public.matches where id=NEW.match_id and league_id=NEW.league_id and processed=1;
  if not found then raise exception 'Choose a completed match in this league' using errcode='23514'; end if;
  select * into participant from public.players where id=NEW.player_id and league_id=NEW.league_id;
  if not found then raise exception 'Player does not belong to this league' using errcode='23514'; end if;
  if not exists(select 1 from unnest(string_to_array(coalesce(game.team_a,'')||','||coalesce(game.team_b,''),',')) n
    where lower(regexp_replace(btrim(n),'\s+',' ','g')) in
      (lower(regexp_replace(btrim(participant.name),'\s+',' ','g')),nullif(lower(regexp_replace(btrim(participant.display_name),'\s+',' ','g')),''))) then
    raise exception 'Choose a player from this match''s recorded lineup' using errcode='23514';
  end if;
  NEW.source:=btrim(NEW.source);
  NEW.version:=gen_random_uuid();
  NEW.updated_at:=clock_timestamp();
  NEW.updated_by:=auth.uid();
  if TG_OP='INSERT' then NEW.created_by:=auth.uid(); NEW.created_at:=NEW.updated_at;
  else NEW.created_by:=OLD.created_by; NEW.created_at:=OLD.created_at; end if;
  return NEW;
end $fn$;
revoke all on function public.validate_match_player_activity() from public,anon,authenticated;
create trigger validate_match_player_activity before insert or update on public.match_player_activity
  for each row execute function public.validate_match_player_activity();

create function public.save_match_player_activity(p_league_id bigint,p_match_id bigint,p_player_id bigint,
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
  -- JSON conversion is also used for retry equality; omitted figures stay unknown.
  desired:=jsonb_build_object('distance_km',(p_values->>'distance_km')::numeric,
    'duration_minutes',(p_values->>'duration_minutes')::numeric,
    'average_heart_rate',(p_values->>'average_heart_rate')::integer,
    'max_heart_rate',(p_values->>'max_heart_rate')::integer,'source',btrim(coalesce(p_values->>'source','')));
  if existing.version is not null and desired=jsonb_build_object('distance_km',existing.distance_km,
    'duration_minutes',existing.duration_minutes,'average_heart_rate',existing.average_heart_rate,
    'max_heart_rate',existing.max_heart_rate,'source',existing.source) then return to_jsonb(existing); end if;
  if existing.version is distinct from p_expected_version then raise exception 'Activity changed while editing' using errcode='40001'; end if;
  if existing.version is null then
    insert into public.match_player_activity(league_id,match_id,player_id,distance_km,duration_minutes,average_heart_rate,max_heart_rate,source)
      values(p_league_id,p_match_id,p_player_id,(desired->>'distance_km')::numeric,(desired->>'duration_minutes')::numeric,
        (desired->>'average_heart_rate')::integer,(desired->>'max_heart_rate')::integer,desired->>'source') returning * into saved;
  else
    update public.match_player_activity set distance_km=(desired->>'distance_km')::numeric,duration_minutes=(desired->>'duration_minutes')::numeric,
      average_heart_rate=(desired->>'average_heart_rate')::integer,max_heart_rate=(desired->>'max_heart_rate')::integer,source=desired->>'source'
      where league_id=p_league_id and match_id=p_match_id and player_id=p_player_id returning * into saved;
  end if;
  return to_jsonb(saved);
end $fn$;
revoke all on function public.save_match_player_activity(bigint,bigint,bigint,uuid,jsonb,boolean) from public,anon;
grant execute on function public.save_match_player_activity(bigint,bigint,bigint,uuid,jsonb,boolean) to authenticated;
notify pgrst, 'reload schema';
