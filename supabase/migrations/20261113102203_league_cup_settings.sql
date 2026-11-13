-- Display names are league settings, not competition IDs. Keep existing match
-- records and the immunity_shield rating policy unchanged.
begin;

create table public.league_cup_settings (
  league_id bigint primary key references public.leagues(id) on delete cascade,
  cup_name text not null default 'Slimfast Cup',
  constraint cup_name_valid check (
    char_length(cup_name) between 1 and 80
    and cup_name = btrim(cup_name)
    and cup_name !~ '[[:cntrl:]]'
    and cup_name ~ '[^[:space:]]'
  )
);

alter table public.league_cup_settings enable row level security;
revoke all on public.league_cup_settings from public, anon, authenticated;
grant select on public.league_cup_settings to authenticated;
grant insert (league_id, cup_name), update (league_id, cup_name) on public.league_cup_settings to authenticated;

create policy cup_settings_member_read on public.league_cup_settings
for select to authenticated using (
  exists (select 1 from public.league_members lm
    where lm.league_id = league_cup_settings.league_id
      and lm.user_id = (select auth.uid()) and lm.status = 'active')
);

create policy cup_settings_admin_insert on public.league_cup_settings
for insert to authenticated with check (
  exists (select 1 from public.league_members lm
    where lm.league_id = league_cup_settings.league_id
      and lm.user_id = (select auth.uid()) and lm.status = 'active'
      and lm.role in ('owner', 'admin'))
);

create policy cup_settings_admin_update on public.league_cup_settings
for update to authenticated using (
  exists (select 1 from public.league_members lm
    where lm.league_id = league_cup_settings.league_id
      and lm.user_id = (select auth.uid()) and lm.status = 'active'
      and lm.role in ('owner', 'admin'))
) with check (
  exists (select 1 from public.league_members lm
    where lm.league_id = league_cup_settings.league_id
      and lm.user_id = (select auth.uid()) and lm.status = 'active'
      and lm.role in ('owner', 'admin'))
);

notify pgrst, 'reload schema';
commit;
