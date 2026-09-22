grant select on public.matches to authenticated;
grant select on public.mmr_history to authenticated;
grant insert, update, delete on public.matches to authenticated;
grant insert, update, delete on public.mmr_history to authenticated;

drop policy if exists matches_league_select on public.matches;
create policy matches_league_select
on public.matches
for select to authenticated
using (
  exists (
    select 1
    from public.league_members lm
    where lm.league_id = matches.league_id
      and lm.user_id = auth.uid()
      and lm.status = 'active'
  )
);

drop policy if exists mmr_history_league_select on public.mmr_history;
create policy mmr_history_league_select
on public.mmr_history
for select to authenticated
using (
  exists (
    select 1
    from public.league_members lm
    where lm.league_id = mmr_history.league_id
      and lm.user_id = auth.uid()
      and lm.status = 'active'
  )
);

drop policy if exists matches_league_admin_write on public.matches;
create policy matches_league_admin_write
on public.matches
for all to authenticated
using (
  exists (
    select 1
    from public.league_members lm
    where lm.league_id = matches.league_id
      and lm.user_id = auth.uid()
      and lm.status = 'active'
      and lm.role in ('owner', 'admin')
  )
)
with check (
  exists (
    select 1
    from public.league_members lm
    where lm.league_id = matches.league_id
      and lm.user_id = auth.uid()
      and lm.status = 'active'
      and lm.role in ('owner', 'admin')
  )
);

drop policy if exists mmr_history_league_admin_write on public.mmr_history;
create policy mmr_history_league_admin_write
on public.mmr_history
for all to authenticated
using (
  exists (
    select 1
    from public.league_members lm
    where lm.league_id = mmr_history.league_id
      and lm.user_id = auth.uid()
      and lm.status = 'active'
      and lm.role in ('owner', 'admin')
  )
)
with check (
  exists (
    select 1
    from public.league_members lm
    where lm.league_id = mmr_history.league_id
      and lm.user_id = auth.uid()
      and lm.status = 'active'
      and lm.role in ('owner', 'admin')
  )
);;
