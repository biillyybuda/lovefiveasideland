export function LeagueInfoContent({
  liveLeagueName,
  demo = false
}: {
  liveLeagueName?: string;
  demo?: boolean;
}) {
  return (
    <section className="info-grid">
      <details open>
        <summary>{liveLeagueName || "What Love Five does"}</summary>
        <p>
          Love Five turns weekly five-a-side results into a league hub. It tracks match history, player ratings,
          current form, team chemistry, rivalries, season awards and balanced team suggestions.
        </p>
        <p>
          {demo
            ? "The demo league is read-only so visitors can explore the app without changing real league data."
            : "Your live league is connected to the Love Five database. Admin tools are available to league admins from the profile menu."}
        </p>
      </details>

      <details open>
        <summary>How to use the site</summary>
        <p>
          Home gives the quick pulse of the league. Matchday Hub helps pick fair teams. Charts & Stats is the main
          analysis centre for players, matches and relationships. Season Review is for awards, yearly trends and
          sortable season tables.
        </p>
        <p>
          Use the season filter when you want a specific year or period. Use all seasons when you want the full rolling
          history.
        </p>
      </details>

      <details>
        <summary>MMR rating</summary>
        <p>
          MMR is the rolling player rating. Players start from a 1000-style baseline. After each match, players gain or
          lose rating based on the result and how strong the opposition was expected to be. Beating a stronger side is
          rewarded more than beating a weaker side. Losing as the stronger side costs more.
        </p>
        <p>
          The normal league table and Matchday Hub use rolling MMR because that is the best signal for team fairness.
          Season Review can also show period movement, so a player&apos;s MMR change shows how much their rolling rating
          moved inside the selected period.
        </p>
      </details>

      <details>
        <summary>MVP and season rating</summary>
        <p>
          MVP is scored from 80 points of seeded season rating and 20 points of attendance. Record and rolling MMR
          change are shown as context, but they do not add extra MVP points.
        </p>
        <p>
          Season rating is recalculated for the selected season or period. Each player is seeded from their rating at
          the start of that period, shrunk 35% toward the league average, then the period&apos;s matches are replayed in
          order with the normal MMR formula. This keeps the award mostly season-based while still recognising that
          beating known strong players early in the year should matter.
        </p>
        <p>
          Most Improved is separate. It ranks rolling MMR gain in the selected period, so comeback seasons and rating
          recoveries are recognised without taking over the MVP race.
        </p>
      </details>

      <details>
        <summary>Form and records</summary>
        <p>
          Form is based on recent results, with wins worth more than draws and draws worth more than losses. Season
          Review also surfaces the best form run inside the selected period, so a strong mid-season spell can still be
          recognised.
        </p>
        <p>
          W-D-L means wins, draws and losses. Goal diff is goals for minus goals against while that player was in the
          team.
        </p>
      </details>

      <details>
        <summary>Matchday Hub</summary>
        <p>
          Pick the players who are available, then the engine tests possible team splits. It balances effective rolling
          MMR, recent form, chemistry, experience, awkward pairings and repeat-risk from similar historic games. The
          recommended teams are ranked by the matchup most likely to produce a fair, competitive game.
        </p>
        <p>
          You can lock players to Team A or Team B when someone needs to be kept with or away from a side. Matchday
          Memory lets you reuse the shape of an old game and clearly shows who from that historic lineup is in or out.
        </p>
      </details>

      <details>
        <summary>Matchday Card</summary>
        <p>
          The Matchday Card explains a selected matchup. Projected score estimates the likely result. Match potential
          is the chance of a close game. Expected margin shows the predicted gap. Analyst verdict explains the green
          lights and watch-outs. Best tweak suggests a small swap when the engine finds a better balance.
        </p>
      </details>

      <details>
        <summary>Chemistry and rivalries</summary>
        <p>
          Chemistry measures whether players perform better together than their separate records suggest they should.
          It starts affecting rankings after four shared games and reaches full confidence after ten, so one good match
          does not outweigh a proven partnership.
        </p>
        <p>
          Rivalry measures repeated head-to-head matchups, especially when the record is competitive and the games are
          close. Lift means a teammate pairing performs better together than expected. Threat means an opponent makes a
          selected player perform worse than expected. Edge means the selected player performs better than expected
          against that opponent.
        </p>
      </details>

      <details>
        <summary>Player Insights</summary>
        <p>
          Player Insights is a single-player deep dive. Select a player to see their rating, MMR change, record,
          recent form, goal difference, best teammates, toughest opponents and every match they have appeared in.
        </p>
      </details>

      <details>
        <summary>Admin tools</summary>
        <p>
          League admins can add players, archive players, add results, correct match scores or dates, delete match
          entries and recalculate stats. When match results are changed, the app rebuilds player records, streaks, MMR
          and MMR history from the processed match list so later matches stay consistent.
        </p>
      </details>
    </section>
  );
}
