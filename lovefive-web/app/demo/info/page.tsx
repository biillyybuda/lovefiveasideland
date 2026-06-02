import { AppShell } from "@/components/app-shell";

export default function DemoInfoPage() {
  return (
    <AppShell active="info">
      <div className="page-head">
        <div>
          <h1>Info</h1>
        </div>
      </div>

      <section className="info-grid">
        <details open>
          <summary>What Love Five does</summary>
          <p>
            Love Five turns weekly five-a-side results into a league hub. It tracks match history, player ratings,
            current form, team chemistry, rivalries and balanced team suggestions. The demo league is read-only so
            visitors can explore everything without changing real league data.
          </p>
        </details>

        <details open>
          <summary>How to use the site</summary>
          <p>
            Home gives the quick pulse of the league. Matchday Hub helps pick fair teams. Charts & Stats is the main
            analysis centre for players, matches and relationships. Season Review is for yearly awards and trends.
            Use the season filter when you want a specific year; use rolling/all years when you want the full history.
          </p>
        </details>

        <details>
          <summary>MMR rating</summary>
          <p>
            MMR is the player rating. Everyone starts from a 1000-style baseline. After each match, players gain or
            lose rating based on the result and how strong the opposition was expected to be. Beating a stronger side
            is rewarded more than beating a weaker side. Losing as the stronger side costs more.
          </p>
          <p>
            Season views reset the display around a season baseline, so a player&apos;s 2026 rating shows how they have
            performed in that period rather than only showing their lifetime number.
          </p>
        </details>

        <details>
          <summary>Form and records</summary>
          <p>
            Form is based on the previous five games, with the newest result carrying the most weight. A win is worth
            more than a draw, and a draw is worth more than a loss. If two players have similar form, goal difference
            and rating help break the tie.
          </p>
          <p>
            Season Review uses best form in the selected period instead of simply showing the final five games, so a
            great mid-season run still gets surfaced with the dates it happened.
          </p>
          <p>
            W-D-L means wins, draws and losses. Goal diff is goals for minus goals against while that player was in
            the team.
          </p>
        </details>

        <details>
          <summary>Matchday Hub</summary>
          <p>
            Pick the players who are available, then the engine tests possible team splits. It balances effective MMR,
            recent form, chemistry, experience, awkward pairings and repeat-risk from similar historic games. The
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
            Chemistry measures whether players perform better together than their separate records suggest they
            should. It starts affecting rankings after four shared games and reaches full confidence after ten, so a
            single good match does not outweigh a proven partnership. Rivalry measures repeated head-to-head matchups,
            especially when the record is competitive and the games are close.
          </p>
          <p>
            Lift means a teammate pairing performs better together than expected. Threat means an opponent makes the
            selected player perform worse than expected. Edge means the selected player performs better than expected
            against that opponent.
          </p>
          <p>
            Relationships has three modes: Head-to-head for two specific players, Teammates for shared lineups, and
            Rivalries for groups or players facing each other.
          </p>
        </details>

        <details>
          <summary>Player Insights</summary>
          <p>
            Player Insights is a single-player deep dive. Select a player to see their rating, MMR change, record,
            recent form, goal difference, best teammates, toughest opponents and every match they have appeared in.
          </p>
        </details>
      </section>
    </AppShell>
  );
}
