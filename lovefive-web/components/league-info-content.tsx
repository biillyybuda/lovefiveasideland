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
        <summary>Public MMR and team-balancing strength</summary>
        <p>
          Public MMR is the rolling rating shown in the league table, player records and rating history. The usual
          starting point is 1,000, although admins can set a different starting rating. Each completed game changes
          MMR using the win, draw or loss, the player&apos;s rating against the opposition average, team imbalance and
          rating volatility. The size of a win or defeat does not directly change public MMR.
        </p>
        <p>
          Matchday&apos;s default Team Generator uses a separate strength estimate, learnt from team scorelines. It
          tracks uncertainty as well as strength. Switching to this selector does not convert public MMR or rewrite
          rating history. Period MMR gain still means the movement in a player&apos;s public rating inside that period.
        </p>
      </details>

      <details open>
        <summary>Strength and uncertainty</summary>
        <p>
          Strength estimates are centred on 1,000. Everyone starts from the model&apos;s neutral estimate and the app
          learns from completed games in date order, using the difference between the scoreline and what it expected.
          Unusually large results have a reduced influence. A custom public starting MMR does not set this separate estimate.
        </p>
        <p>
          The ± number shows uncertainty in the estimate: one standard deviation, not a guaranteed range. A larger
          number means less certainty. Match evidence usually narrows uncertainty; time passing adds some uncertainty
          without lowering the strength estimate. Playing many games with the same teammates can still make individual
          contributions hard to separate, so the model also keeps track of those linked estimates.
        </p>
        <p>
          Team forecasts include this uncertainty and normal variation between games. A matchup with a level expected
          score can still have a modest chance of finishing close. Uncertainty is not a deduction from public MMR,
          an MVP penalty or a Most Improved bonus.
        </p>
      </details>

      <details>
        <summary>MVP and season rating</summary>
        <p>
          MVP is scored from 80 points of seeded season rating and 20 points of attendance. Record and rolling MMR
          change are shown as context, but they do not add extra MVP points.
        </p>
        <p>
          Season rating is recalculated for the selected season or period with the established award formula. The
          starting seed is 1,000 plus 35% of the difference between the player&apos;s starting public MMR and the average
          starting MMR of players involved in that period. The period&apos;s results are then replayed in date order.
          This award calculation is kept separate from the team selector so adopting a new model does not reshuffle the race.
        </p>
        <p>
          A season rating of 900 or below earns 0 rating points; 1,100 or above earns 80, with a straight scale between
          them. Attendance earns up to 20 points based on the share of games played. Both components are rounded.
          Players qualify with one appearance when the period has up to three matches; after that they need at least
          three appearances and 25% of the period&apos;s games, rounded up.
        </p>
        <p>
          Most Improved compares full-year and year-to-date MVP-style score with the same period last season. It uses
          the same season rating and attendance ingredients, then ranks the biggest year-on-year jump. Quarter filters
          switch to Quarter Surge instead, ranking who gained the most MMR inside that quarter. If there is no
          previous-season comparison with eligible players available, the page falls back to period MMR gain. Players
          must qualify in both periods for the year-on-year comparison. Year-to-date uses the latest recorded game in
          the selected year as its cutoff and compares the same calendar cutoff last year. New results can move the
          races naturally; the change of team selector adds no award points.
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
          Pick exactly ten available players and choose the match date. The Team Generator checks every unique
          five-versus-five split: 126 before side locks. It uses completed games on or before the selected date,
          including archived players in those games, to estimate each split&apos;s chance of finishing within two goals.
          A close game includes a draw, a one-goal win or a two-goal win.
        </p>
        <p>
          Options are grouped into one-percentage-point bands below the highest estimated close-game chance. Within
          each band, known positions and recent team variety decide the order, followed by the exact chance. Fitness,
          individual goals, assists, saves and chemistry statistics do not add separate bonuses to this model.
        </p>
        <p>
          You can lock players to Team A or Team B when someone needs to be kept with or away from a side. Matchday
          Memory lets you reuse the shape of an old game: returning players keep their old side and the model fills
          the spaces. Choosing a memory replaces your existing side locks. You can also lock your own two teams to
          see their forecast, choose an alternative recommendation and copy the teams to share.
        </p>
        <p>
          The Legacy selector and Legacy memory tabs retain the earlier rules for comparison. Those rules include
          public MMR, form, profiles and chemistry. Their scores and cards belong to that earlier selector.
        </p>
      </details>

      <details>
        <summary>Matchday Card</summary>
        <p>
          The default Matchday Card starts with a Matchday lineups header, the match date and two team lists:
          blue (Team A) and red (Team B), ready to share. A forecast strip underneath the lineups shows the chances of blue winning,
          a draw or red winning, so you can share just the teams or include the forecast as well.
          Use Save lineup image for a portrait image with larger names and the forecast included. Preview it,
          download it and attach the saved image in WhatsApp; there is no need to crop a website screenshot.
          A separate pair of figures shows how tight the score might be: a finish within two goals, or
          either team winning by three or more. A 30% chance of finishing within two goals is not a 30% chance of a
          particular team winning. The average expected goal difference is explained inside the uncertainty section;
          it is an average across possible results, not a promised scoreline.
          Team comparison bars and player cards below the shareable lineup show balancing strength, uncertainty and
          recent records. Team strength is the average of that side&apos;s five player strength estimates. Recent form
          and public MMR are context; they do not add extra bonuses to the forecast.
        </p>
        <p>
          These are model estimates whose accuracy needs checking against future games. The legacy card&apos;s
          closeness score is a rules-based score out of 100, so it cannot be compared directly with the new model&apos;s
          estimated percentage. Legacy projected scores, profile verdicts and swap suggestions use the earlier rules.
        </p>
      </details>

      <details>
        <summary>Lineup editor</summary>
        <p>
          Open a chosen matchup with Edit lineup, or choose Lineup editor in Matchday Hub. Drag one card onto
          another to swap them, or tap two cards in turn. Choose your keepers and lock their positions before
          using Suggest positions. Every available formation includes a goalkeeper, defender, midfielder and striker.
          Suggestions keep the same five players on each side and use preferred positions, chemistry links and
          recorded goals and assists. Those stats cannot establish who is a good defender or keeper; your choices
          and position locks guide the setup.
        </p>
        <p>
          A link compares the pair&apos;s win percentage together with the mean of their individual win percentages.
          More than five percentage points above is green, within five is amber, and more than five below is red.
          Draws count as games but not wins. Players with no shared games have a red link for zero recorded chemistry;
          tap a link to see the games behind it. This editor uses completed games up to the chosen date.
        </p>
        <p>
          Cards show public MMR. Moving players within a side does not alter the match forecast; swapping players
          between sides recalculates it with the existing strength and uncertainty model. Links and positions do not
          add forecast bonuses or change MMR, MVP or Most Improved points. Your draft saves on this device for this
          league. Save lineup image exports the current formation, colours, players, links and forecast as a portrait PNG.
        </p>
      </details>

      <details>
        <summary>Chemistry and rivalries</summary>
        <p>
          Chemistry measures whether players perform better together than their separate records suggest they should.
          The legacy Matchday chemistry adjustment starts after four shared games and reaches full sample weight
          after ten. Relationship tables remain useful for exploring the league; they are not extra inputs to the
          default strength and uncertainty selector.
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
          entries and recalculate stats. When match results are changed, the app rebuilds player records, streaks, public
          MMR and MMR history from completed games so later matches stay consistent. The Team Generator learns from the
          corrected history when it is next loaded. Record the teams that actually played, including any changes to a suggestion.
        </p>
      </details>
    </section>
  );
}
