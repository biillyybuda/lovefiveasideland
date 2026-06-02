import Link from "next/link";
import { BrandMark } from "@/components/brand-mark";

export default function HomePage() {
  return (
    <main className="main landing-main">
      <div className="topbar">
        <BrandMark />
        <div className="button-row" style={{ marginTop: 0 }}>
          <Link className="button" href="/login">
            Sign in
          </Link>
          <Link className="button primary" href="/demo">
            View demo
          </Link>
        </div>
      </div>

      <section className="hero landing-hero">
        <div className="landing-copy">
          <div className="eyebrow">Five-a-side intelligence</div>
          <h1>Stats for the games your group actually talks about.</h1>
          <p className="lead">
            Love Five turns your weekly football group into a live stats centre, with ratings, form,
            fair-team tools, chemistry, rivalries and season awards.
          </p>
          <div className="hero-actions">
            <Link className="button primary" href="/demo">
              Explore the demo league
            </Link>
            <Link className="button" href="/login">
              Sign in or create account
            </Link>
          </div>
          <div className="landing-trust-row">
            <span>MMR ratings</span>
            <span>AI teams</span>
            <span>Player insights</span>
            <span>Season review</span>
          </div>
        </div>

        <div className="hero-board landing-live-board">
          <div className="live-board-head">
            <div>
              <span>Demo league</span>
              <strong>Match centre</strong>
            </div>
            <em>Live-ready</em>
          </div>
          <div className="live-score-panel">
            <div className="live-team blue">
              <span>Team A</span>
              <strong>10</strong>
            </div>
            <div className="live-vs">
              <small>Latest result</small>
              <b>vs</b>
            </div>
            <div className="live-team red">
              <span>Team B</span>
              <strong>8</strong>
            </div>
          </div>
          <div className="mini-pitch">
            <span className="node a n1">1104</span>
            <span className="node a n2">1021</span>
            <span className="node a n3">1000</span>
            <span className="node b n4">1067</span>
            <span className="node b n5">1044</span>
            <span className="node b n6">994</span>
          </div>
          <div className="hero-cards">
            <div><span>MVP</span><strong>Callum P</strong><small>1104 MMR</small></div>
            <div><span>In form</span><strong>Isaac B</strong><small>Previous 5</small></div>
            <div><span>Duo</span><strong>5.4</strong><small>Chemistry score</small></div>
          </div>
        </div>
      </section>

      <section className="landing-feature-grid">
        <Link href="/demo/matchday">
          <span>Matchday Hub</span>
          <strong>Generate balanced teams before kick-off.</strong>
        </Link>
        <Link href="/demo/charts">
          <span>Charts & Stats</span>
          <strong>Track form, MMR, history and relationships.</strong>
        </Link>
        <Link href="/demo/season-review">
          <span>Season Review</span>
          <strong>Turn the year into awards and stories.</strong>
        </Link>
      </section>
    </main>
  );
}
