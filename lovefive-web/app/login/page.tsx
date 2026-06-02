import Link from "next/link";

export default function LoginPage() {
  return (
    <main className="main">
      <div className="page-head">
        <div>
          <div className="eyebrow">Account</div>
          <h1>Sign in to Love Five</h1>
          <p className="lead">Auth will move here after the demo website is proven.</p>
        </div>
      </div>

      <div className="panel" style={{ maxWidth: 520 }}>
        <p className="muted">
          For now, use the current Streamlit app to sign in and manage leagues. This Next.js version
          is being built around the public demo first.
        </p>
        <div className="button-row">
          <Link className="button primary" href="/demo">
            View demo league
          </Link>
          <a className="button" href="https://lovefive.co.uk">
            Open current app
          </a>
        </div>
      </div>
    </main>
  );
}
