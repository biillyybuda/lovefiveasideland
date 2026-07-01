import { Suspense } from "react";
import { LoginPanel } from "@/components/login-panel";

export const dynamic = "force-dynamic";

export default function LoginPage() {
  return (
    <main className="main">
      <Suspense fallback={<div className="app-loading compact"><span>Love Five</span><strong>Opening account</strong></div>}>
        <LoginPanel />
      </Suspense>
    </main>
  );
}
