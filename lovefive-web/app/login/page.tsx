import { Suspense } from "react";
import { LoginPanel } from "@/components/login-panel";

export const dynamic = "force-dynamic";

export default function LoginPage() {
  return (
    <main className="main">
      <Suspense fallback={<div className="panel">Loading sign in...</div>}>
        <LoginPanel />
      </Suspense>
    </main>
  );
}
