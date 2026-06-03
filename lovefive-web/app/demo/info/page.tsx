import { AppShell } from "@/components/app-shell";
import { LeagueInfoContent } from "@/components/league-info-content";

export default function DemoInfoPage() {
  return (
    <AppShell active="info">
      <div className="page-head">
        <div>
          <h1>Info</h1>
        </div>
      </div>

      <LeagueInfoContent demo />
    </AppShell>
  );
}
