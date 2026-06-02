import { AuthenticatedLeagueHome } from "@/components/authenticated-league-home";

export const dynamic = "force-dynamic";

export default function AppHomePage() {
  return <AuthenticatedLeagueHome />;
}
