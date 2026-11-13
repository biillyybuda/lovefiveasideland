"use client";

import { useCompetitionLabels } from "./cup-name-provider";
import { Suspense, type ComponentProps } from "react";
import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import { competitionHref, competitionScope, type CompetitionScope } from "@/lib/competition";

export function useCompetition(initial?: string) {
  const params = useSearchParams();
  const scope = competitionScope(params?.get("competition") ?? initial);
  function change(value: CompetitionScope) {
    const url = new URL(window.location.href);
    url.searchParams.set("competition", value);
    window.history.replaceState(null, "", url);
  }
  return [scope, change] as const;
}

export function CompetitionControl({ value, onChange }: { value: CompetitionScope; onChange: (value: CompetitionScope) => void }) {
  const competitionLabels = useCompetitionLabels();
  return <label className="competition-control"><span>Competition</span>
    <select value={value} onChange={event => onChange(competitionScope(event.target.value))}>
      {Object.entries(competitionLabels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
    </select>
  </label>;
}

/** Carry scope through app navigation; switching the selected league deliberately starts at League. */
function ScopedLink({ href, ...props }: ComponentProps<typeof Link>) {
  const [scope] = useCompetition();
  const scoped = typeof href === "string" && /^\/(app|demo)(\/|\?|#|$)/.test(href) && !href.includes("competition=")
    ? competitionHref(href, scope) : href;
  return <Link {...props} href={scoped} />;
}
export function CompetitionLink(props: ComponentProps<typeof Link>) {
  return <Suspense fallback={<Link {...props}/>}><ScopedLink {...props}/></Suspense>;
}

/** Server-rendered summaries require navigation, rather than a shallow URL change. */
export function CompetitionNavigation({ value }: { value: CompetitionScope }) {
  const router = useRouter();
  return <CompetitionControl value={value} onChange={next => {
    const url = new URL(window.location.href);
    url.searchParams.set("competition", next);
    router.replace(`${url.pathname}${url.search}${url.hash}`);
  }}/>;
}
