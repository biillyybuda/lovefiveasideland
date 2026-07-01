"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type SetupStep = {
  href: string;
  id: "players" | "result" | "invite";
  label: string;
  title: string;
};

const steps: SetupStep[] = [
  {
    id: "players",
    label: "1",
    title: "Add your regular players",
    href: "/app/admin-tools/player-management"
  },
  {
    id: "result",
    label: "2",
    title: "Record your first score",
    href: "/app/admin-tools/add-result"
  },
  {
    id: "invite",
    label: "3",
    title: "Share the league code",
    href: "/app/join-invite"
  }
];

export function SetupChecklist({
  leagueId,
  enabled,
  currentPath,
  playersCount,
  matchesCount
}: {
  leagueId: number;
  enabled: boolean;
  currentPath: string;
  playersCount: number;
  matchesCount: number;
}) {
  const storageKey = `lovefive-setup-dismissed-${leagueId}`;
  const [dismissed, setDismissed] = useState(true);
  const [open, setOpen] = useState(true);

  useEffect(() => {
    if (!enabled || typeof window === "undefined") return;
    setDismissed(window.localStorage.getItem(storageKey) === "1");
  }, [enabled, storageKey]);

  const complete = useMemo(
    () => ({
      players: playersCount > 0,
      result: matchesCount > 0,
      invite: false
    }),
    [matchesCount, playersCount]
  );
  const completeCount = Number(complete.players) + Number(complete.result);

  if (!enabled || dismissed) return null;

  function dismiss() {
    window.localStorage.setItem(storageKey, "1");
    setDismissed(true);
  }

  return (
    <section className={open ? "setup-checklist open" : "setup-checklist"}>
      <button className="setup-checklist-toggle" type="button" onClick={() => setOpen((value) => !value)}>
        <span>League setup</span>
        <strong>Welcome to your league</strong>
        <small>{completeCount} of 3 done</small>
      </button>

      {open ? (
        <div className="setup-checklist-body">
          <p>Set up the basics first: players, first result, then invite the group.</p>
          <div className="setup-checklist-steps">
            {steps.map((step) => {
              const active = currentPath === step.href;
              const done = complete[step.id];
              return (
                <Link className={active ? "setup-step active" : done ? "setup-step done" : "setup-step"} href={step.href} key={step.id}>
                  <span>{done ? "✓" : step.label}</span>
                  <strong>{step.title}</strong>
                </Link>
              );
            })}
          </div>
          <div className="setup-checklist-actions">
            <button className="button compact-button" type="button" onClick={dismiss}>
              Hide checklist
            </button>
          </div>
        </div>
      ) : null}
    </section>
  );
}
