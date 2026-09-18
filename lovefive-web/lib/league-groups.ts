/** League-provided context. Membership uses stable player IDs, never name guesses. */
export type LeagueGroup = { name: string; description: string; playerIds: number[]; minimumTogether: number };

export function groupsForLeague(leagueId: number): LeagueGroup[] {
  if (leagueId !== 1) return [];
  return [{
    name: "HLHS",
    description: "Glee, Tom D, Kelso, Gav and Sam K went to HLHS together.",
    playerIds: [12, 2, 11, 10, 6],
    minimumTogether: 4
  }];
}
