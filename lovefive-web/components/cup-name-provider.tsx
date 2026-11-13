"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { cupName, DEFAULT_CUP_NAME, getCompetitionLabels } from "@/lib/competition";

const CupNameContext = createContext({ name: DEFAULT_CUP_NAME, setName: (_name: string) => {} });

export function CupNameProvider({ name, children }: { name?: string | null; children: React.ReactNode }) {
  const [currentName, setName] = useState(cupName(name));
  useEffect(() => { setName(cupName(name)); }, [name]);
  return <CupNameContext.Provider value={{ name: currentName, setName }}>{children}</CupNameContext.Provider>;
}

export function useCupName() { return useContext(CupNameContext); }
export function useCompetitionLabels() { return getCompetitionLabels(useCupName().name); }
