"use client";

import { useId } from "react";

export function MatchDatePicker({ date, onChange }: { date: string; onChange: (date: string) => void }) {
  const id = useId();

  function chooseTomorrow() {
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    onChange(`${tomorrow.getFullYear()}-${String(tomorrow.getMonth() + 1).padStart(2, "0")}-${String(tomorrow.getDate()).padStart(2, "0")}`);
  }

  return <div className="match-date-picker">
    <label htmlFor={id}>Game date</label>
    <div className="match-date-picker-inputs">
      <input id={id} type="date" required value={date} onChange={event => {
        if (event.currentTarget.validity.valid) onChange(event.currentTarget.value);
      }} />
      <button className="ghost-button" type="button" onClick={chooseTomorrow}>Tomorrow</button>
    </div>
  </div>;
}
