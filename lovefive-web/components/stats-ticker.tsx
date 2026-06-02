export function StatsTicker({ items }: { items: string[] }) {
  const cleanItems = items.filter(Boolean);
  if (!cleanItems.length) return null;

  const loop = [...cleanItems, ...cleanItems];

  return (
    <div className="stats-ticker" aria-label="League stories">
      <div className="stats-ticker-track">
        {loop.map((item, index) => (
          <span key={`${item}-${index}`}>{item}</span>
        ))}
      </div>
    </div>
  );
}
