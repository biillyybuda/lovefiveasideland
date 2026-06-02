import Image from "next/image";

export function BrandMark({ compact = false }: { compact?: boolean }) {
  return (
    <span className={`brand-mark ${compact ? "compact" : ""}`} aria-label="love5.co.uk">
      <Image
        src="/love5-brand-emblem.png"
        alt="love5.co.uk"
        width={375}
        height={375}
        priority
      />
      {!compact && (
        <span className="brand-text">
          <strong>love<span>5</span></strong>
          <small>.co.uk</small>
        </span>
      )}
    </span>
  );
}
