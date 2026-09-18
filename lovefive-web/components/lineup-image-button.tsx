"use client";

import { useEffect, useId, useRef, useState } from "react";
import { Download, X } from "lucide-react";
import { createLineupImage, type LineupImageData } from "@/lib/lineup-image";

export function LineupImageButton({ data, renderImage }: { data: LineupImageData; renderImage?: () => Promise<{ blob: Blob; filename: string; width: number; height: number }> }) {
  const titleId = useId();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState<{ url: string; filename: string; width: number; height: number } | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const inProgress = useRef(false);
  const disposed = useRef(false);
  useEffect(() => { disposed.current = false; return () => { disposed.current = true; }; }, []);
  useEffect(() => {
    if (!preview) return;
    dialog.current?.showModal();
    return () => { URL.revokeObjectURL(preview.url); };
  }, [preview]);

  function closePreview() {
    dialog.current?.close();
    setPreview(null);
  }

  async function prepareImage() {
    if (inProgress.current) return;
    inProgress.current = true;
    setBusy(true);
    setError("");
    try {
      const result = await (renderImage ? renderImage() : createLineupImage(data));
      if (!disposed.current) setPreview({ url: URL.createObjectURL(result.blob), filename: result.filename, width: result.width, height: result.height });
    } catch (cause) {
      console.error("Lineup image export failed", cause);
      if (!disposed.current) setError("Could not create the lineup image. Refresh the page and try again.");
    } finally {
      inProgress.current = false;
      if (!disposed.current) setBusy(false);
    }
  }

  return <>
    <button className="ghost-button lineup-image-button" type="button" onClick={prepareImage} disabled={busy}>
      <Download size={16} aria-hidden="true" />{busy ? "Creating image…" : "Save lineup image"}
    </button>
    {error ? <span className="lineup-image-error" role="alert">{error}</span> : null}
    {preview ? <dialog ref={dialog} className="lineup-image-dialog" aria-labelledby={titleId} onCancel={event => { event.preventDefault(); closePreview(); }}>
      <div className="lineup-image-dialog-heading">
        <h3 id={titleId}>Ready for WhatsApp</h3>
        <button className="ghost-button" type="button" aria-label="Close image preview" onClick={closePreview}><X size={20} aria-hidden="true" /></button>
      </div>
      <p>Save this portrait image, then attach it in your chat.</p>
      {/* The generated local PNG is also the exact file offered for download. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img className="lineup-image-preview" src={preview.url} width={preview.width} height={preview.height}
        alt={`Lineups for ${data.date}. Blue: ${data.blue.join(", ")}. Red: ${data.red.join(", ")}. Blue win ${(data.forecast.win * 100).toFixed(1)}%, draw ${(data.forecast.draw * 100).toFixed(1)}%, red win ${(data.forecast.loss * 100).toFixed(1)}%.`} />
      <a className="primary-button lineup-image-download" href={preview.url} download={preview.filename}><Download size={18} aria-hidden="true" />Download image</a>
    </dialog> : null}
  </>;
}
