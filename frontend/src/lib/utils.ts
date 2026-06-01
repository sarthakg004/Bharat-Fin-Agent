// Tiny utility helpers — no logic here that the components couldn't host themselves;
// kept here so the components stay tidy.

export function cls(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

export function timeAgo(iso: string): string {
  const d = new Date(iso).getTime();
  if (Number.isNaN(d)) return "";
  const s = Math.max(0, Math.round((Date.now() - d) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} hr ago`;
  return new Date(iso).toLocaleDateString();
}

export function isMac(): boolean {
  if (typeof navigator === "undefined") return false;
  return navigator.platform.toLowerCase().includes("mac");
}

export function modKey(): string {
  return isMac() ? "⌘" : "Ctrl";
}

/** Parse "[1]" / "[2,3]" markers in answer text → array of segments. */
export type AnswerSegment =
  | { kind: "text"; text: string }
  | { kind: "cite"; ids: number[] };

export function parseAnswerCitations(text: string): AnswerSegment[] {
  const out: AnswerSegment[] = [];
  const re = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push({ kind: "text", text: text.slice(last, m.index) });
    const ids = m[1].split(",").map(s => parseInt(s.trim(), 10)).filter(n => !Number.isNaN(n));
    out.push({ kind: "cite", ids });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

/** Map a score in [0,1] to one of "good" | "okay" | "bad" for badge colours. */
export function scoreTier(score: number | null | undefined): "good" | "okay" | "bad" | "none" {
  if (score == null || Number.isNaN(score)) return "none";
  if (score >= 0.8) return "good";
  if (score >= 0.6) return "okay";
  return "bad";
}
