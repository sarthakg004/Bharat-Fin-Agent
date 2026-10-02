/**
 * MarkdownAnswer — renders the model's answer as GitHub-flavoured markdown
 * (headings, bullets, tables, bold, code) and turns `[N]` / `[N, M]` citation
 * markers into clickable chips that highlight the matching ChunkCard in the
 * right-hand panel.
 *
 * Implementation notes
 * --------------------
 * - We use react-markdown + remark-gfm for the rendering pipeline.
 * - A small remark plugin walks text nodes and replaces `[N]` patterns with
 *   `link` nodes carrying a `data-cite` attribute. We then provide a custom
 *   `a` component that, when it sees `data-cite`, renders our citation chip
 *   instead of a normal hyperlink.
 * - We intentionally do NOT linkify URLs in prose — the synth is told to
 *   keep URLs out; they live in the sidebar.
 */

import { type ComponentPropsWithoutRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { visit, SKIP } from "unist-util-visit";
import type { Plugin } from "unified";
import type { Root, Text, Link, PhrasingContent } from "mdast";

import { useChatStore } from "@/store/chatStore";

// Accept `[N]` and the full-width `【N】` some models write instead.
const CITATION_RE = /\[(\d+(?:\s*,\s*\d+)*)\]|【(\d+(?:\s*,\s*\d+)*)】/g;

/** Remark plugin: text nodes → text + link("#cite:N,M") segments. */
const remarkCitations: Plugin<[], Root> = () => (tree) => {
  visit(tree, "text", (node: Text, index, parent) => {
    if (!parent || typeof index !== "number") return;
    const value = node.value;
    if (!value.includes("[") && !value.includes("【")) return;
    CITATION_RE.lastIndex = 0;

    const segments: PhrasingContent[] = [];
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = CITATION_RE.exec(value)) !== null) {
      if (m.index > last) {
        segments.push({ type: "text", value: value.slice(last, m.index) });
      }
      // One of the two capture groups will match (ASCII or CJK brackets).
      const inner = m[1] ?? m[2] ?? "";
      const ids = inner.split(",").map((s) => parseInt(s.trim(), 10));
      const link: Link = {
        type: "link",
        url: `#cite:${ids.join(",")}`,
        title: null,
        children: [{ type: "text", value: m[0] }],
      };
      segments.push(link);
      last = m.index + m[0].length;
    }
    if (segments.length === 0) return;
    if (last < value.length) segments.push({ type: "text", value: value.slice(last) });

    parent.children.splice(index, 1, ...segments);
    return [SKIP, index + segments.length];
  });
};

/**
 * Math in an answer. A single `$` is money here ("$604 million"), so only
 * `$$...$$` counts as math (remark-math with singleDollarTextMath off).
 * Text already inside `$$...$$` is left alone; elsewhere, before parsing:
 * `\\(...\\)` and `\\[...\\]` become `$$...$$`, a lone `$...$` holding a LaTeX
 * command becomes `$$...$$`, and a money amount wrapped as math (`$\\$10,069$`)
 * goes back to text. An escaped `\\$` is never a delimiter.
 */
export function normalizeMath(text: string): string {
  return text
    .split(/(\$\$[\s\S]+?\$\$)/)
    .map((part, i) => (i % 2 ? part : normalizeOutsideMath(part)))
    .join("");
}

function normalizeOutsideMath(text: string): string {
  return text
    .replace(/(^|[^$\\])\$\\\$([^$\n\\]+?)\$(?!\$)/g, (_, pre, amount) => `${pre}$${amount}`)
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, m) => `\n$$\n${m.trim()}\n$$\n`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, m) => `$$${m.trim()}$$`)
    .replace(/(^|[^$\\])\$([^$\n]*\\[a-zA-Z]+[^$\n]*?)(?<!\\)\$(?!\$)/g, (_, pre, m) => `${pre}$$${m}$$`);
}

interface AProps extends ComponentPropsWithoutRef<"a"> {
  href?: string;
}

function MarkdownLink({ href, children, ...rest }: AProps) {
  // Citation chip path — href like "#cite:1,3"
  if (href?.startsWith("#cite:")) {
    const ids = href.slice("#cite:".length).split(",").map((s) => parseInt(s.trim(), 10));
    return <CitationChip ids={ids}>{children}</CitationChip>;
  }
  // Normal hyperlink — open externally
  return (
    <a href={href} target="_blank" rel="noreferrer" className="text-accent hover:underline" {...rest}>
      {children}
    </a>
  );
}

function CitationChip({ ids, children }: { ids: number[]; children: React.ReactNode }) {
  const setHighlight = useChatStore((s) => s.setHighlight);
  // Citations are 1-based in the prompt and in ChunkCard's display; the
  // ChunkCard wraps each chunk in a DOM element id `chunk-${chunk.id}` where
  // chunk.id is 0-based, so we subtract 1 when navigating.
  return (
    <a
      href={`#chunk-${ids[0] - 1}`}
      className="citation-mark"
      onClick={(e) => {
        e.preventDefault();
        const targetId = ids[0] - 1;
        setHighlight(targetId);
        document.getElementById(`chunk-${targetId}`)
          ?.scrollIntoView({ behavior: "smooth", block: "center" });
      }}
    >
      {children}
    </a>
  );
}

// Tailwind classes for each markdown element — terminal noir, sharp corners.
const COMPONENTS = {
  a: MarkdownLink,
  p:  (p: ComponentPropsWithoutRef<"p">) =>
        <p className="my-2 first:mt-0 last:mb-0" {...p} />,
  h1: (p: ComponentPropsWithoutRef<"h1">) =>
        <h1 className="font-display text-[20px] mt-4 mb-2 first:mt-0" {...p} />,
  h2: (p: ComponentPropsWithoutRef<"h2">) =>
        <h2 className="font-display text-[17px] mt-4 mb-2 first:mt-0" {...p} />,
  h3: (p: ComponentPropsWithoutRef<"h3">) =>
        <h3 className="font-ui font-semibold text-[14px] mt-3 mb-1 text-text-primary first:mt-0" {...p} />,
  ul: (p: ComponentPropsWithoutRef<"ul">) =>
        <ul className="my-2 ml-5 list-disc space-y-1" {...p} />,
  ol: (p: ComponentPropsWithoutRef<"ol">) =>
        <ol className="my-2 ml-5 list-decimal space-y-1" {...p} />,
  li: (p: ComponentPropsWithoutRef<"li">) =>
        <li className="pl-1" {...p} />,
  strong: (p: ComponentPropsWithoutRef<"strong">) =>
        <strong className="font-semibold text-text-primary" {...p} />,
  em: (p: ComponentPropsWithoutRef<"em">) =>
        <em className="italic" {...p} />,
  code: ({ children, ...p }: ComponentPropsWithoutRef<"code">) => (
    <code
      className="font-mono text-[12px] bg-bg-elevated border border-border-subtle px-1 py-[1px] text-text-primary"
      {...p}
    >
      {children}
    </code>
  ),
  pre: (p: ComponentPropsWithoutRef<"pre">) => (
    <pre
      className="my-2 overflow-x-auto bg-bg-elevated border border-border-subtle p-3 font-mono text-[12px]"
      {...p}
    />
  ),
  table: (p: ComponentPropsWithoutRef<"table">) => (
    <div className="my-3 overflow-x-auto">
      <table
        className="w-full border-collapse border border-border-subtle font-ui text-[12.5px]"
        {...p}
      />
    </div>
  ),
  thead: (p: ComponentPropsWithoutRef<"thead">) =>
    <thead className="bg-bg-elevated text-text-secondary" {...p} />,
  th: (p: ComponentPropsWithoutRef<"th">) => (
    <th
      className="border border-border-subtle px-3 py-1.5 text-left font-mono text-[10px] uppercase tracking-wider"
      {...p}
    />
  ),
  td: (p: ComponentPropsWithoutRef<"td">) => (
    <td className="border border-border-subtle px-3 py-1.5 align-top" {...p} />
  ),
  blockquote: (p: ComponentPropsWithoutRef<"blockquote">) => (
    <blockquote
      className="my-2 border-l-2 border-border-strong pl-3 italic text-text-secondary"
      {...p}
    />
  ),
  hr: () => <hr className="my-3 border-border-subtle" />,
};

interface Props {
  text: string;
}

export function MarkdownAnswer({ text }: Props) {
  return (
    <div className="font-ui text-[14px] leading-relaxed text-text-primary">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, [remarkMath, { singleDollarTextMath: false }], remarkCitations]}
        rehypePlugins={[rehypeKatex]}
        components={COMPONENTS}
      >
        {normalizeMath(text)}
      </ReactMarkdown>
    </div>
  );
}

/** A source passage (a filing excerpt, a computed metric) as markdown, so its
 *  tables and bold figures render. No citation chips: passages cite nothing. */
export function MarkdownSource({ text }: Props) {
  return (
    <div className="font-ui text-[12.5px] leading-relaxed text-text-primary">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
