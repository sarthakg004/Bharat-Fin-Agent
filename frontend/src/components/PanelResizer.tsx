import { useRef } from "react";

interface Props {
  /** Current width of the panel being resized (px). */
  width: number;
  /** Called with the desired new width whenever the user drags. */
  onResize: (px: number) => void;
  /** Which side of the chat panel this resizer lives on. The sidebar resizer
   *  is on its right edge — dragging right widens it. The citations resizer
   *  is on its left edge — dragging right shrinks it. */
  side: "left" | "right";
  /** Tooltip text. */
  title?: string;
}

/**
 * 4px-wide vertical drag handle. Hovering shows an accent line; while dragging
 * the body cursor switches to col-resize and text selection is suppressed.
 * Window-level mouse listeners are used so the drag continues correctly even
 * when the pointer overshoots the handle itself.
 */
export function PanelResizer({ width, onResize, side, title }: Props) {
  const startRef = useRef({ x: 0, w: width });

  function onMouseMove(e: MouseEvent) {
    const delta = e.clientX - startRef.current.x;
    // sidebar (left): dragging right (+x) widens the sidebar
    // citations (right): dragging right (+x) shrinks the citations panel
    const next = side === "left"
      ? startRef.current.w + delta
      : startRef.current.w - delta;
    onResize(next);
  }

  function stop() {
    window.removeEventListener("mousemove", onMouseMove);
    window.removeEventListener("mouseup", stop);
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }

  function onMouseDown(e: React.MouseEvent) {
    e.preventDefault();
    startRef.current = { x: e.clientX, w: width };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", stop);
  }

  function onDoubleClick() {
    // Reset to the default width on double-click — matches Tweetdeck / Linear.
    onResize(side === "left" ? 240 : 340);
  }

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      title={title || "Drag to resize · double-click to reset"}
      onMouseDown={onMouseDown}
      onDoubleClick={onDoubleClick}
      className="group relative z-10 hidden h-full w-[4px] shrink-0 cursor-col-resize bg-transparent md:block"
    >
      {/* Visible bar on hover / active drag. */}
      <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-px bg-border-subtle transition-colors group-hover:bg-accent group-active:bg-accent" />
    </div>
  );
}
