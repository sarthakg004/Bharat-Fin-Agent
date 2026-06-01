/**
 * ChartView — inline candlestick chart for market-data answers.
 *
 * Renders TradingView's open-source `lightweight-charts` library, styled to
 * match the Terminal Noir palette (sharp corners, accent green for up candles,
 * --error red for down). Volume bars are drawn on a secondary scale at the
 * bottom so the candle action stays readable.
 *
 * The chart auto-resizes with its container via ResizeObserver and tears
 * itself down cleanly when the message is unmounted.
 */

import { useEffect, useRef } from "react";
import { createChart, ColorType, CrosshairMode, type IChartApi } from "lightweight-charts";

import type { ChartSpec } from "@/lib/api";

interface Props {
  spec: ChartSpec;
}

export function ChartView({ spec }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;

    // Read CSS variables so the chart picks up tokens (no hard-coded hex here
    // beyond the up/down colours — those follow the spec's --accent / --error).
    const cssVar = (name: string) =>
      getComputedStyle(document.documentElement).getPropertyValue(name).trim() || undefined;

    const chart: IChartApi = createChart(node, {
      width: node.clientWidth,
      height: 280,
      autoSize: false,
      layout: {
        background: { type: ColorType.Solid, color: cssVar("--bg-surface") || "#111116" },
        textColor: cssVar("--text-secondary") || "#B2B2C2",
        fontFamily: "var(--font-mono)",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: cssVar("--border-subtle") || "#1E1E28" },
        horzLines: { color: cssVar("--border-subtle") || "#1E1E28" },
      },
      rightPriceScale: { borderColor: cssVar("--border-default") || "#2A2A38" },
      timeScale: { borderColor: cssVar("--border-default") || "#2A2A38", timeVisible: false },
      crosshair: { mode: CrosshairMode.Magnet },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: cssVar("--accent") || "#00D084",
      downColor: cssVar("--error") || "#F87171",
      borderUpColor: cssVar("--accent") || "#00D084",
      borderDownColor: cssVar("--error") || "#F87171",
      wickUpColor: cssVar("--accent") || "#00D084",
      wickDownColor: cssVar("--error") || "#F87171",
    });
    candleSeries.setData(spec.candles);

    // Volume on a separate (overlay) price scale at the bottom 25 %.
    if (spec.volume && spec.volume.length) {
      const volSeries = chart.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
        color: cssVar("--accent-dim") || "#00D08420",
      });
      volSeries.setData(spec.volume);
      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.75, bottom: 0 },
      });
    }

    chart.timeScale().fitContent();

    // Responsive width via ResizeObserver.
    const ro = new ResizeObserver(() => {
      if (node) chart.resize(node.clientWidth, 280);
    });
    ro.observe(node);

    return () => {
      ro.disconnect();
      chart.remove();
    };
  }, [spec]);

  const summary = computeSummary(spec);

  return (
    <figure className="my-3 border border-border-subtle bg-bg-surface">
      {/* Header strip — symbol + period + pct change */}
      <figcaption className="flex items-baseline justify-between border-b border-border-subtle px-3 py-2 font-mono text-[11px]">
        <span className="text-text-primary">
          <strong className="text-accent">{spec.symbol}</strong>
          <span className="ml-2 text-text-muted">
            {spec.period} · {spec.interval} · {spec.candles.length} pts
          </span>
        </span>
        {summary && (
          <span
            className={
              summary.pct >= 0
                ? "text-accent"
                : "text-err"
            }
          >
            {summary.pct >= 0 ? "+" : ""}
            {summary.pct.toFixed(2)}%
            <span className="ml-2 text-text-muted">
              ({summary.first.toFixed(2)} → {summary.last.toFixed(2)})
            </span>
          </span>
        )}
      </figcaption>
      <div ref={containerRef} className="w-full" />
    </figure>
  );
}

function computeSummary(spec: ChartSpec): { pct: number; first: number; last: number } | null {
  if (!spec.candles || spec.candles.length === 0) return null;
  const first = spec.candles[0].close;
  const last = spec.candles[spec.candles.length - 1].close;
  if (!first) return null;
  return { pct: ((last - first) / first) * 100, first, last };
}
