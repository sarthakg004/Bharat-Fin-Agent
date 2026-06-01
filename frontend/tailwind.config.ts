import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          base: "var(--bg-base)",
          surface: "var(--bg-surface)",
          elevated: "var(--bg-elevated)",
          hover: "var(--bg-hover)",
        },
        // Both keys exist so `border-border` (Tailwind default) and the
        // explicit `border-border-default` both resolve to the same colour.
        border: {
          subtle: "var(--border-subtle)",
          DEFAULT: "var(--border-default)",
          default: "var(--border-default)",
          strong: "var(--border-strong)",
        },
        text: {
          primary: "var(--text-primary)",
          secondary: "var(--text-secondary)",
          muted: "var(--text-muted)",
        },
        accent: {
          DEFAULT: "var(--accent)",
          dim: "var(--accent-dim)",
          hover: "var(--accent-hover)",
        },
        warning: { DEFAULT: "var(--warning)", dim: "var(--warning-dim)" },
        err:     { DEFAULT: "var(--error)",   dim: "var(--error-dim)"   },
        info:    { DEFAULT: "var(--info)",    dim: "var(--info-dim)"    },
        chart: {
          1: "var(--chart-1)", 2: "var(--chart-2)",
          3: "var(--chart-3)", 4: "var(--chart-4)",
        },
      },
      fontFamily: {
        display: ["'DM Serif Display'", "Georgia", "serif"],
        ui:      ["Geist", "'Helvetica Neue'", "sans-serif"],
        mono:    ["'IBM Plex Mono'", "'Courier New'", "monospace"],
      },
      borderRadius: {
        none: "0px",
        DEFAULT: "4px",
        sm: "2px",
        md: "4px",
      },
      keyframes: {
        blink:   { "0%,49%": { opacity: "1" }, "50%,100%": { opacity: "0" } },
        shimmer: { "0%": { backgroundPosition: "-400px 0" }, "100%": { backgroundPosition: "400px 0" } },
        pulseDot: { "0%,100%": { transform: "scale(1)", opacity: "1" }, "50%": { transform: "scale(1.25)", opacity: "0.6" } },
      },
      animation: {
        blink:    "blink 1s steps(2) infinite",
        shimmer:  "shimmer 1.6s linear infinite",
        pulseDot: "pulseDot 1.4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
} satisfies Config;
