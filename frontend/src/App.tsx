import { useState } from "react";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import { Toaster } from "react-hot-toast";
import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";
import { useConfigStore } from "@/store/configStore";
import { useRAGQuery } from "@/hooks/useRAGQuery";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";

import { Header } from "@/components/Header";
import { Sidebar } from "@/components/Sidebar";
import { ChatPanel } from "@/components/ChatPanel";
import { CitationsPanel } from "@/components/CitationsPanel";
import { CompareModal } from "@/components/CompareModal";
import { CommandPalette } from "@/components/CommandPalette";
import { ShortcutsOverlay } from "@/components/ShortcutsOverlay";
import { useChatStore } from "@/store/chatStore";

export function App() {
  const reduce = useReducedMotion();
  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
    refetchOnWindowFocus: false,
    retry: 1,
  });
  const backendOnline = health.isSuccess;

  const { market, config, companyFilter, sidebarOpen, citationsOpen, toggleSidebar, toggleCitations } = useConfigStore();
  const clearChat = useChatStore((s) => s.clear);

  const [paletteOpen, setPaletteOpen] = useState(false);
  const [compareOpen, setCompareOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);

  const { ask } = useRAGQuery(market, config, companyFilter);

  const closeAll = () => {
    setPaletteOpen(false);
    setCompareOpen(false);
    setShortcutsOpen(false);
  };

  useKeyboardShortcuts({
    "mod+k":      () => setPaletteOpen((v) => !v),
    "mod+shift+c":() => setCompareOpen((v) => !v),
    "mod+l":      () => clearChat(),
    "mod+[":      () => toggleSidebar(),
    "mod+]":      () => toggleCitations(),
    "escape":     () => closeAll(),
    "?":          () => setShortcutsOpen((v) => !v),
  });

  const stagger = reduce ? 0 : 0.08;

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden">
      <motion.div
        initial={reduce ? false : { y: -12, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.4 }}
      >
        <Header onCommandPalette={() => setPaletteOpen(true)} />
      </motion.div>

      <main className="flex min-h-0 flex-1">
        {/* Sidebar */}
        <AnimatePresence initial={false}>
          {sidebarOpen && (
            <motion.div
              className="hidden md:block"
              initial={reduce ? false : { x: -240, opacity: 0 }}
              animate={{ x: 0, opacity: 1, transition: { delay: stagger * 0, type: "spring", stiffness: 300, damping: 30 } }}
              exit={{ x: -240, opacity: 0, transition: { duration: 0.18 } }}
            >
              <Sidebar onAsk={ask} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Chat */}
        <motion.div
          className="min-w-0 flex-1"
          initial={reduce ? false : { y: 12, opacity: 0 }}
          animate={{ y: 0, opacity: 1, transition: { delay: stagger * 1, duration: 0.35 } }}
        >
          <ChatPanel backendOnline={backendOnline} />
        </motion.div>

        {/* Citations */}
        <AnimatePresence initial={false}>
          {citationsOpen && (
            <motion.div
              className="hidden lg:block"
              initial={reduce ? false : { x: 340, opacity: 0 }}
              animate={{ x: 0, opacity: 1, transition: { delay: stagger * 2, type: "spring", stiffness: 300, damping: 30 } }}
              exit={{ x: 340, opacity: 0, transition: { duration: 0.18 } }}
            >
              <CitationsPanel />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Mobile bottom sheets */}
      <MobileSheets sidebarOpen={sidebarOpen} citationsOpen={citationsOpen} onAsk={ask} />

      {/* Overlays */}
      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        onCompare={() => setCompareOpen(true)}
        onShortcuts={() => setShortcutsOpen(true)}
      />
      <CompareModal open={compareOpen} onClose={() => setCompareOpen(false)} />
      <ShortcutsOverlay open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />

      <Toaster
        position="bottom-right"
        toastOptions={{
          style: {
            background: "var(--bg-elevated)",
            color: "var(--text-primary)",
            border: "1px solid var(--border-default)",
            fontFamily: "var(--font-mono)",
            fontSize: "12px",
            borderRadius: "4px",
          },
        }}
      />
    </div>
  );
}

function MobileSheets({
  sidebarOpen,
  citationsOpen,
  onAsk,
}: {
  sidebarOpen: boolean;
  citationsOpen: boolean;
  onAsk: (q: string) => void;
}) {
  const reduce = useReducedMotion();
  const { toggleSidebar, toggleCitations } = useConfigStore();

  return (
    <>
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            key="m-sidebar"
            className="fixed inset-x-0 bottom-0 z-40 max-h-[70vh] overflow-y-auto border-t border-border-default bg-bg-base md:hidden"
            initial={reduce ? false : { y: "100%" }}
            animate={{ y: 0 }}
            exit={{ y: "100%" }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            drag="y"
            dragConstraints={{ top: 0, bottom: 0 }}
            dragElastic={0.2}
            onDragEnd={(_, info) => {
              if (info.offset.y > 120) toggleSidebar();
            }}
          >
            <div className="mx-auto my-2 h-1 w-10 rounded bg-border-strong" />
            <Sidebar onAsk={onAsk} />
          </motion.aside>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {citationsOpen && (
          <motion.aside
            key="m-citations"
            className="fixed inset-x-0 bottom-0 z-40 max-h-[70vh] overflow-y-auto border-t border-border-default bg-bg-base lg:hidden"
            initial={reduce ? false : { y: "100%" }}
            animate={{ y: 0 }}
            exit={{ y: "100%" }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            drag="y"
            dragConstraints={{ top: 0, bottom: 0 }}
            dragElastic={0.2}
            onDragEnd={(_, info) => {
              if (info.offset.y > 120) toggleCitations();
            }}
          >
            <div className="mx-auto my-2 h-1 w-10 rounded bg-border-strong" />
            <CitationsPanel />
          </motion.aside>
        )}
      </AnimatePresence>
    </>
  );
}
