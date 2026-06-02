import { useState } from "react";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import { Toaster } from "react-hot-toast";
import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";
import { useConfigStore } from "@/store/configStore";
import { useThreadStore } from "@/store/threadStore";
import { useChatStore } from "@/store/chatStore";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";

import { Header } from "@/components/Header";
import { Sidebar } from "@/components/Sidebar";
import { ChatPanel } from "@/components/ChatPanel";
import { CitationsPanel } from "@/components/CitationsPanel";
import { CommandPalette } from "@/components/CommandPalette";
import { SettingsModal } from "@/components/SettingsModal";
import { ShortcutsOverlay } from "@/components/ShortcutsOverlay";
import { PanelResizer } from "@/components/PanelResizer";
import { PanelOpenButton } from "@/components/PanelOpenButton";

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

  const {
    market, sidebarOpen, citationsOpen,
    sidebarWidth, citationsWidth,
    toggleSidebar, toggleCitations,
    setSidebarWidth, setCitationsWidth,
  } = useConfigStore();
  const clearChat = useChatStore((s) => s.clear);
  const createChat = useThreadStore((s) => s.createChat);

  const [paletteOpen, setPaletteOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const closeAll = () => {
    setPaletteOpen(false);
    setShortcutsOpen(false);
    setSettingsOpen(false);
  };

  useKeyboardShortcuts({
    "mod+k":      () => setPaletteOpen((v) => !v),
    "mod+n":      () => createChat("New chat", market),
    "mod+,":      () => setSettingsOpen((v) => !v),
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
        <AnimatePresence initial={false}>
          {sidebarOpen ? (
            <motion.div
              key="sidebar"
              className="hidden md:block"
              style={{ width: sidebarWidth }}
              initial={reduce ? false : { x: -sidebarWidth, opacity: 0 }}
              animate={{ x: 0, opacity: 1,
                transition: { delay: stagger * 0, type: "spring", stiffness: 300, damping: 30 } }}
              exit={{ x: -sidebarWidth, opacity: 0, transition: { duration: 0.18 } }}
            >
              <Sidebar />
            </motion.div>
          ) : null}
        </AnimatePresence>

        {sidebarOpen ? (
          <PanelResizer side="left" width={sidebarWidth} onResize={setSidebarWidth} />
        ) : (
          <PanelOpenButton side="left" onClick={toggleSidebar} />
        )}

        <motion.div
          className="min-w-0 flex-1"
          initial={reduce ? false : { y: 12, opacity: 0 }}
          animate={{ y: 0, opacity: 1, transition: { delay: stagger * 1, duration: 0.35 } }}
        >
          <ChatPanel backendOnline={backendOnline} />
        </motion.div>

        {citationsOpen ? (
          <PanelResizer side="right" width={citationsWidth} onResize={setCitationsWidth} />
        ) : (
          <PanelOpenButton side="right" onClick={toggleCitations} />
        )}

        <AnimatePresence initial={false}>
          {citationsOpen ? (
            <motion.div
              key="citations"
              className="hidden lg:block"
              style={{ width: citationsWidth }}
              initial={reduce ? false : { x: citationsWidth, opacity: 0 }}
              animate={{ x: 0, opacity: 1,
                transition: { delay: stagger * 2, type: "spring", stiffness: 300, damping: 30 } }}
              exit={{ x: citationsWidth, opacity: 0, transition: { duration: 0.18 } }}
            >
              <CitationsPanel />
            </motion.div>
          ) : null}
        </AnimatePresence>
      </main>

      {/* Mobile bottom sheets */}
      <MobileSheets sidebarOpen={sidebarOpen} citationsOpen={citationsOpen} />

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        onShortcuts={() => setShortcutsOpen(true)}
        onSettings={() => setSettingsOpen(true)}
      />
      <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
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
}: {
  sidebarOpen: boolean;
  citationsOpen: boolean;
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
            <Sidebar />
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
