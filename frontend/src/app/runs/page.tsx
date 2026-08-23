"use client";

import { History, Plus, RefreshCw } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

const MIN_REFRESH_VISIBLE_MS = 600; // keep the spinner on-screen this long
import { AppHeader } from "@/components/app-header";
import { ConnectGithub } from "@/components/connect-github";
import { RunList, RunListError, RunListSkeleton } from "@/components/run-list";
import { Button } from "@/components/ui/button";
import { ApiError, getMe, getRuns, type RunRecord } from "@/lib/api";

type ViewState = "loading" | "guest" | "ready";

export default function RunsPage() {
  const [view, setView] = useState<ViewState>("loading");
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const loadingRef = useRef(false);

  const load = useCallback(async () => {
    if (loadingRef.current) return; // ignore clicks while a refresh is in flight
    loadingRef.current = true;
    setLoading(true);
    setError(null);
    const started = Date.now();
    try {
      setRuns(await getRuns());
    } catch (err) {
      setError(
        err instanceof ApiError && err.status === 401
          ? "Sign in to view your runs."
          : "Failed to load runs.",
      );
    } finally {
      // Local API resolves in a few ms — keep the spinner perceivable for a
      // minimum duration so the refresh feedback is visible every click.
      const elapsed = Date.now() - started;
      const remaining = Math.max(0, MIN_REFRESH_VISIBLE_MS - elapsed);
      await new Promise((resolve) => setTimeout(resolve, remaining));
      loadingRef.current = false;
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function bootstrap() {
      try {
        await getMe();
        if (cancelled) return;
        setView("ready");
        await load();
      } catch {
        if (cancelled) return;
        setView("guest");
      }
    }
    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, [load]);

  // stale-while-revalidate: only the first load (no data yet) shows the
  // skeleton; refreshes keep the current list visible so the page never jumps.
  const refreshing = loading && runs.length > 0;
  const firstLoad = loading && runs.length === 0;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <AppHeader />
      <main className="mx-auto max-w-6xl px-8 py-12">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight">
            <History className="size-5" />
            Runs
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Fix runs and their judge verdicts.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => void load()} disabled={loading} className="cursor-pointer">
            <RefreshCw className={`size-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button size="sm" className="cursor-pointer">
            <Link href="/runs/new" className="inline-flex items-center gap-1.5">
              <Plus className="size-3.5" />
              New fix run
            </Link>
          </Button>
        </div>
      </div>
      {view === "guest" ? (
        <ConnectGithub />
      ) : (
        <div className={`mt-8 transition-opacity duration-200 ${refreshing ? "pointer-events-none opacity-60" : "opacity-100"}`}>
          {error ? (
            <RunListError detail={error} />
          ) : firstLoad ? (
            <RunListSkeleton />
          ) : (
            <RunList runs={runs} />
          )}
        </div>
      )}
      </main>
    </div>
  );
}
