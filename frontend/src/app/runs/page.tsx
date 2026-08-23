"use client";

import { History, Plus } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { RunList, RunListError, RunListSkeleton } from "@/components/run-list";
import { Button } from "@/components/ui/button";
import { ApiError, getRuns, type RunRecord } from "@/lib/api";

export default function RunsPage() {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setRuns(await getRuns());
    } catch (err) {
      setError(
        err instanceof ApiError && err.status === 401
          ? "Sign in to view your runs."
          : "Failed to load runs.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="mx-auto max-w-5xl px-8 py-12">
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
          <Button variant="outline" size="sm" onClick={() => void load()} disabled={loading}>
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
      <div className="mt-8">
        {error ? (
          <>
            <RunListError detail={error} />
            {error.startsWith("Sign in") && (
              <Link href="/dashboard" className="mt-4 inline-block text-sm text-primary">
                Go to dashboard
              </Link>
            )}
          </>
        ) : loading ? (
          <RunListSkeleton />
        ) : (
          <RunList runs={runs} />
        )}
      </div>
    </div>
  );
}
