"use client";

import { GitPullRequest, History, Loader2, Plus } from "lucide-react";
import Link from "next/link";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { RunRecord } from "@/lib/api";

const STATUS_STYLE: Record<string, string> = {
  completed: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  failed: "bg-destructive/15 text-destructive border-destructive/40",
  running: "bg-sky-500/15 text-sky-400 border-sky-500/30",
  queued: "bg-amber-500/15 text-amber-400 border-amber-500/30",
};

const VERDICT_STYLE: Record<string, string> = {
  approved: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  changes_requested: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  failed: "bg-destructive/15 text-destructive border-destructive/40",
};

function formatCompleted(completedAt: string | null): string {
  if (!completedAt) return "";
  const date = new Date(completedAt);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function RunList({ runs }: { runs: RunRecord[] }) {
  if (runs.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">No runs yet</CardTitle>
          <CardDescription>
            Start your first fix run — AgentOS investigates, fixes, opens a PR, and scores it
            with the judge.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Link href="/runs/new">
            <Button className="cursor-pointer">
              <Plus className="size-3.5" />
              New fix run
            </Button>
          </Link>
        </CardContent>
      </Card>
    );
  }

  return (
    <ul className="space-y-3">
      {runs.map((run) => (
        <li key={run.run_id}>
          <Link href={`/runs/${run.run_id}`} className="block h-full">
            <Card className="transition-colors hover:border-primary/40">
              <CardContent className="flex items-center justify-between gap-4 py-3">
                <div className="flex min-w-0 items-center gap-3">
                  <span className="flex size-9 shrink-0 items-center justify-center rounded-lg border bg-secondary/60">
                    {run.pr_url ? (
                      <GitPullRequest className="size-4.5 text-muted-foreground" />
                    ) : (
                      <History className="size-4.5 text-muted-foreground" />
                    )}
                  </span>
                  <div className="min-w-0">
                    <p className="truncate font-mono text-sm">
                      {run.repo_full_name} #{run.number ?? "?"}
                    </p>
                    <p className="truncate text-xs text-muted-foreground">
                      {run.title || (run.kind === "issue" ? "Issue fix" : "PR fix")}
                      {run.completed_at && ` · ${formatCompleted(run.completed_at)}`}
                    </p>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  {run.evaluation && (
                    <Badge className={VERDICT_STYLE[run.evaluation.verdict] ?? ""}>
                      {run.evaluation.verdict === "approved"
                        ? "Approved"
                        : run.evaluation.verdict === "changes_requested"
                          ? "Changes requested"
                          : "Failed"}
                    </Badge>
                  )}
                  <Badge className={STATUS_STYLE[run.status] ?? ""}>{run.status}</Badge>
                </div>
              </CardContent>
            </Card>
          </Link>
        </li>
      ))}
    </ul>
  );
}

export function RunListSkeleton() {
  return (
    <div className="space-y-3">
      {Array.from({ length: 4 }, (_, index) => (
        <div key={index} className="flex items-center gap-3 rounded-xl border p-4">
          <Skeleton className="size-9 rounded-lg" />
          <div className="flex-1">
            <Skeleton className="h-4 w-1/3" />
            <Skeleton className="mt-2 h-3 w-1/2" />
          </div>
          <Skeleton className="h-5 w-20" />
        </div>
      ))}
    </div>
  );
}

export function RunListError({ detail }: { detail: string }) {
  return (
    <p className="flex items-center gap-2 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
      <Loader2 className="size-3.5" />
      {detail}
    </p>
  );
}
