"use client";

import { ArrowLeft, ExternalLink, GitPullRequest, Loader2, Terminal } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import {
  EvaluationCard,
  EvaluationCardSkeleton,
} from "@/components/evaluation-card";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ApiError, getRun, subscribeRunEvents, type RunDetail, type RunEvent } from "@/lib/api";

export default function RunDetailPage() {
  const params = useParams<{ runId: string }>();
  const runId = params.runId;
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [liveEvents, setLiveEvents] = useState<RunEvent[]>([]);

  const load = useCallback(async () => {
    setError(null);
    try {
      setRun(await getRun(runId));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load run.");
    }
  }, [runId]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!run) return;
    const isLive = run.status === "running" || run.status === "queued";
    if (!isLive) return;
    const close = subscribeRunEvents(
      runId,
      (event) => {
        setLiveEvents((previous) => [...previous, event]);
        if (event.type === "final" || event.type === "error") {
          void load();
        }
      },
      () => void load(),
    );
    return close;
  }, [run, runId, load]);

  if (error) {
    return (
      <div className="mx-auto max-w-4xl px-8 py-12">
        <Link href="/runs" className="mb-6 inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="size-3.5" />
          Runs
        </Link>
        <p className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </p>
      </div>
    );
  }

  if (!run) {
    return (
      <div className="mx-auto flex max-w-4xl items-center gap-2 px-8 py-24 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Loading run…
      </div>
    );
  }

  const status = run.status;
  const evaluation = run.state?.evaluation ?? null;
  const changes = run.state?.proposed_changes ?? [];
  const events = run.events.length > 0 ? run.events : liveEvents;

  return (
    <div className="mx-auto max-w-4xl space-y-6 px-8 py-12">
      <div className="flex items-center justify-between gap-4">
        <Link href="/runs" className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="size-3.5" />
          Runs
        </Link>
        <Badge>{status}</Badge>
      </div>

      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-mono text-lg font-semibold tracking-tight">
            {runId.slice(0, 12)}…
          </h1>
          {run.state?.pr_url ? (
            <a
              href={run.state.pr_url}
              target="_blank"
              rel="noreferrer"
              className="mt-1 inline-flex items-center gap-1.5 text-sm text-primary hover:underline"
            >
              <GitPullRequest className="size-3.5" />
              View pull request
              <ExternalLink className="size-3" />
            </a>
          ) : (
            <p className="mt-1 text-sm text-muted-foreground">
              {run.state?.applied_branch ? `Branch: ${run.state.applied_branch}` : "No pull request opened."}
            </p>
          )}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Investigation</CardTitle>
              <CardDescription>What the agent found</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {run.state?.investigation && (
                <p className="text-sm leading-6 text-muted-foreground">{run.state.investigation}</p>
              )}
              {run.state?.root_cause_hypothesis && (
                <div>
                  <p className="mb-1 text-xs font-medium text-muted-foreground">Hypothesis</p>
                  <p className="text-sm leading-6 text-muted-foreground">
                    {run.state.root_cause_hypothesis}
                  </p>
                </div>
              )}
              {!run.state?.investigation && !run.state?.root_cause_hypothesis && (
                <p className="text-sm text-muted-foreground">No investigation recorded.</p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Terminal className="size-4" />
                Events
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ol className="space-y-1.5">
                {events.map((event, index) => (
                  <li key={index} className="flex gap-2 text-xs">
                    <span className="w-20 shrink-0 font-mono text-muted-foreground/70">
                      {event.stage ?? event.type}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-muted-foreground">
                      {event.detail ?? event.type}
                    </span>
                  </li>
                ))}
                {events.length === 0 && (
                  <li className="text-xs text-muted-foreground">No events yet.</li>
                )}
              </ol>
            </CardContent>
          </Card>

          {changes.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Proposed changes</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {changes.map((change) => (
                    <li key={change.path} className="flex items-center justify-between gap-2 rounded-lg border bg-secondary/40 px-3 py-2">
                      <span className="truncate font-mono text-xs">{change.path}</span>
                      <span className="shrink-0 text-xs text-muted-foreground">
                        {change.delete ? "delete" : change.edits?.length ? `${change.edits.length} edit(s)` : "replace"}
                      </span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </div>

        <div>
          {evaluation ? (
            <EvaluationCard evaluation={evaluation} />
          ) : status === "running" || status === "queued" ? (
            <EvaluationCardSkeleton />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Evaluation</CardTitle>
                <CardDescription>No judge verdict recorded for this run.</CardDescription>
              </CardHeader>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
