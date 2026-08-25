"use client";

import { ArrowLeft, ExternalLink, GitPullRequest, Loader2, Terminal } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { AppHeader } from "@/components/app-header";
import {
  EvaluationCard,
  EvaluationCardSkeleton,
} from "@/components/evaluation-card";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { ApiError, getRun, subscribeRunEvents, type RunDetail, type RunEvaluation, type RunEvent } from "@/lib/api";

export default function RunDetailPage() {
  const params = useParams<{ runId: string }>();
  const runId = params.runId;
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Single source of truth for the timeline: seeded from the last fetch's
  // backlog, then appended by SSE live events (deduped by event time).
  const [liveEvents, setLiveEvents] = useState<RunEvent[]>([]);
  const [liveStatus, setLiveStatus] = useState<string | null>(null);
  const [liveEvaluation, setLiveEvaluation] = useState<RunEvaluation | null>(null);
  const seenEventTimes = useRef<Set<string>>(new Set());

  const load = useCallback(async () => {
    setError(null);
    try {
      const data = await getRun(runId);
      setRun(data);
      const backlog = data.events ?? [];
      seenEventTimes.current = new Set(
        backlog.map((event) => event.time ?? `${event.type}-${event.stage}-${event.kind}`),
      );
      setLiveEvents(backlog);
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
        const key = event.time ?? `${event.type}-${event.stage}-${event.kind}`;
        if (seenEventTimes.current.has(key)) return;
        seenEventTimes.current.add(key);
        setLiveEvents((previous) => [...previous, event]);
        // Realtime state from the stream: status transitions + the judge verdict.
        if (event.type === "start") setLiveStatus("running");
        if (event.type === "final") setLiveStatus("completed");
        if (event.type === "error") setLiveStatus("failed");
        if (event.type === "event" && event.stage === "eval" && event.kind === "verdict") {
          if (event.evaluation) setLiveEvaluation(event.evaluation);
        }
        if (event.type === "final" || event.type === "error") {
          // The terminal SSE payload carries the full compacted state — apply
          // it immediately so Proposed Changes / PR URL appear without a reload
          // (a follow-up load() would race the worker's Redis final marker).
          if (event.type === "final" && event.state) {
            setRun({
              run_id: runId,
              status: "completed",
              state: event.state,
              detail: null,
              events: [],
            });
          }
          void load();
        }
      },
      () => void load(),
    );
    return close;
  }, [run, runId, load]);

  // Realtime status: SSE wins over the (possibly stale) fetch snapshot.
  const status = liveStatus ?? run?.status ?? "unknown";
  const evaluation = run?.state?.evaluation ?? liveEvaluation;
  const changes = run?.state?.proposed_changes ?? [];
  // liveEvents now carries the backlog AND every live event — the timeline is
  // always current, no shadowing from the stale fetch snapshot.
  const events = liveEvents;
  const isLive = status === "running" || status === "queued";
  const inFlightIndex = isLive ? events.length - 1 : -1;

  let content;
  if (error) {
    content = (
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
  } else if (!run) {
    content = (
      <div className="mx-auto flex max-w-4xl items-center gap-2 px-8 py-24 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Loading run…
      </div>
    );
  } else {
    content = (
      <main className="mx-auto max-w-7xl space-y-6 px-8 py-12">
      <div className="flex items-center justify-between gap-4">
        <Link href="/runs" className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="size-3.5" />
          Runs
        </Link>
        <ShimmerButton shimmer={status !== "completed"}>{status}</ShimmerButton>
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
                {events.map((event, index) => {
                  const inProgress = index === inFlightIndex;
                  return (
                    <li key={event.time ?? index} className="flex items-center gap-2 text-xs">
                      <span className="w-20 shrink-0 font-mono text-muted-foreground/70">
                        {event.stage ?? event.type}
                      </span>
                      {inProgress ? (
                        <span className="event-shimmer min-w-0 flex-1 truncate">{event.detail ?? event.type}</span>
                      ) : (
                        <span className="min-w-0 flex-1 truncate text-muted-foreground/80">
                          {event.detail ?? event.type}
                        </span>
                      )}
                      {inProgress && (
                        <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-primary" aria-hidden />
                      )}
                    </li>
                  );
                })}
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
      </main>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <AppHeader />
      {content}
    </div>
  );
}
