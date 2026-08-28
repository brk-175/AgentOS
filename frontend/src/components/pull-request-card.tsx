"use client";

import {
  CalendarClock,
  ExternalLink,
  FileDiff,
  GitPullRequest,
  User,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { RunPullRequest } from "@/lib/api";
import { cn } from "@/lib/utils";

function formatDate(value: string | null | undefined): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function PullRequestCard({
  pr,
  fallbackUrl,
  fallbackTitle,
}: {
  pr?: RunPullRequest | null;
  fallbackUrl?: string | null;
  fallbackTitle?: string | null;
}) {
  const url = pr?.url || fallbackUrl;
  const title = pr?.title || fallbackTitle || `Pull request #${pr?.number ?? "?"}`;
  const isOpen = (pr?.state ?? "open").toLowerCase() === "open";

  return (
    <Card className="overflow-hidden">
      <div className="h-0.5 bg-linear-to-r from-primary via-primary/50 to-transparent" aria-hidden />
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <span className="inline-flex size-6 items-center justify-center rounded-md bg-primary/10 text-primary">
              <GitPullRequest className="size-3.5" aria-hidden />
            </span>
            Pull Request
          </CardTitle>
          {url && (
            <Badge
              variant="outline"
              className={cn(
                isOpen
                  ? "border-emerald-500/30 bg-emerald-500/15 text-emerald-400"
                  : "border-muted-foreground/30 bg-secondary text-muted-foreground",
              )}
            >
              {isOpen ? "Open" : (pr?.state ?? "") || "Closed"}
            </Badge>
          )}
        </div>
        <CardDescription>Opened for this fix by the agent</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <p className="text-sm font-medium leading-5">
            {title}
            {pr?.number != null && (
              <span className="ml-2 font-mono text-xs text-muted-foreground">
                #{pr.number}
              </span>
            )}
          </p>
          {pr?.body && (
            <p className="mt-1.5 line-clamp-3 text-sm leading-5 text-muted-foreground">
              {pr.body}
            </p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
          {pr?.author && (
            <span className="inline-flex items-center gap-1.5">
              <User className="size-3" aria-hidden />
              @{pr.author}
            </span>
          )}
          {pr?.created_at && (
            <span className="inline-flex items-center gap-1.5">
              <CalendarClock className="size-3" aria-hidden />
              {formatDate(pr.created_at)}
            </span>
          )}
          {pr?.changed_files != null && (
            <span className="inline-flex items-center gap-1.5">
              <FileDiff className="size-3" aria-hidden />
              {pr.changed_files} file{pr.changed_files === 1 ? "" : "s"}
            </span>
          )}
        </div>

        {(pr?.head || pr?.base) && (
          <div className="flex items-center gap-1.5 font-mono text-xs">
            <span className="rounded-md border bg-secondary/60 px-1.5 py-0.5 text-muted-foreground">
              {pr.base ?? "?"}
            </span>
            <span className="text-muted-foreground/60">→</span>
            <span className="rounded-md border bg-primary/10 px-1.5 py-0.5 text-primary">
              {pr.head ?? "?"}
            </span>
          </div>
        )}

        {url && (
          <div className="flex items-center justify-between gap-2 pt-1">
            <Button
              nativeButton={false}
              size="sm"
              render={<a href={url} target="_blank" rel="noreferrer" />}
              className="cursor-pointer"
            >
              <GitPullRequest className="size-3.5" />
              View PR
              <ExternalLink className="size-3" />
            </Button>
            {pr?.additions != null && pr?.deletions != null && (
              <span className="font-mono text-xs tabular-nums">
                <span className="text-emerald-500">+{pr.additions}</span>
                <span className="mx-1 text-muted-foreground/60">/</span>
                <span className="text-destructive">-{pr.deletions}</span>
              </span>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}