"use client";

import { ArrowLeft, FolderGit2, Loader2, Play, RefreshCw } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppHeader } from "@/components/app-header";
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
import {
  ApiError,
  getRepos,
  getTargets,
  startRun,
  type Repo,
  type RepoTarget,
  type RunKind,
} from "@/lib/api";

export default function NewRunPage() {
  const router = useRouter();
  const [repos, setRepos] = useState<Repo[]>([]);
  const [reposLoading, setReposLoading] = useState(true);
  const [repoError, setRepoError] = useState<string | null>(null);

  const [repoFullName, setRepoFullName] = useState("");
  const [kind, setKind] = useState<RunKind>("issue");
  const [targetNumber, setTargetNumber] = useState("");
  const [title, setTitle] = useState("");
  const [baseBranch, setBaseBranch] = useState("main");
  const [targetsByKind, setTargetsByKind] = useState<{
    issue: RepoTarget[];
    pr: RepoTarget[];
  }>({ issue: [], pr: [] });
  const [targetsLoading, setTargetsLoading] = useState(false);
  const [targetError, setTargetError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadRepos = useCallback(async () => {
    setReposLoading(true);
    setRepoError(null);
    try {
      setRepos(await getRepos());
    } catch (err) {
      setRepoError(err instanceof ApiError ? err.message : "Failed to load repositories");
    } finally {
      setReposLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadRepos();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedRepo = useMemo(
    () => repos.find((repo) => repo.full_name === repoFullName) ?? null,
    [repos, repoFullName],
  );

  // Both kinds are fetched together when a repo is selected, so switching
  // kind later is instant — the dropdown just filters the cached lists.
  const loadTargets = useCallback(async (repo: string) => {
    if (!repo) return;
    setTargetsLoading(true);
    setTargetError(null);
    try {
      const [issues, pulls] = await Promise.all([
        getTargets(repo, "issue"),
        getTargets(repo, "pr"),
      ]);
      setTargetsByKind({ issue: issues, pr: pulls });
    } catch (err) {
      setTargetError(err instanceof ApiError ? err.message : "Failed to load issues and PRs");
      setTargetsByKind({ issue: [], pr: [] });
    } finally {
      setTargetsLoading(false);
    }
  }, []);

  const handleRepoChange = (value: string) => {
    setRepoFullName(value);
    setTargetNumber("");
    setTitle("");
    const repo = repos.find((candidate) => candidate.full_name === value);
    setBaseBranch(repo?.default_branch ?? "main");
    void loadTargets(value);
  };

  const handleKindChange = (value: RunKind) => {
    setKind(value);
    setTargetNumber("");
  };

  const targetOptions = targetsByKind[kind];
  const selectedTarget = useMemo(
    () => targetOptions.find((target) => String(target.number) === targetNumber) ?? null,
    [targetOptions, targetNumber],
  );

  const handleTargetChange = (value: string) => {
    setTargetNumber(value);
    const target = targetOptions.find((candidate) => String(candidate.number) === value);
    setTitle(target?.title ?? "");
  };

  const canSubmit =
    reposLoading === false &&
    repoFullName !== "" &&
    targetNumber !== "" &&
    baseBranch.trim() !== "" &&
    submitting === false;

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit || !selectedTarget) return;
    setSubmitting(true);
    setError(null);
    try {
      const payload = await startRun({
        repo_full_name: repoFullName,
        kind,
        number: selectedTarget.number,
        title: title.trim() !== "" ? title.trim() : selectedTarget.title,
        base_branch: baseBranch.trim(),
      });
      router.push(`/runs/${payload.run_id}`);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.status === 429
            ? "Concurrent run limit reached — wait for another run to finish."
            : err.message
          : "Failed to start the run.",
      );
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <AppHeader />
      <main className="mx-auto max-w-2xl space-y-8 px-8 py-12">
      <div className="flex items-center justify-between gap-4">
        <Link
          href="/runs"
          className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="size-3.5" />
          Runs
        </Link>
      </div>

      <div>
        <h1 className="text-2xl font-semibold tracking-tight">New fix run</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          AgentOS investigates the target, writes the fix, opens a PR, and scores it with a judge.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base ml-1">Target</CardTitle>
          <CardDescription className="ml-1">What should AgentOS fix?</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-5">
            <div className="space-y-2">
              <label className="text-xs font-medium text-muted-foreground ml-1">Repository</label>
              {reposLoading ? (
                <div className="space-y-2">
                  <Skeleton className="h-9 w-full" />
                  <Skeleton className="h-9 w-full" />
                </div>
              ) : repoError ? (
                <div className="flex items-center justify-between gap-3 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-2.5">
                  <p className="text-sm text-destructive">{repoError}</p>
                  <Button type="button" variant="outline" size="sm" onClick={() => void loadRepos()}>
                    <RefreshCw className="size-3.5" />
                    Retry
                  </Button>
                </div>
              ) : (
                <select
                  value={repoFullName}
                  onChange={(event) => handleRepoChange(event.target.value)}
                  className="h-9 w-full cursor-pointer rounded-lg border bg-background px-3 text-sm outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 mt-1"
                >
                  <option value="">Select a repository…</option>
                  {repos.map((repo) => (
                    <option key={repo.full_name} value={repo.full_name}>
                      {repo.full_name}
                    </option>
                  ))}
                </select>
              )}
              {selectedRepo && (
                <p className="flex items-center gap-1.5 text-xs text-muted-foreground ml-1">
                  <FolderGit2 className="size-3" />
                  {selectedRepo.default_branch} · updated{" "}
                  {new Date(selectedRepo.updated_at).toLocaleDateString()}
                </p>
              )}
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-xs font-medium text-muted-foreground ml-1">Kind</label>
                <div className="flex gap-2">
                  {(["issue", "pr"] as RunKind[]).map((value) => (
                    <Button
                      key={value}
                      type="button"
                      variant={kind === value ? "default" : "outline"}
                      size="sm"
                      className="cursor-pointer capitalize mt-1 ml-1"
                      onClick={() => handleKindChange(value)}
                    >
                      {value}
                    </Button>
                  ))}
                </div>
              </div>
              <div className="space-y-2">
                <label className="text-xs font-medium text-muted-foreground ml-1">Base branch</label>
                <input
                  value={baseBranch}
                  onChange={(event) => setBaseBranch(event.target.value)}
                  placeholder={selectedRepo?.default_branch ?? "main"}
                  className="h-9 w-full rounded-lg border bg-background px-3 font-mono text-sm outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 mt-1"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-xs font-medium text-muted-foreground ml-1">
                {kind === "issue" ? "Issue" : "Pull request"}
              </label>
              {targetsLoading ? (
                <Skeleton className="h-9 w-full mt-1" />
              ) : targetError ? (
                <div className="flex items-center justify-between gap-3 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-2.5">
                  <p className="text-sm text-destructive">{targetError}</p>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => void loadTargets(repoFullName)}
                  >
                    <RefreshCw className="size-3.5" />
                    Retry
                  </Button>
                </div>
              ) : (
                <select
                  value={targetNumber}
                  onChange={(event) => handleTargetChange(event.target.value)}
                  disabled={!repoFullName || targetOptions.length === 0}
                  className="h-9 w-full cursor-pointer rounded-lg border bg-background px-3 text-sm outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 mt-1 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <option value="">
                    {!repoFullName
                      ? "Select a repository first…"
                      : targetOptions.length === 0
                        ? `No ${kind === "issue" ? "issues" : "pull requests"} found`
                        : `Select an ${kind === "issue" ? "issue" : "PR"}…`}
                  </option>
                  {targetOptions.map((target) => (
                    <option key={target.number} value={String(target.number)}>
                      #{target.number} · {target.title} ({target.state}
                      {target.merged_at ? " · merged" : ""})
                    </option>
                  ))}
                </select>
              )}
              {repoFullName && !targetsLoading && !targetError && (
                <p className="text-xs text-muted-foreground ml-1">
                  {targetOptions.length}{" "}
                  {kind === "issue"
                    ? `${targetOptions.length === 1 ? "issue" : "issues"}`
                    : `${targetOptions.length === 1 ? "PR" : "PRs"}`}{" "}
                  available
                </p>
              )}
            </div>

            <div className="space-y-2">
              <label className="text-xs font-medium text-muted-foreground ml-1">
                Title <span className="text-muted-foreground/50">(optional)</span>
              </label>
              <input
                value={title}
                onChange={(event) => setTitle(event.target.value)}
                placeholder="Short description used for retrieval + commit message"
                className="h-9 w-full rounded-lg border bg-background px-3 text-sm outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 mt-1"
              />
            </div>

            {error && (
              <p className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                {error}
              </p>
            )}

            <Button type="submit" disabled={!canSubmit} className="w-full cursor-pointer">
              {submitting ? (
                <>
                  <Loader2 className="size-4 animate-spin" />
                  Starting…
                </>
              ) : (
                <>
                  <Play className="size-4" />
                  Start fix run
                </>
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      {repos.length > 0 && !repoError && (
        <Badge variant="outline" className="gap-1 text-xs py-3">
          {repos.length} repositories available
        </Badge>
      )}
      </main>
    </div>
  );
}
