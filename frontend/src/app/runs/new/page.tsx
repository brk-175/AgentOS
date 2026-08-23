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
  startRun,
  type Repo,
  type RunKind,
} from "@/lib/api";

export default function NewRunPage() {
  const router = useRouter();
  const [repos, setRepos] = useState<Repo[]>([]);
  const [reposLoading, setReposLoading] = useState(true);
  const [repoError, setRepoError] = useState<string | null>(null);

  const [repoFullName, setRepoFullName] = useState("");
  const [kind, setKind] = useState<RunKind>("issue");
  const [number, setNumber] = useState("");
  const [title, setTitle] = useState("");
  const [baseBranch, setBaseBranch] = useState("main");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadRepos = useCallback(async () => {
    setReposLoading(true);
    setRepoError(null);
    try {
      setRepos(await getRepos());
      if (!repoFullName && repos.length === 0) {
        // defer: set after load in effect
      }
    } catch (err) {
      setRepoError(err instanceof ApiError ? err.message : "Failed to load repositories");
    } finally {
      setReposLoading(false);
    }
  }, [repoFullName, repos.length]);

  useEffect(() => {
    void loadRepos();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedRepo = useMemo(
    () => repos.find((repo) => repo.full_name === repoFullName) ?? null,
    [repos, repoFullName],
  );

  const numberValue = number.trim() === "" ? null : Number(number);
  const canSubmit =
    reposLoading === false &&
    repoFullName !== "" &&
    numberValue !== null &&
    Number.isInteger(numberValue) &&
    numberValue > 0 &&
    baseBranch.trim() !== "" &&
    submitting === false;

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit || numberValue === null) return;
    setSubmitting(true);
    setError(null);
    try {
      const payload = await startRun({
        repo_full_name: repoFullName,
        kind,
        number: numberValue,
        title: title.trim() !== "" ? title.trim() : undefined,
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
                  onChange={(event) => setRepoFullName(event.target.value)}
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
                    onClick={() => setKind(value)}
                  >
                    {value}
                  </Button>
                ))}
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-xs font-medium text-muted-foreground ml-1">
                  {kind === "issue" ? "Issue number" : "PR number"}
                </label>
                <input
                  type="number"
                  min={1}
                  step={1}
                  inputMode="numeric"
                  value={number}
                  onChange={(event) => setNumber(event.target.value)}
                  placeholder="e.g. 11"
                  className="h-9 w-full rounded-lg border bg-background px-3 text-sm outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 mt-1"
                />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-medium text-muted-foreground ml-1">Base branch</label>
                <input
                  value={baseBranch}
                  onChange={(event) => setBaseBranch(event.target.value)}
                  placeholder="main"
                  className="h-9 w-full rounded-lg border bg-background px-3 font-mono text-sm outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 mt-1"
                />
              </div>
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
