"use client";

import { RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { AppHeader } from "@/components/app-header";
import { ConnectGithub } from "@/components/connect-github";
import { RepoList, RepoListSkeleton } from "@/components/repo-list";
import { Button } from "@/components/ui/button";
import { ApiError, type Repo, getMe, getRepos } from "@/lib/api";

type ViewState = "loading" | "guest" | "ready";

export default function DashboardPage() {
  const [view, setView] = useState<ViewState>("loading");
  const [repos, setRepos] = useState<Repo[]>([]);
  const [reposLoading, setReposLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadRepos = useCallback(async () => {
    setReposLoading(true);
    setError(null);
    try {
      setRepos(await getRepos());
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load repositories");
    } finally {
      setReposLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function bootstrap() {
      try {
        await getMe();
        if (cancelled) return;
        setView("ready");
        await loadRepos();
      } catch {
        if (cancelled) return;
        setView("guest");
      }
    }
    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, [loadRepos]);

  return (
    <div className="relative min-h-screen bg-background text-foreground">
      <AppHeader />
      <main className="mx-auto max-w-7xl px-8 py-12">
        {view === "loading" && <RepoListSkeleton />}
        {view === "guest" && <ConnectGithub />}
        {view === "ready" && (
          <>
            <div className="flex flex-wrap items-end justify-between gap-4">
              <div>
                <h1 className="text-2xl font-semibold tracking-tight">Repositories</h1>
                <p className="mt-1 text-sm text-muted-foreground">
                  Pick a repository to investigate an issue or pull request.
                </p>
              </div>
              <Button variant="outline" size="sm" onClick={() => void loadRepos()} disabled={reposLoading} className="cursor-pointer">
                <RefreshCw className={`size-3.5 ${reposLoading ? "animate-spin" : ""}`} />
                Refresh
              </Button>
            </div>
            {error && (
              <p className="mt-4 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                {error}
              </p>
            )}
            {reposLoading && !error ? <RepoListSkeleton /> : <RepoList repos={repos} />}
          </>
        )}
      </main>
    </div>
  );
}