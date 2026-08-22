"use client";

import { Boxes, LogOut, RefreshCw } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { ConnectGithub } from "@/components/connect-github";
import { RepoList, RepoListSkeleton } from "@/components/repo-list";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { ApiError, logout, type Me, type Repo, getMe, getRepos } from "@/lib/api";

type ViewState = "loading" | "guest" | "ready";

export default function DashboardPage() {
  const [view, setView] = useState<ViewState>("loading");
  const [user, setUser] = useState<Me | null>(null);
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
        const me = await getMe();
        if (cancelled) return;
        setUser(me);
        setView("ready");
        await loadRepos();
      } catch (err) {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 401) {
          setView("guest");
        } else {
          setView("guest");
        }
      }
    }
    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, [loadRepos]);

  const handleLogout = async () => {
    await logout();
    window.location.href = "/";
  };

  return (
    <div className="relative min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-50 border-b border-border/60 bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-8xl items-center justify-between px-8">
          <Link href="/" className="flex items-center gap-2">
            <span className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Boxes className="size-4.5" />
            </span>
            <span className="text-xl font-semibold tracking-tight">AgentOS</span>
          </Link>
          {view === "ready" && user && (
            <div className="flex items-center gap-3">
              <Button variant="ghost" size="sm" className="cursor-pointer">
                <Link href="/runs">Runs</Link>
              </Button>
              <span className="flex items-center gap-2 text-sm text-muted-foreground">
                <Avatar className="size-7">
                  {user.avatar_url ? <AvatarImage src={user.avatar_url} alt={user.username} /> : null}
                  <AvatarFallback className="text-xs">{user.username.slice(0, 2).toUpperCase()}</AvatarFallback>
                </Avatar>
                {user.name ?? user.username}
              </span>
              <Separator orientation="vertical" className="h-5" />
              <Button variant="ghost" size="sm" className="cursor-pointer" onClick={handleLogout}>
                <LogOut className="size-3.5" />
                Log out
              </Button>
            </div>
          )}
        </div>
      </header>

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
              <Button variant="outline" size="sm" onClick={() => void loadRepos()} disabled={reposLoading}>
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