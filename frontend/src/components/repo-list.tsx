import { FolderGit2, Lock } from "lucide-react";
import Link from "next/link";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { Repo } from "@/lib/api";

function formatUpdated(updatedAt: string): string {
  const date = new Date(updatedAt);
  if (Number.isNaN(date.getTime())) {
    return updatedAt;
  }
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

export function RepoList({ repos }: { repos: Repo[] }) {
  if (repos.length === 0) {
    return (
      <Card className="mx-auto mt-8 max-w-2xl">
        <CardHeader>
          <CardTitle className="text-base">No repositories found</CardTitle>
          <CardDescription>
            Your GitHub account has no repositories AgentOS can see. Create one, then refresh.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <ul className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {repos.map((repo) => (
        <li key={repo.full_name}>
          <Link href={repo.html_url} target="_blank" rel="noreferrer" className="block h-full">
            <Card className="h-full transition-colors hover:border-primary/40">
              <CardHeader>
                <div className="flex items-start justify-between gap-2">
                  <span className="flex size-9 items-center justify-center rounded-lg border bg-secondary/60">
                    <FolderGit2 className="size-4.5 text-muted-foreground" />
                  </span>
                  {repo.private && (
                    <Badge variant="outline" className="gap-1 text-xs">
                      <Lock className="size-3" />
                      Private
                    </Badge>
                  )}
                </div>
                <CardTitle className="mt-3 font-mono text-sm">{repo.full_name}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="line-clamp-2 min-h-10 text-sm leading-5 text-muted-foreground">
                  {repo.description ?? "No description"}
                </p>
                <p className="mt-3 text-xs text-muted-foreground/70">
                  default: {repo.default_branch} · updated {formatUpdated(repo.updated_at)}
                </p>
              </CardContent>
            </Card>
          </Link>
        </li>
      ))}
    </ul>
  );
}

export function RepoListSkeleton() {
  return (
    <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: 6 }, (_, index) => (
        <div key={index} className="rounded-xl border p-5">
          <Skeleton className="size-9 rounded-lg" />
          <Skeleton className="mt-4 h-4 w-2/3" />
          <Skeleton className="mt-3 h-3 w-full" />
          <Skeleton className="mt-2 h-3 w-4/5" />
        </div>
      ))}
    </div>
  );
}
