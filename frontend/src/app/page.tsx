import Link from "next/link";
import {
  Activity,
  Boxes,
  Database,
  FileSearch,
  Gauge,
  GitPullRequest,
  Plug,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

import { GithubMark } from "@/components/github-mark";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const GITHUB_LOGIN_URL = `${API_URL}/api/v1/auth/github/login`;

const steps = [
  {
    step: "01",
    title: "Connect a repository",
    description:
      "Sign in with GitHub and pick the repository your team wants AgentOS to work on.",
    icon: GithubMark,
  },
  {
    step: "02",
    title: "Pick an Issue or PR",
    description:
      "Select the problem. AgentOS investigates it and retrieves the context that matters.",
    icon: FileSearch,
  },
  {
    step: "03",
    title: "Review the Pull Request",
    description:
      "AgentOS creates a branch, commits the fix, and opens a PR — with a full trace of how it got there.",
    icon: GitPullRequest,
  },
];

const features = [
  {
    title: "Investigation summary",
    description:
      "A root-cause hypothesis grounded in your codebase, not a guess. Every run explains its reasoning.",
    icon: FileSearch,
  },
  {
    title: "Repository-aware context",
    description:
      "A lightweight retrieval layer surfaces READMEs, docs, similar issues, and relevant source files.",
    icon: Database,
  },
  {
    title: "Tool layer via MCP",
    description:
      "Every GitHub action — branches, commits, pull requests — flows through a dedicated MCP server.",
    icon: Plug,
  },
  {
    title: "Evaluation score",
    description:
      "Each fix is scored for correctness, groundedness, and quality by an LLM judge with a golden dataset.",
    icon: Gauge,
  },
  {
    title: "Live execution trace",
    description:
      "Stream the agent's steps as they happen: what it read, why it changed files, and what it created.",
    icon: Activity,
  },
  {
    title: "Audit-ready by design",
    description:
      "Important actions are logged, retried, and rate-limited. Built to be reviewed, not trusted.",
    icon: ShieldCheck,
  },
];

const stack = [
  "Next.js",
  "FastAPI",
  "LangGraph",
  "Celery",
  "GitHub MCP",
  "OpenCode",
  "pgvector",
  "Redis",
  "PostgreSQL",
  "Caddy",
  "Docker",
];

export default function LandingPage() {
  return (
    <div className="relative min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-50 border-b border-border/60 bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-8xl items-center justify-between px-8">
          <Link href="/" className="flex items-center gap-2">
            <span className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Boxes className="size-4.5" />
            </span>
            <span className="text-xl font-semibold tracking-tight ml-1">AgentOS</span>
          </Link>
          <Button nativeButton={false} size="sm" render={<Link href={GITHUB_LOGIN_URL} />}>
            <GithubMark className="size-4" />
            Continue with GitHub
          </Button>
        </div>
      </header>

      <main>
        <section className="relative overflow-hidden">
          <div className="bg-grid pointer-events-none absolute inset-0" aria-hidden />
          <div
            className="pointer-events-none absolute -top-40 left-1/2 h-96 w-[42rem] -translate-x-1/2 rounded-full bg-primary/10 blur-3xl"
            aria-hidden
          />

          <div className="relative mx-auto max-w-6xl px-6 pb-24 pt-24 text-center sm:pt-32">
            <Badge variant="outline" className="mb-6 gap-1.5 rounded-full px-3 py-3 text-xs">
              <Sparkles className="size-3" />
              From issue to pull request — automatically
            </Badge>
            <h1 className="mx-auto max-w-3xl text-balance text-4xl font-semibold tracking-tight sm:text-6xl">
              Investigate the problem.{" "}
              <span className="bg-gradient-to-r from-primary via-primary/80 to-primary/40 bg-clip-text text-transparent">
                Ship the fix.
              </span>
            </h1>
            <p className="mx-auto mt-6 max-w-2xl text-pretty text-base leading-7 text-muted-foreground sm:text-lg">
              AgentOS connects to your GitHub repository, investigates an Issue or PR, retrieves the
              relevant context, writes the fix, and opens a new pull request — so your engineers
              review the change instead of writing it from scratch.
            </p>
            <div className="mt-10 flex flex-col items-center justify-center gap-3 sm:flex-row">
              <Button nativeButton={false} size="lg" className="w-full sm:w-auto" render={<Link href={GITHUB_LOGIN_URL} />}>
                <GithubMark className="size-4" />
                Continue with GitHub
              </Button>
              <Button nativeButton={false} variant="outline" size="lg" className="w-full sm:w-auto" render={<Link href="#how-it-works" />}>See how it works</Button>
            </div>
          </div>
        </section>

        <section id="how-it-works" className="mx-auto max-w-6xl scroll-mt-20 px-6 py-20">
          <div className="mb-12 text-center">
            <Badge variant="secondary" className="mb-3 py-3">
              How it works
            </Badge>
            <h2 className="text-balance text-3xl font-semibold tracking-tight sm:text-4xl">
              Three steps from bug report to review-ready PR
            </h2>
          </div>
          <div className="grid gap-6 md:grid-cols-3">
            {steps.map(({ step, title, description, icon: Icon }) => (
              <Card key={step} className="relative overflow-hidden">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <span className="flex size-10 items-center justify-center rounded-lg border bg-secondary/60">
                      <Icon className="size-5 text-muted-foreground" />
                    </span>
                    <span className="font-mono text-sm text-muted-foreground/60">{step}</span>
                  </div>
                  <CardTitle className="mt-4 text-lg">{title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-sm leading-6">{description}</CardDescription>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        <section className="border-y bg-secondary/30">
          <div className="mx-auto max-w-6xl px-6 py-20">
            <div className="mb-12 text-center">
              <Badge variant="secondary" className="mb-3 py-3">
                Built for engineering teams
              </Badge>
              <h2 className="text-balance text-3xl font-semibold tracking-tight sm:text-4xl">
                Everything a reviewer needs, nothing they don&apos;t
              </h2>
            </div>
            <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
              {features.map(({ title, description, icon: Icon }) => (
                <Card key={title}>
                  <CardHeader>
                    <span className="mb-3 flex size-10 items-center justify-center rounded-lg border bg-background">
                      <Icon className="size-5 text-muted-foreground" />
                    </span>
                    <CardTitle className="text-base">{title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-sm leading-6">{description}</CardDescription>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-6xl px-6 py-16 text-center">
          <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground/70">
            Built on a production-grade stack
          </p>
          <div className="mt-6 flex flex-wrap items-center justify-center gap-x-8 gap-y-4">
            {stack.map((item) => (
              <span key={item} className="font-mono text-sm text-muted-foreground">
                {item}
              </span>
            ))}
          </div>
        </section>

        <section className="mx-auto max-w-6xl px-6 pb-24">
          <div className="rounded-2xl border bg-gradient-to-b from-secondary/70 to-secondary/30 px-6 py-14 text-center">
            <h2 className="text-balance text-3xl font-semibold tracking-tight sm:text-4xl">
              Let the agent do the groundwork.
              <br />
              <span className="text-muted-foreground">Keep the review for your engineers.</span>
            </h2>
            <Button nativeButton={false} size="lg" className="mt-8" render={<Link href={GITHUB_LOGIN_URL} />}>
              <GithubMark className="size-4" />
              Continue with GitHub
            </Button>
          </div>
        </section>
      </main>

      <footer className="border-t border-border/60">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-6 py-8 sm:flex-row">
          <div className="flex items-center gap-2">
            <Boxes className="size-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">
              AgentOS · AI code-fix assistant for GitHub
            </span>
          </div>
          <Separator orientation="vertical" className="hidden h-4 sm:block" />
          <p className="text-sm text-muted-foreground/70">
            MVP v1 · Investigate · Fix · Pull Request
          </p>
        </div>
      </footer>
    </div>
  );
}