import Link from "next/link";

import { GithubMark } from "@/components/github-mark";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { GITHUB_LOGIN_URL } from "@/lib/api";

export function ConnectGithub() {
  return (
    <Card className="mx-auto mt-16 max-w-md text-center">
      <CardHeader className="items-center">
        <span className="flex size-12 items-center justify-center rounded-xl border bg-secondary/60">
          <GithubMark className="size-6" />
        </span>
        <CardTitle className="text-xl">Connect your GitHub account</CardTitle>
        <CardDescription>
          AgentOS needs access to your repositories to investigate issues and open pull requests.
        </CardDescription>
      </CardHeader>
      <CardContent className="pb-6">
        <Button nativeButton={false} size="lg" className="w-full" render={<Link href={GITHUB_LOGIN_URL} />}>
          <GithubMark className="size-4" />
          Continue with GitHub
        </Button>
      </CardContent>
    </Card>
  );
}
