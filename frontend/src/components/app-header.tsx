"use client";

import { Boxes, FolderGit2, History, LogOut } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { ApiError, getMe, logout, type Me } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/dashboard", label: "Repositories", icon: FolderGit2 },
  { href: "/runs", label: "Runs", icon: History },
];

/** Shared authenticated header: logo, primary navigation, account controls. */
export function AppHeader() {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<Me | null>(null);

  const loadUser = useCallback(async () => {
    try {
      setUser(await getMe());
    } catch (err) {
      if (!(err instanceof ApiError && err.status === 401)) {
        setUser(null);
      }
    }
  }, []);

  useEffect(() => {
    void loadUser();
  }, [loadUser]);

  const isActive = (href: string) =>
    pathname === href || (href === "/runs" && pathname.startsWith("/runs"));

  const handleLogout = async () => {
    await logout();
    router.push("/");
    router.refresh();
  };

  return (
    <header className="sticky top-0 z-50 border-b border-border/60 bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-8xl items-center justify-between px-8">
        <div className="flex items-center gap-8">
          <Link href="/" className="flex items-center gap-2">
            <span className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Boxes className="size-4.5" />
            </span>
            <span className="text-xl font-semibold tracking-tight">AgentOS</span>
          </Link>
          <nav className="hidden items-center gap-1 md:flex">
            {NAV_ITEMS.map(({ href, label, icon: Icon }) => (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  isActive(href)
                    ? "bg-secondary text-foreground"
                    : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground",
                )}
              >
                <Icon className="size-3.5" />
                {label}
              </Link>
            ))}
          </nav>
        </div>

        {user && (
          <div className="flex items-center gap-3">
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
  );
}
