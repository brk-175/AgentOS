const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const GITHUB_LOGIN_URL = `${API_URL}/api/v1/auth/github/login`;

export interface Me {
  id: string;
  github_id: number;
  username: string;
  name: string | null;
  email: string | null;
  avatar_url: string | null;
}

export interface Repo {
  full_name: string;
  private: boolean;
  default_branch: string;
  description: string | null;
  updated_at: string;
  html_url: string;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function getJSON<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    credentials: "include",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") {
        detail = body.detail;
      }
    } catch {
      // non-JSON error body
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export function getMe(): Promise<Me> {
  return getJSON<Me>("/api/v1/auth/me");
}

export function getRepos(): Promise<Repo[]> {
  return getJSON<Repo[]>("/api/v1/repos");
}

export async function logout(): Promise<void> {
  await fetch(`${API_URL}/api/v1/auth/logout`, {
    method: "POST",
    credentials: "include",
  });
}
