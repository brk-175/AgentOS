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

export interface JudgeScores {
  correctness: number;
  minimality: number;
  behavior_preservation: number;
  grounding: number;
}

export type JudgeVerdictValue = "approved" | "changes_requested" | "failed";

export interface RunEvaluation {
  verdict: JudgeVerdictValue;
  scores: JudgeScores;
  summary: string;
  issues: string[];
}

export interface ProposedChange {
  path: string;
  content?: string;
  edits?: { before: string; after: string }[];
  delete?: boolean;
  explanation?: string;
}

export interface RunRecord {
  run_id: string;
  repo_full_name: string;
  kind: string;
  number: number | null;
  title: string;
  base_branch: string;
  status: string;
  applied_branch: string | null;
  pr_url: string | null;
  investigation: string;
  root_cause_hypothesis: string;
  proposed_changes: ProposedChange[];
  evaluation: RunEvaluation | null;
  completed_at: string | null;
}

export interface RunEvent {
  type: string;
  stage?: string;
  kind?: string;
  detail?: string;
  time?: string;
}

export interface RunDetail {
  run_id: string;
  status: string;
  state: {
    investigation?: string | null;
    root_cause_hypothesis?: string | null;
    proposed_changes?: ProposedChange[];
    applied_branch?: string | null;
    pr_url?: string | null;
    evaluation?: RunEvaluation | null;
  } | null;
  detail?: string | null;
  events: RunEvent[];
}

export function getRuns(): Promise<RunRecord[]> {
  return getJSON<RunRecord[]>("/api/v1/runs?limit=50");
}

export function getRun(runId: string): Promise<RunDetail> {
  return getJSON<RunDetail>(`/api/v1/runs/${runId}`);
}

export type RunKind = "issue" | "pr";

export interface StartRunRequest {
  repo_full_name: string;
  kind: RunKind;
  number: number;
  title?: string;
  base_branch?: string;
}

export interface StartRunResponse {
  run_id: string;
  status: string;
}

export async function startRun(payload: StartRunRequest): Promise<StartRunResponse> {
  const response = await fetch(`${API_URL}/api/v1/runs`, {
    method: "POST",
    credentials: "include",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (typeof body.detail === "object" && body.detail !== null) {
        detail = JSON.stringify(body.detail);
      }
    } catch {
      // non-JSON error body
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as StartRunResponse;
}

/** Subscribe to a run's SSE event stream; returns a close() function. */
export function subscribeRunEvents(
  runId: string,
  onEvent: (event: RunEvent) => void,
  onClose: () => void,
): () => void {
  const source = new EventSource(`${API_URL}/api/v1/runs/${runId}/events`, {
    withCredentials: true,
  });
  source.onmessage = (message) => {
    try {
      onEvent(JSON.parse(message.data) as RunEvent);
    } catch {
      // malformed SSE payload — ignore
    }
  };
  source.onerror = () => {
    source.close();
    onClose();
  };
  return () => {
    source.close();
  };
}
