"use client";

import { CheckCircle2, ExternalLink, ListChecks, XCircle } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { RunEvaluation, JudgeVerdictValue } from "@/lib/api";
import { cn } from "@/lib/utils";

const VERDICT_STYLES: Record<JudgeVerdictValue, { label: string; className: string }> = {
  approved: { label: "Approved", className: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30" },
  changes_requested: { label: "Changes requested", className: "bg-amber-500/15 text-amber-400 border-amber-500/30" },
  failed: { label: "Failed", className: "bg-destructive/15 text-destructive border-destructive/40" },
};

const SCORE_LABELS: { key: keyof RunEvaluation["scores"]; label: string }[] = [
  { key: "correctness", label: "Correctness" },
  { key: "minimality", label: "Minimality" },
  { key: "behavior_preservation", label: "Behavior preservation" },
  { key: "grounding", label: "Grounding" },
];

function VerdictIcon({ verdict }: { verdict: JudgeVerdictValue }) {
  if (verdict === "approved") {
    return <CheckCircle2 className="size-4" aria-hidden />;
  }
  return <XCircle className="size-4" aria-hidden />;
}

function ScoreRow({ label, value }: { label: string; value: number }) {
  const pct = Math.max(0, Math.min(100, value * 20));
  return (
    <div className="flex items-center gap-3">
      <span className="w-36 shrink-0 text-xs text-muted-foreground">{label}</span>
      <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-secondary">
        <div
          className={cn(
            "absolute inset-y-0 left-0 rounded-full",
            value >= 4 ? "bg-emerald-500" : value >= 3 ? "bg-amber-500" : "bg-destructive",
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-8 shrink-0 text-right font-mono text-xs tabular-nums">
        {value.toFixed(1)}
      </span>
    </div>
  );
}

export function EvaluationCardSkeleton() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Evaluation</CardTitle>
        <CardDescription>Judge review pending…</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {SCORE_LABELS.map(({ label }) => (
          <div key={label} className="flex items-center gap-3">
            <span className="w-36 shrink-0 text-xs text-muted-foreground">{label}</span>
            <div className="h-1.5 flex-1 animate-pulse rounded-full bg-secondary" />
            <span className="w-8" />
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export function EvaluationCard({ evaluation }: { evaluation: RunEvaluation }) {
  const style = VERDICT_STYLES[evaluation.verdict];
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <ListChecks className="size-4" />
            Evaluation
          </CardTitle>
          <span
            className={cn(
              "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium",
              style.className,
            )}
          >
            <VerdictIcon verdict={evaluation.verdict} />
            {style.label}
          </span>
        </div>
        <CardDescription>Review by the AgentOS judge model</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          {SCORE_LABELS.map(({ key, label }) => (
            <ScoreRow key={key} label={label} value={evaluation.scores[key]} />
          ))}
        </div>
        {evaluation.summary && (
          <p className="text-sm leading-6 text-muted-foreground">{evaluation.summary}</p>
        )}
        {evaluation.issues.length > 0 && (
          <div>
            <p className="mb-1 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              <ExternalLink className="size-3" />
              Issues flagged
            </p>
            <ul className="space-y-1">
              {evaluation.issues.map((issue, index) => (
                <li key={index} className="list-inside list-disc text-xs leading-5 text-destructive/90">
                  {issue}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
