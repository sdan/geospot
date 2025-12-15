"use client";

import { Sidebar } from "@/components/ui/sidebar";
import { Badge } from "@/components/ui/badge";
import { useState } from "react";

export default function TrainingRunDetailsPage({
  params,
}: {
  params: { runId: string };
}) {
  const [activeNav, setActiveNav] = useState("training-runs");

  // Decode the ID if it was URL encoded (though usually params.id handles this, good to be safe for display)
  const runId = decodeURIComponent(params.runId);

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar
        activeItem={activeNav}
        onNavigate={(item) => setActiveNav(item)}
      />
      <main className="flex-1 overflow-auto">
        <div className="p-6">
          <div className="space-y-2">
            <header className="pt-2 pb-6">
              <div className="flex items-center gap-4">
                <div>
                  <div className="flex items-center gap-3">
                    <h1 className="text-2xl font-semibold tracking-tight">
                      {runId}
                    </h1>
                    <Badge variant="green" className="rounded-full px-2">
                      Active
                    </Badge>
                  </div>
                  <p className="mt-0.5 text-sm text-muted-foreground">
                    Base Model: Qwen/Qwen3-VL-30B-A3B-Instruct
                  </p>
                </div>
              </div>
            </header>

            <div className="mt-6 space-y-6">
              <div className="rounded-lg border border-border p-4 bg-card text-card-foreground">
                <h2 className="text-lg font-semibold tracking-tight mb-4">
                  Training Run Details
                </h2>
                <div className="grid grid-cols-2 gap-x-4 gap-y-6">
                  <div className="flex flex-col gap-1">
                    <h3 className="text-sm font-medium tracking-tight">
                      Base Model
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      Qwen/Qwen3-VL-30B-A3B-Instruct
                    </p>
                  </div>
                  <div className="flex flex-col gap-1">
                    <h3 className="text-sm font-medium tracking-tight">
                      Owner
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      tml:organization_user:c42680f9-889e-482a-89b5-2d35a8d28494
                    </p>
                  </div>
                  <div className="flex flex-col gap-1">
                    <h3 className="text-sm font-medium tracking-tight">
                      LoRA Rank
                    </h3>
                    <p className="text-sm text-muted-foreground">32</p>
                  </div>
                  <div className="flex flex-col gap-1">
                    <h3 className="text-sm font-medium tracking-tight">
                      Training Run ID
                    </h3>
                    <p className="text-sm text-muted-foreground">{runId}</p>
                  </div>
                  <div className="flex flex-col gap-1">
                    <h3 className="text-sm font-medium tracking-tight">
                      Status
                    </h3>
                    <p className="text-sm text-muted-foreground">Active</p>
                  </div>
                  <div className="flex flex-col gap-1">
                    <h3 className="text-sm font-medium tracking-tight">
                      Last Training Request Time
                    </h3>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <time dateTime="2025-12-14T09:46:34.278213Z">
                        15 hours ago
                      </time>
                    </div>
                  </div>
                </div>

                <h2 className="text-lg font-semibold tracking-tight mt-10 mb-3">
                  Checkpoints
                </h2>
                <div className="mt-3 overflow-x-auto">
                  <table className="w-full caption-bottom text-sm border-collapse">
                    <thead className="[&_tr]:border-b [&_tr]:hover:bg-transparent">
                      <tr className="border-b border-[var(--alpha-10)] transition-colors hover:bg-[var(--neutral-100)]">
                        <th className="p-4 text-left font-medium text-table-header uppercase">
                          Checkpoint ID
                        </th>
                        <th className="p-4 text-left font-medium text-table-header uppercase">
                          Type
                        </th>
                        <th className="p-4 text-left font-medium text-table-header uppercase">
                          Time
                        </th>
                        <th className="p-4 text-left font-medium text-table-header uppercase">
                          Full Path
                        </th>
                      </tr>
                    </thead>
                    <tbody className="[&_tr:last-child]:border-0">
                      <tr className="border-b border-[var(--alpha-10)] transition-colors hover:bg-[var(--neutral-100)]">
                        <td className="p-4 align-middle">
                          checkpoint-500
                        </td>
                        <td className="p-4 align-middle">
                          <Badge variant="outline">Interim</Badge>
                        </td>
                        <td className="p-4 align-middle text-muted-foreground">
                          2 hours ago
                        </td>
                        <td className="p-4 align-middle font-mono text-xs text-muted-foreground">
                          s3://bucket/path/to/checkpoint-500
                        </td>
                      </tr>
                      <tr className="border-b border-[var(--alpha-10)] transition-colors hover:bg-[var(--neutral-100)]">
                        <td className="p-4 align-middle">
                          checkpoint-1000
                        </td>
                        <td className="p-4 align-middle">
                          <Badge variant="outline">Interim</Badge>
                        </td>
                        <td className="p-4 align-middle text-muted-foreground">
                          1 hour ago
                        </td>
                        <td className="p-4 align-middle font-mono text-xs text-muted-foreground">
                          s3://bucket/path/to/checkpoint-1000
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
