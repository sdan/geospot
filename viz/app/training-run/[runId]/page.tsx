"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/ui/sidebar";
import { Badge } from "@/components/ui/badge";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Map, Marker } from "pigeon-maps";
import { useLiveTraining, ImageWithSamples } from "@/hooks/useLiveTraining";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  AreaChart,
  Area,
} from "recharts";
import type { Run } from "@/lib/db";

function rewardColor(reward: number): string {
  if (reward > 0.5) return "#34c759";
  if (reward > 0.2) return "#ff9500";
  return "#ff3b30";
}

function distanceColor(distance: number | null): string {
  if (distance === null) return "#888";
  if (distance < 25) return "#34c759";
  if (distance < 100) return "#5ac8fa";
  if (distance < 500) return "#ff9500";
  return "#ff3b30";
}

function ImageCard({ image }: { image: ImageWithSamples }) {
  const [selectedSample, setSelectedSample] = useState(0);
  const validSamples = image.samples.filter(
    (s) => s.pred_lat !== null && s.pred_lon !== null
  );

  return (
    <Card className="overflow-hidden">
      <div className="grid grid-cols-2 gap-0">
        <div className="relative aspect-[4/3] bg-muted">
          <img
            src={image.path}
            alt={`Step ${image.step} Group ${image.group_idx}`}
            className="w-full h-full object-cover"
          />
          <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
            {image.gt_city || "Unknown"}, {image.gt_country || "Unknown"}
          </div>
          <div className="absolute top-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded font-mono">
            Step {image.step}
          </div>
        </div>

        <div className="aspect-[4/3]">
          <Map center={[image.gt_lat, image.gt_lon]} zoom={3} height={200}>
            {/* Ground truth marker */}
            <Marker anchor={[image.gt_lat, image.gt_lon]} color="#ff3b30" />
            {/* All prediction markers */}
            {validSamples.map((sample) => (
              <Marker
                key={sample.id}
                anchor={[sample.pred_lat!, sample.pred_lon!]}
                color={distanceColor(sample.distance_km)}
              />
            ))}
          </Map>
        </div>
      </div>

      <CardContent className="p-3 bg-muted/50">
        <div className="flex flex-wrap gap-1 mb-2">
          {image.samples.map((sample, i) => (
            <button
              key={sample.id}
              onClick={() => setSelectedSample(i)}
              className={`w-6 h-6 text-xs font-mono rounded border transition-all ${
                selectedSample === i
                  ? "bg-foreground text-background"
                  : "bg-background border-border hover:border-foreground"
              }`}
              style={{
                borderColor: sample.format_valid ? rewardColor(sample.reward) : "#888",
              }}
            >
              {i}
            </button>
          ))}
        </div>

        {image.samples[selectedSample] && (
          <div className="text-xs font-mono grid grid-cols-3 gap-2">
            <div>
              <span className="text-muted-foreground">Distance: </span>
              <span style={{ color: distanceColor(image.samples[selectedSample].distance_km) }}>
                {image.samples[selectedSample].distance_km?.toFixed(0) ?? "N/A"} km
              </span>
            </div>
            <div>
              <span className="text-muted-foreground">Reward: </span>
              <span style={{ color: rewardColor(image.samples[selectedSample].reward) }}>
                {image.samples[selectedSample].reward.toFixed(4)}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground">Format: </span>
              <span>{image.samples[selectedSample].format_valid ? "✓" : "✗"}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function TrainingRunDetailsPage({
  params,
}: {
  params: { runId: string };
}) {
  const [activeNav, setActiveNav] = useState("training-runs");
  const runId = decodeURIComponent(params.runId);

  const [run, setRun] = useState<Run | null>(null);
  const { steps, images, connected } = useLiveTraining(runId);

  // Fetch run metadata
  useEffect(() => {
    fetch(`/api/runs/${runId}`)
      .then((r) => r.json())
      .then((data) => setRun(data.run))
      .catch(() => {});
  }, [runId]);

  const config = run?.config ? JSON.parse(run.config) : {};
  const latestStep = steps[steps.length - 1];

  // Chart data
  const chartData = steps.map((s) => ({
    step: s.step,
    reward: s.mean_reward ?? 0,
    distance: s.mean_distance_km ?? 0,
    tau: s.coord_tau ?? 0,
  }));

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar activeItem={activeNav} onNavigate={(item) => setActiveNav(item)} />

      <main className="flex-1 overflow-auto">
        <div className="p-6">
          {/* Header */}
          <header className="pb-6 border-b border-border mb-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-3">
                  <h1 className="text-2xl font-semibold tracking-tight">{runId}</h1>
                  <Badge variant={connected ? "green" : "destructive"}>
                    {connected ? "Live" : "Disconnected"}
                  </Badge>
                  {run?.type && (
                    <Badge variant="outline" className="uppercase">
                      {run.type}
                    </Badge>
                  )}
                </div>
                <p className="mt-1 text-sm text-muted-foreground">
                  {run?.name || "Loading..."} • Started {run?.started_at ? new Date(run.started_at).toLocaleString() : "..."}
                </p>
              </div>

              {latestStep && (
                <div className="text-right">
                  <div className="text-2xl font-mono font-bold">
                    Step {latestStep.step}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {latestStep.mean_reward?.toFixed(4)} reward • {latestStep.mean_distance_km?.toFixed(0)} km
                  </div>
                </div>
              )}
            </div>
          </header>

          {/* Config */}
          {Object.keys(config).length > 0 && (
            <Card className="mb-6">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Configuration</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-4 gap-4 text-sm">
                  {Object.entries(config).map(([key, value]) => (
                    <div key={key}>
                      <span className="text-muted-foreground">{key}: </span>
                      <span className="font-mono">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Charts */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Mean Reward</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={180}>
                  <AreaChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="step" stroke="#888" fontSize={10} />
                    <YAxis stroke="#888" fontSize={10} domain={[0, 1]} />
                    <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333" }} />
                    <Area type="monotone" dataKey="reward" stroke="#34c759" fill="#34c75922" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Mean Distance (km)</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={180}>
                  <AreaChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="step" stroke="#888" fontSize={10} />
                    <YAxis stroke="#888" fontSize={10} />
                    <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333" }} />
                    <Area type="monotone" dataKey="distance" stroke="#ff9500" fill="#ff950022" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Tau (Curriculum)</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={180}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="step" stroke="#888" fontSize={10} />
                    <YAxis stroke="#888" fontSize={10} />
                    <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333" }} />
                    <Line type="monotone" dataKey="tau" stroke="#5ac8fa" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Latest Predictions */}
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              Latest Predictions {latestStep && `(Step ${latestStep.step})`}
            </h2>
            <span className="text-sm text-muted-foreground">
              {images.length} images • {images.reduce((acc, img) => acc + img.samples.length, 0)} samples
            </span>
          </div>

          {images.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {images.map((img) => (
                <ImageCard key={img.id} image={img} />
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="py-12 text-center text-muted-foreground">
                {connected ? "Waiting for training data..." : "Not connected to training run"}
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
}
