"use client";

import { useState, useEffect } from "react";
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

function rewardColor(r: number) {
  if (r > 0.5) return "#34c759";
  if (r > 0.2) return "#ff9500";
  return "#ff3b30";
}

function distanceColor(d: number | null) {
  if (d === null) return "#888";
  if (d < 25) return "#34c759";
  if (d < 100) return "#5ac8fa";
  if (d < 500) return "#ff9500";
  return "#ff3b30";
}

function ImageCard({ image }: { image: ImageWithSamples }) {
  const [idx, setIdx] = useState(0);
  const sample = image.samples[idx];

  return (
    <Card className="overflow-hidden">
      <div className="grid grid-cols-2 gap-0">
        <div className="relative aspect-[4/3] bg-muted">
          <img src={image.path} alt="" className="w-full h-full object-cover" />
          <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
            {image.gt_city || "Unknown"}, {image.gt_country || "Unknown"}
          </div>
        </div>
        <div className="aspect-[4/3]">
          <Map center={[image.gt_lat, image.gt_lon]} zoom={3} height={200}>
            <Marker anchor={[image.gt_lat, image.gt_lon]} color="#ff3b30" />
            {image.samples
              .filter((s) => s.pred_lat && s.pred_lon)
              .map((s) => (
                <Marker key={s.id} anchor={[s.pred_lat!, s.pred_lon!]} color={distanceColor(s.distance_km)} />
              ))}
          </Map>
        </div>
      </div>
      <CardContent className="p-3 bg-muted/50">
        <div className="flex flex-wrap gap-1 mb-2">
          {image.samples.map((s, i) => (
            <button
              key={s.id}
              onClick={() => setIdx(i)}
              className={`w-6 h-6 text-xs font-mono rounded border ${idx === i ? "bg-foreground text-background" : "bg-background"}`}
              style={{ borderColor: s.format_valid ? rewardColor(s.reward) : "#888" }}
            >
              {i}
            </button>
          ))}
        </div>
        {sample && (
          <div className="text-xs font-mono grid grid-cols-3 gap-2">
            <div>
              <span className="text-muted-foreground">Distance: </span>
              <span style={{ color: distanceColor(sample.distance_km) }}>{sample.distance_km?.toFixed(0) ?? "N/A"} km</span>
            </div>
            <div>
              <span className="text-muted-foreground">Reward: </span>
              <span style={{ color: rewardColor(sample.reward) }}>{sample.reward.toFixed(4)}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Format: </span>
              <span>{sample.format_valid ? "✓" : "✗"}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function TrainingRunPage({ params }: { params: { runId: string } }) {
  const runId = decodeURIComponent(params.runId);
  const [run, setRun] = useState<Run | null>(null);
  const { steps, images, connected } = useLiveTraining(runId);

  useEffect(() => {
    fetch(`/api/runs/${runId}`)
      .then((r) => r.json())
      .then((d) => setRun(d.run))
      .catch(() => {});
  }, [runId]);

  const config = run?.config ? JSON.parse(run.config) : {};
  const latest = steps[steps.length - 1];
  const chartData = steps.map((s) => ({ step: s.step, reward: s.mean_reward ?? 0, distance: s.mean_distance_km ?? 0, tau: s.coord_tau ?? 0 }));

  return (
    <div className="p-6">
      {/* Header */}
      <header className="pb-6 border-b border-border mb-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-semibold tracking-tight">{runId}</h1>
              <Badge variant={connected ? "green" : "destructive"}>{connected ? "Live" : "Disconnected"}</Badge>
              {run?.type && <Badge variant="outline" className="uppercase">{run.type}</Badge>}
            </div>
            <p className="mt-1 text-sm text-muted-foreground">
              {run?.name || "Loading..."} • Started {run?.started_at ? new Date(run.started_at).toLocaleString() : "..."}
            </p>
          </div>
          {latest && (
            <div className="text-right">
              <div className="text-2xl font-mono font-bold">Step {latest.step}</div>
              <div className="text-sm text-muted-foreground">{latest.mean_reward?.toFixed(4)} reward • {latest.mean_distance_km?.toFixed(0)} km</div>
            </div>
          )}
        </div>
      </header>

      {/* Config */}
      {Object.keys(config).length > 0 && (
        <Card className="mb-6">
          <CardHeader className="pb-2"><CardTitle className="text-sm">Configuration</CardTitle></CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4 text-sm">
              {Object.entries(config).map(([k, v]) => (
                <div key={k}><span className="text-muted-foreground">{k}: </span><span className="font-mono">{String(v)}</span></div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Charts */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm">Mean Reward</CardTitle></CardHeader>
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
          <CardHeader className="pb-2"><CardTitle className="text-sm">Mean Distance (km)</CardTitle></CardHeader>
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
          <CardHeader className="pb-2"><CardTitle className="text-sm">Tau</CardTitle></CardHeader>
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

      {/* Images */}
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Latest Predictions {latest && `(Step ${latest.step})`}</h2>
        <span className="text-sm text-muted-foreground">{images.length} images</span>
      </div>

      {images.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {images.map((img) => <ImageCard key={img.id} image={img} />)}
        </div>
      ) : (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            {connected ? "Waiting for training data..." : "Not connected"}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
