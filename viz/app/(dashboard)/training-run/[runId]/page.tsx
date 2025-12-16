"use client";

import { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Map, Marker } from "pigeon-maps";
import { useLiveTraining, ImageWithSamples } from "@/hooks/useLiveTraining";
import { useReplayTraining } from "@/hooks/useReplayTraining";
import { Button } from "@/components/ui/button";
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
import type { Run, Sample } from "@/lib/db";

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

function PredictionModal({
  image,
  open,
  onOpenChange,
}: {
  image: ImageWithSamples;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [idx, setIdx] = useState(0);
  const sample = image.samples[idx];
  const validSamples = image.samples.filter((s) => s.pred_lat && s.pred_lon);

  // Calculate map center based on selected sample or ground truth
  const mapCenter: [number, number] = sample?.pred_lat && sample?.pred_lon
    ? [(image.gt_lat + sample.pred_lat) / 2, (image.gt_lon + sample.pred_lon) / 2]
    : [image.gt_lat, image.gt_lon];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            <div className="flex items-center gap-2">
              <span className="text-xl font-semibold">{image.gt_city || "Unknown"}, {image.gt_country || "Unknown"}</span>
              <Badge variant="outline" className="text-xs">Step {image.step}</Badge>
              <Badge variant="outline" className="text-xs">Group {image.group_idx}</Badge>
            </div>
          </DialogTitle>
        </DialogHeader>

        <div className="grid grid-cols-2 gap-6 mt-2">
          {/* Left: Image */}
          <div className="space-y-4">
            <div className="relative aspect-[4/3] bg-muted rounded-lg overflow-hidden">
              <img
                src={image.path}
                alt=""
                className="w-full h-full object-cover"
              />
              <div className="absolute bottom-3 left-3 bg-black/70 text-white text-sm px-3 py-1.5 rounded">
                Ground Truth: {image.gt_lat.toFixed(4)}, {image.gt_lon.toFixed(4)}
              </div>
            </div>

            {/* Sample selector */}
            <div>
              <div className="text-sm text-muted-foreground mb-2">Select prediction (0-{image.samples.length - 1})</div>
              <div className="flex flex-wrap gap-1.5">
                {image.samples.map((s, i) => (
                  <button
                    key={s.id}
                    onClick={() => setIdx(i)}
                    className={`w-8 h-8 text-sm font-mono rounded border-2 transition-all ${
                      idx === i
                        ? "bg-foreground text-background scale-110"
                        : "bg-background hover:bg-muted"
                    }`}
                    style={{ borderColor: s.format_valid ? rewardColor(s.reward) : "#666" }}
                  >
                    {i}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right: Map + Details */}
          <div className="space-y-4">
            <div className="h-[320px] rounded-lg overflow-hidden border border-border">
              <Map center={mapCenter} zoom={2} height={320}>
                {/* Ground truth (red) */}
                <Marker anchor={[image.gt_lat, image.gt_lon]} color="#ff3b30" />
                {/* All predictions (colored by distance) */}
                {validSamples.map((s) => (
                  <Marker
                    key={s.id}
                    anchor={[s.pred_lat!, s.pred_lon!]}
                    color={image.samples.indexOf(s) === idx ? "#fff" : distanceColor(s.distance_km)}
                    onClick={() => setIdx(image.samples.indexOf(s))}
                  />
                ))}
              </Map>
            </div>

            {/* Selected sample details */}
            {sample && (
              <div className="bg-muted/50 rounded-lg p-4 space-y-3">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Distance: </span>
                    <span className="font-mono font-bold" style={{ color: distanceColor(sample.distance_km) }}>
                      {sample.distance_km?.toFixed(1) ?? "N/A"} km
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Reward: </span>
                    <span className="font-mono font-bold" style={{ color: rewardColor(sample.reward) }}>
                      {sample.reward.toFixed(4)}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Predicted: </span>
                    <span className="font-mono">
                      {sample.pred_lat?.toFixed(4) ?? "N/A"}, {sample.pred_lon?.toFixed(4) ?? "N/A"}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Format: </span>
                    <span className={sample.format_valid ? "text-green-500" : "text-red-500"}>
                      {sample.format_valid ? "Valid" : "Invalid"}
                    </span>
                  </div>
                </div>

                {/* Raw prediction text */}
                {sample.pred_text && (
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">Model output:</div>
                    <pre className="text-xs font-mono bg-black/30 p-3 rounded overflow-x-auto max-h-32 whitespace-pre-wrap">
                      {sample.pred_text}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Distance legend */}
        <div className="flex items-center gap-4 text-xs text-muted-foreground pt-2 border-t border-border">
          <span>Distance:</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#34c759]" /> &lt;25km</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#5ac8fa]" /> &lt;100km</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#ff9500]" /> &lt;500km</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#ff3b30]" /> &gt;500km</span>
          <span className="ml-auto">Red marker = Ground truth</span>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function ImageCard({ image, onClick }: { image: ImageWithSamples; onClick: () => void }) {
  const [idx, setIdx] = useState(0);
  const sample = image.samples[idx];

  return (
    <Card
      className="overflow-hidden cursor-pointer transition-all hover:ring-2 hover:ring-primary/50 hover:shadow-lg"
      onClick={onClick}
    >
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
      <CardContent className="p-3 bg-muted/50" onClick={(e) => e.stopPropagation()}>
        <div className="flex flex-wrap gap-1 mb-2">
          {image.samples.map((s, i) => (
            <button
              key={s.id}
              onClick={(e) => { e.stopPropagation(); setIdx(i); }}
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
  const [selectedGroupIdx, setSelectedGroupIdx] = useState<number | null>(null);
  const [isReplayMode, setIsReplayMode] = useState(false);

  const live = useLiveTraining(runId);
  const replay = useReplayTraining(runId);

  // Use replay data when in replay mode, otherwise live
  const steps = isReplayMode ? replay.steps : live.steps;
  const images = isReplayMode ? replay.images : live.images;
  const connected = isReplayMode ? replay.isReplaying : live.connected;

  // Get selected image from current images array (stays synced during replay)
  const selectedImage = selectedGroupIdx !== null ? images.find(img => img.group_idx === selectedGroupIdx) || images[0] : null;

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
              <Badge variant={connected ? "green" : "destructive"}>
                {isReplayMode ? (replay.isReplaying ? "Replaying" : "Replay Done") : (connected ? "Live" : "Disconnected")}
              </Badge>
              {run?.type && <Badge variant="outline" className="uppercase">{run.type}</Badge>}
              {isReplayMode && replay.totalSteps > 0 && (
                <span className="text-sm text-muted-foreground">
                  {replay.replayProgress}/{replay.totalSteps}
                </span>
              )}
            </div>
            <p className="mt-1 text-sm text-muted-foreground">
              {run?.name || "Loading..."} • Started {run?.started_at ? new Date(run.started_at).toLocaleString() : "..."}
            </p>
          </div>
          <div className="flex items-center gap-4">
            {/* Replay controls */}
            <div className="flex items-center gap-2">
              {!isReplayMode ? (
                <Button
                  variant="primary"
                  size="sm"
                  className="text-white"
                  onClick={() => {
                    setIsReplayMode(true);
                    replay.startReplay(300);
                  }}
                >
                  Replay
                </Button>
              ) : (
                <Button
                  variant="primary"
                  size="sm"
                  className="text-white"
                  onClick={() => {
                    setIsReplayMode(false);
                    replay.stopReplay();
                  }}
                >
                  Stop Replay
                </Button>
              )}
            </div>
            {latest && (
              <div className="text-right">
                <div className="text-2xl font-mono font-bold">Step {latest.step}</div>
                <div className="text-sm text-muted-foreground">{latest.mean_reward?.toFixed(4)} reward • {latest.mean_distance_km?.toFixed(0)} km</div>
              </div>
            )}
          </div>
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
          {images.map((img) => (
            <ImageCard
              key={img.id}
              image={img}
              onClick={() => setSelectedGroupIdx(img.group_idx)}
            />
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            {connected ? "Waiting for training data..." : "Not connected"}
          </CardContent>
        </Card>
      )}

      {/* Prediction detail modal */}
      {selectedImage && (
        <PredictionModal
          image={selectedImage}
          open={!!selectedImage}
          onOpenChange={(open) => !open && setSelectedGroupIdx(null)}
        />
      )}
    </div>
  );
}
