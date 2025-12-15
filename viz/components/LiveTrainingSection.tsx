"use client";

import { useState, useEffect } from "react";
import { Map, Marker } from "pigeon-maps";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useLiveTraining, ImageWithSamples } from "@/hooks/useLiveTraining";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
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
        </div>

        <div className="aspect-[4/3]">
          <Map center={[image.gt_lat, image.gt_lon]} zoom={4} height={200}>
            <Marker anchor={[image.gt_lat, image.gt_lon]} color="#ff3b30" />
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
          <div className="text-xs font-mono flex gap-4">
            <span>
              Dist:{" "}
              <span style={{ color: distanceColor(image.samples[selectedSample].distance_km) }}>
                {image.samples[selectedSample].distance_km?.toFixed(0) ?? "N/A"} km
              </span>
            </span>
            <span>
              Reward:{" "}
              <span style={{ color: rewardColor(image.samples[selectedSample].reward) }}>
                {image.samples[selectedSample].reward.toFixed(3)}
              </span>
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function LiveTrainingSection() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const { steps, images, connected } = useLiveTraining(selectedRunId);

  // Fetch available runs
  useEffect(() => {
    fetch("/api/runs")
      .then((r) => r.json())
      .then((data) => {
        setRuns(data);
        if (data.length > 0 && !selectedRunId) {
          setSelectedRunId(data[0].id);
        }
      })
      .catch(() => {});
  }, []);

  const chartData = steps.map((s) => ({
    step: s.step,
    reward: s.mean_reward ?? 0,
    distance: s.mean_distance_km ?? 0,
  }));

  const latestStep = steps[steps.length - 1];

  if (runs.length === 0) {
    return (
      <Card className="mb-6">
        <CardContent className="py-8 text-center text-muted-foreground">
          No training runs yet. Start training to see live data.
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="mb-8">
      {/* Run selector */}
      <div className="flex items-center gap-4 mb-4">
        <h2 className="text-lg font-semibold">Live Training</h2>
        <select
          value={selectedRunId || ""}
          onChange={(e) => setSelectedRunId(e.target.value)}
          className="bg-background border border-border rounded px-3 py-1.5 text-sm"
        >
          {runs.map((run) => (
            <option key={run.id} value={run.id}>
              {run.id} - {run.name} ({run.type})
            </option>
          ))}
        </select>
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`}
          />
          <span className="text-xs text-muted-foreground">
            {connected ? "Connected" : "Disconnected"}
          </span>
        </div>
        {latestStep && (
          <Badge variant="blue">
            Step {latestStep.step} | {latestStep.mean_reward?.toFixed(3)} reward
          </Badge>
        )}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Mean Reward</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="step" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} domain={[0, 1]} />
                <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333" }} />
                <Line type="monotone" dataKey="reward" stroke="#34c759" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Mean Distance (km)</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="step" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333" }} />
                <Line type="monotone" dataKey="distance" stroke="#ff9500" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Images grid */}
      {images.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {images.slice(0, 6).map((img) => (
            <ImageCard key={img.id} image={img} />
          ))}
        </div>
      ) : (
        <div className="text-center text-muted-foreground py-4 text-sm">
          Waiting for training data...
        </div>
      )}
    </div>
  );
}
