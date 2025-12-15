"use client";

import { useState } from "react";
import { Map, Marker, Overlay } from "pigeon-maps";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Sidebar } from "@/components/ui/sidebar";
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

function haversine(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

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

interface ImageCardProps {
  image: ImageWithSamples;
}

function ImageCard({ image }: ImageCardProps) {
  const [selectedSample, setSelectedSample] = useState(0);

  // Filter valid predictions
  const validSamples = image.samples.filter(
    (s) => s.pred_lat !== null && s.pred_lon !== null
  );

  return (
    <Card className="overflow-hidden">
      <div className="grid grid-cols-2 gap-0">
        {/* Left: Image */}
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

        {/* Right: Map with predictions */}
        <div className="aspect-[4/3]">
          <Map
            center={[image.gt_lat, image.gt_lon]}
            zoom={4}
            height={300}
          >
            {/* Ground truth marker (red) */}
            <Marker anchor={[image.gt_lat, image.gt_lon]} color="#ff3b30" />

            {/* Prediction markers */}
            {validSamples.map((sample, i) => (
              <Marker
                key={sample.id}
                anchor={[sample.pred_lat!, sample.pred_lon!]}
                color={distanceColor(sample.distance_km)}
                onClick={() => setSelectedSample(i)}
              />
            ))}

            {/* Line from GT to selected prediction */}
            {validSamples[selectedSample] && (
              <Overlay
                anchor={[image.gt_lat, image.gt_lon]}
                offset={[0, 0]}
              >
                <svg
                  style={{
                    position: "absolute",
                    width: "100%",
                    height: "100%",
                    pointerEvents: "none",
                  }}
                >
                  <line
                    x1="0"
                    y1="0"
                    x2="50"
                    y2="50"
                    stroke="#ff3b30"
                    strokeWidth="2"
                    strokeDasharray="4"
                  />
                </svg>
              </Overlay>
            )}
          </Map>
        </div>
      </div>

      {/* Sample selector */}
      <CardContent className="p-3 bg-muted/50">
        <div className="flex flex-wrap gap-1">
          {image.samples.map((sample, i) => (
            <button
              key={sample.id}
              onClick={() => setSelectedSample(i)}
              className={`w-8 h-8 text-xs font-mono rounded border transition-all ${
                selectedSample === i
                  ? "bg-foreground text-background border-foreground"
                  : "bg-background border-border hover:border-foreground"
              }`}
              style={{
                borderColor: sample.format_valid
                  ? rewardColor(sample.reward)
                  : "#888",
              }}
            >
              {i}
            </button>
          ))}
        </div>

        {/* Selected sample details */}
        {image.samples[selectedSample] && (
          <div className="mt-3 text-xs font-mono space-y-1">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Distance:</span>
              <span
                style={{
                  color: distanceColor(image.samples[selectedSample].distance_km),
                }}
              >
                {image.samples[selectedSample].distance_km?.toFixed(1) ?? "N/A"} km
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Reward:</span>
              <span
                style={{
                  color: rewardColor(image.samples[selectedSample].reward),
                }}
              >
                {image.samples[selectedSample].reward.toFixed(4)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Format:</span>
              <span>
                {image.samples[selectedSample].format_valid ? "Valid" : "Invalid"}
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function LiveDashboard({
  params,
}: {
  params: { runId: string };
}) {
  const { runId } = params;
  const { steps, images, connected } = useLiveTraining(runId);

  // Chart data from steps
  const chartData = steps.map((s) => ({
    step: s.step,
    reward: s.mean_reward ?? 0,
    distance: s.mean_distance_km ?? 0,
  }));

  const latestStep = steps[steps.length - 1];

  return (
    <div className="min-h-screen bg-background p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Live Training: {runId}</h1>
          <p className="text-muted-foreground text-sm">
            Step {latestStep?.step ?? 0} |{" "}
            {latestStep?.mean_reward?.toFixed(4) ?? "0.0000"} avg reward
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div
            className={`w-3 h-3 rounded-full ${
              connected ? "bg-green-500" : "bg-red-500"
            }`}
          />
          <span className="text-sm text-muted-foreground">
            {connected ? "Connected" : "Disconnected"}
          </span>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Mean Reward</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="step" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{ background: "#1a1a1a", border: "1px solid #333" }}
                />
                <Line
                  type="monotone"
                  dataKey="reward"
                  stroke="#34c759"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Mean Distance (km)</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="step" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip
                  contentStyle={{ background: "#1a1a1a", border: "1px solid #333" }}
                />
                <Line
                  type="monotone"
                  dataKey="distance"
                  stroke="#ff9500"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Image grid */}
      <h2 className="text-lg font-semibold mb-4">
        Latest Predictions (Step {latestStep?.step ?? 0})
      </h2>
      {images.length === 0 ? (
        <div className="text-center text-muted-foreground py-12">
          Waiting for training data...
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {images.map((img) => (
            <ImageCard key={img.id} image={img} />
          ))}
        </div>
      )}
    </div>
  );
}
