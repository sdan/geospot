import Database from "better-sqlite3";
import path from "path";

const DB_PATH = path.join(process.cwd(), "live.db");

export function getDb() {
  return new Database(DB_PATH, { readonly: true });
}

// Type definitions matching the Python schema

export interface Run {
  id: string;
  name: string;
  type: "sft" | "rl";
  started_at: string;
  config: string; // JSON string
}

export interface Step {
  id: number;
  run_id: string;
  step: number;
  timestamp: string;
  loss: number | null;
  mean_reward: number | null;
  mean_distance_km: number | null;
  num_tokens: number | null;
  num_datums: number | null;
  learning_rate: number | null;
  coord_tau: number | null;
  elapsed_s: number | null;
}

export interface Image {
  id: number;
  run_id: string;
  step: number;
  group_idx: number;
  path: string; // "/live/images/{run_id}/{step}_{group}.jpg"
  gt_lat: number;
  gt_lon: number;
  gt_city: string | null;
  gt_country: string | null;
  timestamp: string;
}

export interface Sample {
  id: number;
  image_id: number;
  sample_idx: number; // 0-15 within group
  pred_lat: number | null;
  pred_lon: number | null;
  pred_text: string | null;
  distance_km: number | null;
  reward: number;
  format_valid: number; // 0 or 1
  mean_logprob: number | null;
}

// Joined view for frontend - image with all its samples
export interface ImageWithSamples extends Image {
  samples: Sample[];
}
