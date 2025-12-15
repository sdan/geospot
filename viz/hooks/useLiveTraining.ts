"use client";

import { useState, useEffect } from "react";
import type { Step, Image, Sample } from "@/lib/db";

export interface ImageWithSamples extends Image {
  samples: Sample[];
}

export function useLiveTraining(runId: string | null) {
  const [steps, setSteps] = useState<Step[]>([]);
  const [images, setImages] = useState<ImageWithSamples[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!runId) return;

    const eventSource = new EventSource(`/api/live/stream?run_id=${runId}`);

    eventSource.onopen = () => setConnected(true);
    eventSource.onerror = () => setConnected(false);

    eventSource.onmessage = (event) => {
      const { type, data } = JSON.parse(event.data);
      if (type === "steps") {
        setSteps((prev) => [...prev, ...data]);
      } else if (type === "images") {
        setImages(data); // Replace with latest step's images
      }
    };

    return () => eventSource.close();
  }, [runId]);

  return { steps, images, connected };
}
