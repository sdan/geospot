import { useState, useCallback } from "react";
import type { Step } from "@/lib/db";
import type { ImageWithSamples } from "./useLiveTraining";

export function useReplayTraining(runId: string | null) {
  const [steps, setSteps] = useState<Step[]>([]);
  const [images, setImages] = useState<ImageWithSamples[]>([]);
  const [isReplaying, setIsReplaying] = useState(false);
  const [replayProgress, setReplayProgress] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);

  const startReplay = useCallback(
    (delay: number = 800) => {
      if (!runId || isReplaying) return;

      setIsReplaying(true);
      setSteps([]);
      setImages([]);
      setReplayProgress(0);

      const eventSource = new EventSource(
        `/api/replay/stream?run_id=${runId}&delay=${delay}`
      );

      eventSource.onmessage = (event) => {
        const { type, data, total_steps, message } = JSON.parse(event.data);

        if (type === "replay_start") {
          setTotalSteps(total_steps);
        } else if (type === "steps") {
          setSteps(data);
          setReplayProgress(data.length);
        } else if (type === "images") {
          setImages(data);
        } else if (type === "replay_end") {
          setIsReplaying(false);
          eventSource.close();
        } else if (type === "error") {
          console.error("Replay error:", message);
          setIsReplaying(false);
          eventSource.close();
        }
      };

      eventSource.onerror = () => {
        setIsReplaying(false);
        eventSource.close();
      };

      return () => {
        eventSource.close();
        setIsReplaying(false);
      };
    },
    [runId, isReplaying]
  );

  const stopReplay = useCallback(() => {
    setIsReplaying(false);
  }, []);

  return {
    steps,
    images,
    isReplaying,
    replayProgress,
    totalSteps,
    startReplay,
    stopReplay,
  };
}
