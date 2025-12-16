import { getDb, Step } from "@/lib/db";

export const dynamic = "force-dynamic";

interface ImageRow {
  id: number;
  run_id: string;
  step: number;
  group_idx: number;
  path: string;
  gt_lat: number;
  gt_lon: number;
  gt_city: string | null;
  gt_country: string | null;
  timestamp: string;
  samples: string;
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const runId = url.searchParams.get("run_id");
  const delay = parseInt(url.searchParams.get("delay") || "1000"); // ms between steps

  if (!runId) {
    return new Response("Missing run_id parameter", { status: 400 });
  }

  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      try {
        const db = getDb();

        // Get all steps for this run
        const allSteps = db
          .prepare("SELECT * FROM steps WHERE run_id = ? ORDER BY step ASC")
          .all(runId) as Step[];

        // Get all unique step numbers that have images
        const stepNumbers = db
          .prepare("SELECT DISTINCT step FROM images WHERE run_id = ? ORDER BY step ASC")
          .all(runId) as { step: number }[];

        db.close();

        if (allSteps.length === 0) {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ type: "error", message: "No data for this run" })}\n\n`)
          );
          controller.close();
          return;
        }

        // Send initial message
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({ type: "replay_start", total_steps: allSteps.length })}\n\n`)
        );

        // Replay each step with delay
        for (let i = 0; i < allSteps.length; i++) {
          const step = allSteps[i];

          // Send step data
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ type: "steps", data: allSteps.slice(0, i + 1) })}\n\n`)
          );

          // Get images for this step
          const db2 = getDb();
          const images = db2
            .prepare(
              `
              SELECT i.*,
                     json_group_array(json_object(
                       'id', s.id,
                       'sample_idx', s.sample_idx,
                       'pred_lat', s.pred_lat,
                       'pred_lon', s.pred_lon,
                       'pred_text', s.pred_text,
                       'distance_km', s.distance_km,
                       'reward', s.reward,
                       'format_valid', s.format_valid,
                       'mean_logprob', s.mean_logprob
                     )) as samples
              FROM images i
              LEFT JOIN samples s ON s.image_id = i.id
              WHERE i.run_id = ? AND i.step = ?
              GROUP BY i.id
              ORDER BY i.group_idx ASC
            `
            )
            .all(runId, step.step) as ImageRow[];
          db2.close();

          if (images.length > 0) {
            const parsed = images.map((img) => ({
              ...img,
              samples: JSON.parse(img.samples),
            }));
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ type: "images", data: parsed })}\n\n`)
            );
          }

          // Wait before next step (except for last one)
          if (i < allSteps.length - 1) {
            await new Promise((resolve) => setTimeout(resolve, delay));
          }
        }

        // Send completion message
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({ type: "replay_end" })}\n\n`)
        );
        controller.close();
      } catch (e) {
        console.error("Replay error:", e);
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({ type: "error", message: String(e) })}\n\n`)
        );
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
