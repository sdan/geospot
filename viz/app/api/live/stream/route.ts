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
  samples: string; // JSON string from json_group_array
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const runId = url.searchParams.get("run_id");
  const afterStep = parseInt(url.searchParams.get("after_step") || "0");

  if (!runId) {
    return new Response("Missing run_id parameter", { status: 400 });
  }

  const encoder = new TextEncoder();
  let lastStep = afterStep;
  let isActive = true;

  const stream = new ReadableStream({
    async start(controller) {
      const poll = async () => {
        if (!isActive) return;

        try {
          const db = getDb();

          // Get new steps
          const steps = db
            .prepare(
              `
            SELECT * FROM steps
            WHERE run_id = ? AND step > ?
            ORDER BY step ASC LIMIT 100
          `
            )
            .all(runId, lastStep) as Step[];

          if (steps.length > 0) {
            lastStep = steps[steps.length - 1].step;
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ type: "steps", data: steps })}\n\n`)
            );
          }

          // Get images + samples for latest step (joined)
          const images = db
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
            .all(runId, lastStep) as ImageRow[];

          if (images.length > 0) {
            // Parse nested JSON samples
            const parsed = images.map((img) => ({
              ...img,
              samples: JSON.parse(img.samples),
            }));
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ type: "images", data: parsed })}\n\n`)
            );
          }

          db.close();
        } catch (e) {
          // DB might not exist yet, that's ok
          console.error("SSE poll error:", e);
        }

        setTimeout(poll, 500); // Poll every 500ms
      };

      poll();
    },
    cancel() {
      isActive = false;
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
