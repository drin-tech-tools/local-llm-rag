import { config } from "@/lib/config";
import { hasModel, listModels } from "@/lib/llm";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Is the model server up, and are both configured models pulled? Drives the status bar. */
export async function GET() {
  const { baseUrl, chatModel, embedModel } = config();
  try {
    const available = await listModels(AbortSignal.timeout(5_000));
    const chat = hasModel(available, chatModel);
    const embedding = hasModel(available, embedModel);
    return Response.json({
      ok: chat && embedding,
      baseUrl,
      chatModel: { name: chatModel, available: chat },
      embedModel: { name: embedModel, available: embedding },
    });
  } catch (err) {
    return Response.json({
      ok: false,
      baseUrl,
      chatModel: { name: chatModel, available: false },
      embedModel: { name: embedModel, available: false },
      error: err instanceof Error ? err.message : "Model server check failed",
    });
  }
}
