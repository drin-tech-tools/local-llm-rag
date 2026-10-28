import { config } from "@/lib/config";
import { listDocuments } from "@/lib/store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const { embedModel } = config();
  const documents = (await listDocuments()).map((d) => ({
    ...d,
    // Embedded with another model: not searchable until re-uploaded.
    stale: d.embedModel !== embedModel,
  }));
  return Response.json({ documents });
}
