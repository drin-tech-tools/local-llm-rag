import { deleteDocument } from "@/lib/store";

export const runtime = "nodejs";

export async function DELETE(_req: Request, { params }: { params: { id: string } }) {
  const removed = await deleteDocument(params.id);
  if (!removed) return Response.json({ error: "Document not found" }, { status: 404 });
  return Response.json({ ok: true });
}
