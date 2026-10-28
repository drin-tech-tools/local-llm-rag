import type { ChatMessage } from "./llm";
import type { Match } from "./store";

export const SYSTEM_PROMPT = `You answer questions using ONLY the numbered sources provided.
Cite sources inline like [1] or [2][3] right after each claim.
If the sources do not contain the answer, say you don't know. Do not use outside knowledge.`;

/** System rules plus one user turn holding the numbered sources and the question. */
export function buildMessages(question: string, sources: Match[]): ChatMessage[] {
  const context = sources
    .map((s, i) => `[${i + 1}] (${s.documentName}, p.${s.page})\n${s.content}`)
    .join("\n\n");
  return [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: `Sources:\n${context}\n\nQuestion: ${question}` },
  ];
}
