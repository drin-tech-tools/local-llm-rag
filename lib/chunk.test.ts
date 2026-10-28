import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { chunkPages } from "./chunk";

describe("chunkPages", () => {
  it("keeps page numbers and overlaps windows", () => {
    const words = Array.from({ length: 300 }, (_, i) => `w${i}`).join(" ");
    const chunks = chunkPages(["", words], 200, 40);
    assert.equal(chunks.length, 2);
    assert.ok(chunks.every((c) => c.page === 2));
    assert.ok(chunks[1].content.startsWith("w160 "));
    assert.deepEqual(chunks.map((c) => c.index), [0, 1]);
  });

  it("skips near-empty pages", () => {
    assert.equal(chunkPages(["  ", "tiny"]).length, 0);
  });
});
