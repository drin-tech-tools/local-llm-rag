import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { cosine, decodeVector, encodeVector, topK } from "./vector";

describe("cosine", () => {
  it("is 1 for the same direction, 0 for orthogonal, -1 for opposite", () => {
    assert.equal(cosine([1, 2, 3], [2, 4, 6]).toFixed(6), "1.000000");
    assert.equal(cosine([1, 0], [0, 1]), 0);
    assert.equal(cosine([1, 0], [-1, 0]), -1);
  });

  it("returns 0 for a zero vector and rejects mismatched lengths", () => {
    assert.equal(cosine([0, 0], [1, 1]), 0);
    assert.throws(() => cosine([1], [1, 2]), /length mismatch/);
  });
});

describe("topK", () => {
  it("returns the best k items, highest score first", () => {
    const out = topK([{ v: 1 }, { v: 5 }, { v: 3 }], (x) => x.v, 2);
    assert.deepEqual(out.map((x) => x.v), [5, 3]);
  });
});

describe("encodeVector / decodeVector", () => {
  it("round-trips through base64 Float32", () => {
    const v = [0.25, -1.5, 3, 0.125];
    assert.deepEqual(Array.from(decodeVector(encodeVector(v))), v);
  });

  it("decodes many vectors regardless of buffer pool alignment", () => {
    for (let n = 1; n < 40; n++) {
      const v = Array.from({ length: n }, (_, i) => i / 4);
      assert.deepEqual(Array.from(decodeVector(encodeVector(v))), v);
    }
  });
});
