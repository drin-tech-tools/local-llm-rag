import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, before, describe, it } from "node:test";
import * as store from "./store";

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "rag-store-"));
// config() reads DATA_DIR on every call, so setting it here is early enough.
process.env.DATA_DIR = dir;

before(() => {
  delete (globalThis as { __ragStore?: unknown }).__ragStore;
});
after(() => fs.rmSync(dir, { recursive: true, force: true }));

describe("file store", () => {
  it("adds, searches, lists and deletes documents", async () => {
    const a = await store.addDocument({ name: "a.pdf", pages: 2, embedModel: "m1" }, [
      { page: 1, index: 0, content: "cats", vector: [1, 0, 0] },
      { page: 2, index: 1, content: "dogs", vector: [0, 1, 0] },
    ]);
    const b = await store.addDocument({ name: "b.pdf", pages: 1, embedModel: "m2" }, [
      { page: 1, index: 0, content: "other model", vector: [1, 0, 0] },
    ]);
    assert.equal(a.chunks, 2);
    assert.equal(a.dims, 3);

    const hits = await store.search([0.9, 0.1, 0], { embedModel: "m1", k: 5 });
    assert.deepEqual(hits.map((h) => h.content), ["cats", "dogs"]);
    assert.equal(hits[0].documentName, "a.pdf");
    assert.ok(hits.every((h) => h.documentId === a.id), "other-model chunks are excluded");

    const scoped = await store.search([1, 0, 0], { embedModel: "m2", k: 5, documentId: b.id });
    assert.deepEqual(scoped.map((h) => h.content), ["other model"]);

    assert.ok(fs.existsSync(path.join(dir, "store.json")));
    delete (globalThis as { __ragStore?: unknown }).__ragStore; // force a reload from disk
    assert.deepEqual((await store.listDocuments()).map((d) => d.name).sort(), ["a.pdf", "b.pdf"]);

    assert.equal(await store.deleteDocument(a.id), true);
    assert.equal(await store.deleteDocument(a.id), false);
    assert.deepEqual(await store.search([1, 0, 0], { embedModel: "m1", k: 5 }), []);
  });

  it("does not lose writes when uploads finish together", async () => {
    await Promise.all(
      Array.from({ length: 5 }, (_, i) =>
        store.addDocument({ name: `c${i}.pdf`, pages: 1, embedModel: "m3" }, [
          { page: 1, index: 0, content: `c${i}`, vector: [i, 1] },
        ]),
      ),
    );
    const names = (await store.listDocuments()).map((d) => d.name).filter((n) => n.startsWith("c"));
    assert.equal(names.length, 5);
  });
});
