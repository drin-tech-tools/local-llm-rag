/**
 * `npm run doctor`: checks this machine and the model server, then warms the chat model.
 * Run it after setup and again a minute before a demo.
 */
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import { config } from "../lib/config";
import { chatStream, embed, hasModel, listModels } from "../lib/llm";

// Use the same settings as `npm run dev`, which reads .env.local.
// process.loadEnvFile needs Node 20.12+; on older versions .env.local is skipped here.
for (const file of [".env.local", ".env"]) {
  if (fs.existsSync(file) && typeof process.loadEnvFile === "function") process.loadEnvFile(file);
}

const GB = 1024 ** 3;
let failures = 0;

function ok(msg: string) {
  console.log(`  ok    ${msg}`);
}
function warn(msg: string) {
  console.log(`  warn  ${msg}`);
}
function fail(msg: string) {
  failures++;
  console.log(`  FAIL  ${msg}`);
}

/**
 * Memory available for a model. On macOS os.freemem() ignores reclaimable cache and reads
 * near zero, so use the system's own "memory free percentage" instead.
 */
function availableMemoryGb(totalGb: number): number {
  if (process.platform === "darwin") {
    try {
      const out = execFileSync("memory_pressure", { encoding: "utf8", timeout: 5_000 });
      const pct = Number(out.match(/free percentage:\s*(\d+)%/)?.[1]);
      if (Number.isFinite(pct)) return (totalGb * pct) / 100;
    } catch {
      /* fall through */
    }
  }
  return os.freemem() / GB;
}

function recommendedChatModel(totalGb: number): string {
  if (totalGb < 12) return "llama3.2:3b";
  if (totalGb < 23) return "llama3.1:8b";
  return "llama3.1:8b (or a 14B model such as qwen2.5:14b)";
}

async function main() {
  const cfg = config();
  console.log("\nMachine");

  const major = Number(process.versions.node.split(".")[0]);
  if (major >= 20) ok(`Node.js ${process.versions.node}`);
  else fail(`Node.js ${process.versions.node}; version 20 or newer is required`);

  if (process.platform === "darwin" && process.arch === "arm64") ok("Apple Silicon (models run on the GPU)");
  else if (process.platform === "darwin") warn("Intel Mac: Ollama runs on the CPU only and will be slow");
  else ok(`${process.platform}/${process.arch}`);

  const totalGb = os.totalmem() / GB;
  // Machines report slightly under their marketed size (16 GB shows as ~15.6).
  if (totalGb >= 15) ok(`${totalGb.toFixed(0)} GB memory`);
  else if (totalGb >= 7.5) warn(`${totalGb.toFixed(0)} GB memory: use CHAT_MODEL=llama3.2:3b`);
  else fail(`${totalGb.toFixed(0)} GB memory: at least 8 GB is needed`);
  console.log(`        recommended chat model for this machine: ${recommendedChatModel(totalGb)}`);
  const freeGb = availableMemoryGb(totalGb);
  if (freeGb < 6)
    warn(`only ${freeGb.toFixed(1)} GB memory free right now: close heavy apps and browser tabs before a demo`);

  try {
    const stat = fs.statfsSync(process.cwd());
    const diskGb = (stat.bavail * stat.bsize) / GB;
    if (diskGb >= 10) ok(`${diskGb.toFixed(0)} GB free disk`);
    else warn(`${diskGb.toFixed(1)} GB free disk: models need about 6 GB`);
  } catch {
    warn("could not read free disk space");
  }

  console.log(`\nModel server (${cfg.baseUrl})`);
  let available: string[];
  try {
    available = await listModels(AbortSignal.timeout(5_000));
    ok("reachable");
  } catch (err) {
    fail(err instanceof Error ? err.message : String(err));
    return;
  }

  const missing = [cfg.embedModel, cfg.chatModel].filter((m) => !hasModel(available, m));
  for (const m of [cfg.embedModel, cfg.chatModel]) {
    if (hasModel(available, m)) ok(`model ${m} is pulled`);
    else fail(`model ${m} is missing: run \`ollama pull ${m}\``);
  }
  if (missing.length) return;

  console.log("\nSmoke tests");
  try {
    const t0 = performance.now();
    const [vector] = await embed(["The quick brown fox."], "document");
    ok(`embeddings: ${vector.length} dimensions in ${Math.round(performance.now() - t0)} ms`);
  } catch (err) {
    fail(`embeddings: ${err instanceof Error ? err.message : String(err)}`);
  }

  try {
    console.log("        warming the chat model (the first load can take up to a minute)…");
    const t0 = performance.now();
    const tokens = await chatStream(
      [{ role: "user", content: "Reply with exactly one word: ready" }],
      AbortSignal.timeout(cfg.firstTokenTimeoutMs),
    );
    let firstMs = 0;
    let text = "";
    for await (const t of tokens) {
      if (!firstMs) firstMs = performance.now() - t0;
      text += t;
    }
    ok(
      `chat: "${text.trim().slice(0, 40)}", first token ${(firstMs / 1000).toFixed(1)}s, total ${((performance.now() - t0) / 1000).toFixed(1)}s`,
    );
  } catch (err) {
    fail(`chat: ${err instanceof Error ? err.message : String(err)}`);
  }
}

main()
  .catch((err) => fail(err instanceof Error ? err.message : String(err)))
  .finally(() => {
    console.log(failures ? `\n${failures} check(s) failed.\n` : "\nAll checks passed. Run `npm run dev`.\n");
    process.exitCode = failures ? 1 : 0;
  });
