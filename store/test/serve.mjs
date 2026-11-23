// Serve the Worker on a local port with a fresh local D1, for the Python client's test.
// Prints the URL, then runs until killed. Token: $TOKEN_LAPTOP.

import { readdirSync, readFileSync } from "node:fs";
import { Miniflare } from "miniflare";

const mf = new Miniflare({
  modules: true,
  scriptPath: "src/worker.js",
  compatibilityDate: "2025-09-01",
  host: "127.0.0.1",
  port: 0,
  d1Databases: { DB: "gpubox-reputation-serve" },
  bindings: { TOKEN_LAPTOP: process.env.TOKEN_LAPTOP, TOKEN_OTHER: process.env.TOKEN_OTHER, SOURCE_SALT: "serve" },
});
const db = await mf.getD1Database("DB");
for (const file of readdirSync("migrations").sort()) {
  const sql = readFileSync(`migrations/${file}`, "utf8").replace(/--.*$/gm, "");
  for (const stmt of sql.split(";").map((s) => s.trim()).filter(Boolean)) {
    await db.prepare(stmt).run();
  }
}
const url = await mf.ready;
console.log(url.origin);
