// The Worker against a local D1 (Miniflare, pinned in package.json). No network.
//
//   npm test

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { after, before, beforeEach, test } from "node:test";
import { Miniflare } from "miniflare";

const LAPTOP = "laptop-token-0123456789abcdef0123456789abcdef";
const CI = "ci-token-0123456789abcdef0123456789abcdef0123";
const DAY = 86400 * 1000;
let mf, db;

before(async () => {
  mf = new Miniflare({
    modules: true,
    scriptPath: "src/worker.js",
    compatibilityDate: "2025-09-01",
    d1Databases: { DB: "gpubox-reputation-test" },
    bindings: { TOKEN_LAPTOP: LAPTOP, TOKEN_QUACK_CI: CI, TOKEN_SHORT: "short", OTHER: LAPTOP },
  });
  db = await mf.getD1Database("DB");
  const sql = readFileSync("migrations/0001_init.sql", "utf8").replace(/--.*$/gm, "");
  for (const stmt of sql.split(";").map((s) => s.trim()).filter(Boolean)) {
    await db.prepare(stmt).run();
  }
});
after(() => mf.dispose());
beforeEach(async () => {
  await db.prepare("DELETE FROM observations").run();
  await db.prepare("DELETE FROM revoked_sources").run();
});

let counter = 0;
const id = () => (++counter).toString(16).padStart(32, "0");
const obs = (extra = {}) => ({
  id: id(), provider: "vast", machine_id: 101, host_id: 7, kind: "healthy",
  gpu: "RTX 5090", num_gpus: 2, gate_version: 1, policy_version: 1, secs_to_ready: 58.4, ...extra,
});
const failed = (machine_id, extra = {}) =>
  obs({ machine_id, kind: "failed_admission", reason_code: "busy_gpu", secs_to_ready: null, ...extra });

async function call(method, path, { token = LAPTOP, body, raw } = {}) {
  const headers = token ? { authorization: `Bearer ${token}` } : {};
  const init = { method, headers };
  if (raw !== undefined) init.body = raw;
  else if (body !== undefined) init.body = JSON.stringify(body);
  const res = await mf.dispatchFetch(`http://store${path}`, init);
  return { status: res.status, json: await res.json() };
}
const post = (observations, opts) => call("POST", "/v1/observations", { body: { observations }, ...opts });
const list = (query = "", opts) => call("GET", `/v1/observations?provider=vast${query}`, opts);
const seed = (row) =>
  db.prepare(
    "INSERT INTO observations (id, received_at, source, provider, machine_id, kind, reason_code, policy_version)" +
      " VALUES (?1, ?2, ?3, 'vast', ?4, ?5, ?6, 1)",
  ).bind(row.id ?? id(), row.at, row.source ?? "laptop", row.machine ?? 1, row.kind ?? "healthy", row.reason ?? null).run();

test("health needs no token and says nothing", async () => {
  assert.deepEqual(await call("GET", "/v1/health", { token: null }), { status: 200, json: { ok: true } });
});

test("anything else needs a known token", async () => {
  for (const token of [null, "", "wrong", "short", LAPTOP + "x", LAPTOP.slice(1), "x".repeat(700)]) {
    assert.equal((await list("", { token })).status, 401, `token ${String(token).slice(0, 12)}`);
    assert.equal((await post([obs()], { token })).status, 401);
  }
  // A binding that is not named TOKEN_* is not a token, and a short secret is not accepted.
  assert.equal((await list("", { token: "short" })).status, 401);
  assert.equal((await list()).status, 200);
  assert.equal((await call("GET", "/v1/other")).status, 404);
  assert.equal((await call("DELETE", "/v1/observations")).status, 405);
});

test("a row is stored once, with the source and time the Worker decides", async () => {
  const row = obs({ source: "someone-else", ts: 1, received_at: 1, run_id: "quack-ci.x.0123abcd" });
  const before = Date.now();
  assert.deepEqual((await post([row])).json, { accepted: 1, ignored: 0 });
  assert.deepEqual((await post([row])).json, { accepted: 0, ignored: 1 }); // a resend
  assert.deepEqual((await post([row], { token: CI })).json, { accepted: 0, ignored: 1 });
  const { json } = await list();
  assert.equal(json.observations.length, 1);
  const got = json.observations[0];
  assert.equal(got.source, "laptop");
  assert.ok(got.ts * 1000 >= before - 5 && got.ts * 1000 <= Date.now() + 5);
  assert.equal(got.machine_id, 101);
  assert.equal(got.secs_to_ready, 58.4);
  assert.ok(!("run_id" in got) && !("received_at" in got)); // nothing that was not asked for
  assert.equal(json.next, null);
});

test("rows that do not fit the schema reject the whole batch", async () => {
  const bad = [
    {}, null, [], "text", obs({ id: "short" }), obs({ id: "Z".repeat(32) }), obs({ id: 7 }),
    obs({ provider: "runpod" }), obs({ machine_id: 0 }), obs({ machine_id: "101" }), obs({ machine_id: 1.5 }),
    obs({ host_id: "7" }), obs({ kind: "exploded" }), obs({ kind: "healthy", reason_code: "died" }),
    obs({ kind: "failed_admission" }), obs({ kind: "failed_admission", reason_code: "Error: token=abc at https://x" }),
    obs({ gpu: "x".repeat(65) }), obs({ gpu: "<script>" }), obs({ num_gpus: 0 }), obs({ gate_version: -1 }),
    obs({ policy_version: null }), obs({ secs_to_ready: -1 }), obs({ secs_to_ready: "58" }),
  ];
  for (const row of bad) {
    const res = await post([obs(), row]);
    assert.equal(res.status, 400, JSON.stringify(row));
  }
  assert.equal((await list()).json.observations.length, 0); // not even the good first row
  for (const body of [{}, { observations: [] }, { observations: "x" }, { observations: Array(51).fill(obs()) }]) {
    assert.equal((await call("POST", "/v1/observations", { body })).status, 400);
  }
  assert.equal((await call("POST", "/v1/observations", { raw: "{not json" })).status, 400);
});

test("one source cannot pile failures on a machine", async () => {
  assert.deepEqual((await post([failed(5), failed(5), failed(6)])).json, { accepted: 2, ignored: 1 });
  assert.deepEqual((await post([failed(5, { reason_code: "cuda_init" })])).json, { accepted: 0, ignored: 1 });
  assert.deepEqual((await post([obs({ machine_id: 5 })])).json, { accepted: 1, ignored: 0 }); // healthy always counts
  assert.deepEqual((await post([failed(5)], { token: CI })).json, { accepted: 1, ignored: 0 }); // another source
  // After the spacing has passed, the same source may report it again.
  await db.prepare("UPDATE observations SET received_at = received_at - ?1").bind(3601 * 1000).run();
  assert.deepEqual((await post([failed(5)])).json, { accepted: 1, ignored: 0 });
});

test("a source has a daily limit", async () => {
  const now = Date.now();
  const stmt = db.prepare(
    "INSERT INTO observations (id, received_at, source, provider, machine_id, kind, policy_version)" +
      " VALUES (?1, ?2, 'laptop', 'vast', ?3, 'healthy', 1)",
  );
  for (let i = 0; i < 20; i++) {
    await db.batch(Array.from({ length: 100 }, (_, j) => stmt.bind(id(), now - 1000, i * 100 + j + 1)));
  }
  assert.equal((await post([obs()])).status, 429);
  assert.equal((await post([obs()], { token: CI })).status, 200); // per source
});

test("reads go back thirty days and no further", async () => {
  const now = Date.now();
  await seed({ at: now - 31 * DAY, machine: 1 });
  await seed({ at: now - 29 * DAY, machine: 2 });
  await seed({ at: now - 1000, machine: 3 });
  const { json } = await list();
  assert.deepEqual(json.observations.map((o) => o.machine_id), [2, 3]);
  // A cursor from before the window cannot reach behind it either.
  const old = await list(`&after=${now - 40 * DAY}:${"0".repeat(32)}`);
  assert.deepEqual(old.json.observations.map((o) => o.machine_id), [2, 3]);
});

test("paging returns every row once, even when times collide", async () => {
  const now = Date.now();
  for (let i = 0; i < 25; i++) await seed({ at: now - 5000 + Math.floor(i / 5), machine: i + 1 });
  const seen = [];
  let cursor = "";
  for (let guard = 0; guard < 20; guard++) {
    const { status, json } = await list(`&limit=7${cursor}`);
    assert.equal(status, 200);
    assert.ok(json.observations.length <= 7);
    seen.push(...json.observations.map((o) => o.machine_id));
    if (!json.next) break;
    cursor = `&after=${json.next}`;
  }
  assert.deepEqual(seen, Array.from({ length: 25 }, (_, i) => i + 1));
  for (const bad of ["&after=x", "&after=12:zz", "&limit=0", "&limit=1001", "&limit=abc"]) {
    assert.equal((await list(bad)).status, 400, bad);
  }
  assert.equal((await call("GET", "/v1/observations?provider=aws")).status, 400);
});

test("a revoked source can neither write nor be read", async () => {
  await post([obs({ machine_id: 1 })]);
  await post([obs({ machine_id: 2 })], { token: CI });
  await db.prepare("INSERT INTO revoked_sources (source, revoked_at) VALUES ('laptop', ?1)").bind(Date.now()).run();
  assert.equal((await post([obs()])).status, 403);
  assert.equal((await list()).status, 403);
  const { json } = await list("", { token: CI });
  assert.deepEqual(json.observations.map((o) => [o.machine_id, o.source]), [[2, "quack_ci"]]);
});

test("the daily job deletes rows older than ninety days", async () => {
  const now = Date.now();
  await seed({ at: now - 91 * DAY, machine: 1 });
  await seed({ at: now - 89 * DAY, machine: 2 });
  const worker = await mf.getWorker();
  await worker.scheduled({ cron: "17 4 * * *" });
  const { results } = await db.prepare("SELECT machine_id FROM observations").all();
  assert.deepEqual(results.map((r) => r.machine_id), [2]);
});
