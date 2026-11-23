// The Worker against a local D1 (Miniflare, pinned in package.json). No network.
//
//   npm test

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { after, before, beforeEach, test } from "node:test";
import { Miniflare } from "miniflare";

const LAPTOP = "laptop-token-0123456789abcdef0123456789abcdef";
const CI = "ci-token-0123456789abcdef0123456789abcdef0123";
const NOT_A_TOKEN = "some-other-secret-0123456789abcdef0123456789";
const DAY = 86400 * 1000;
const QUOTA = 500;
let mf, db;

export async function migrate(database) {
  for (const file of readdirSync("migrations").sort()) {
    const sql = readFileSync(`migrations/${file}`, "utf8").replace(/--.*$/gm, "");
    for (const stmt of sql.split(";").map((s) => s.trim()).filter(Boolean)) {
      await database.prepare(stmt).run();
    }
  }
}

before(async () => {
  mf = new Miniflare({
    modules: true,
    scriptPath: "src/worker.js",
    compatibilityDate: "2025-09-01", // the date in cloudflare.config.ts
    d1Databases: { DB: "gpubox-reputation-test" },
    bindings: {
      TOKEN_LAPTOP: LAPTOP, TOKEN_QUACK_CI: CI, TOKEN_SHORT: "short",
      UNRELATED_SECRET: NOT_A_TOKEN, SOURCE_SALT: "salt-for-tests-0123456789abcdef0123456789",
    },
  });
  db = await mf.getD1Database("DB");
  await migrate(db);
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
const stored = async (where = "1=1") =>
  (await db.prepare(`SELECT * FROM observations WHERE ${where} ORDER BY seq`).all()).results;
const seed = (row) =>
  db.prepare(
    "INSERT INTO observations (id, received_at, source, provider, machine_id, kind, reason_code, policy_version)" +
      " VALUES (?1, ?2, ?3, 'vast', ?4, ?5, ?6, 1)",
  ).bind(row.id ?? id(), row.at ?? Date.now(), row.source ?? "laptop", row.machine ?? 1,
    row.kind ?? "failed_admission", row.kind === "healthy" ? null : (row.reason ?? "busy_gpu")).run();

test("health needs no token and says nothing", async () => {
  assert.deepEqual(await call("GET", "/v1/health", { token: null }), { status: 200, json: { ok: true } });
});

test("anything else needs a known token", async () => {
  // Other secrets of the Worker, however long, are not tokens; nor is the salt.
  const others = [NOT_A_TOKEN, "salt-for-tests-0123456789abcdef0123456789"];
  for (const token of [null, "", "wrong", "short", ...others, LAPTOP + "x", LAPTOP.slice(1), "x".repeat(700)]) {
    assert.equal((await list("", { token })).status, 401, `token ${String(token).slice(0, 12)}`);
    assert.equal((await post([obs()], { token })).status, 401);
  }
  assert.equal((await stored()).length, 0);
  assert.equal((await list()).status, 200);
  assert.equal((await call("GET", "/v1/other")).status, 404);
  assert.equal((await call("DELETE", "/v1/observations")).status, 405);
});

test("a row is stored once, with the source and time the store decides", async () => {
  const row = failed(101, { source: "someone-else", src: "x", ts: 1, received_at: 1, seq: 999, run_id: "quack-ci.x.0123abcd" });
  const before = Date.now();
  assert.deepEqual((await post([row])).json, { accepted: 1, ignored: 0 });
  assert.deepEqual((await post([row])).json, { accepted: 0, ignored: 1 }); // a resend
  // The same id from another source changes nothing either: not the row, not who wrote it.
  assert.deepEqual((await post([{ ...row, machine_id: 5 }], { token: CI })).json, { accepted: 0, ignored: 1 });
  const rows = await stored();
  assert.equal(rows.length, 1);
  assert.equal(rows[0].source, "laptop");
  assert.equal(rows[0].machine_id, 101);
  assert.ok(rows[0].received_at >= before - 2000 && rows[0].received_at <= Date.now() + 2000);
  assert.equal(rows[0].gpu, "RTX 5090");
});

test("readers get reason codes and an opaque source, nothing else", async () => {
  await post([failed(101)]);
  await post([failed(101)], { token: CI });
  const { json } = await list();
  assert.equal(json.observations.length, 2);
  for (const row of json.observations) {
    assert.deepEqual(Object.keys(row).sort(), ["id", "kind", "machine_id", "policy_version", "provider", "reason_code", "seq", "src", "ts"]);
    assert.match(row.src, /^[0-9a-f]{16}$/);
    assert.ok(Math.abs(row.ts * 1000 - Date.now()) < 5000);
  }
  const [a, b] = json.observations;
  assert.notEqual(a.src, b.src); // two sources stay distinguishable...
  assert.ok(!JSON.stringify(json).includes("laptop") && !JSON.stringify(json).includes("quack")); // ...and nameless
  assert.equal(json.next, null);
  assert.deepEqual(json.revoked, []);
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
  assert.equal((await stored()).length, 0); // not even the good first row
  for (const body of [{}, { observations: [] }, { observations: "x" }, { observations: Array(51).fill(obs()) }]) {
    assert.equal((await call("POST", "/v1/observations", { body })).status, 400);
  }
  assert.equal((await call("POST", "/v1/observations", { raw: "{not json" })).status, 400);
});

test("a body that is too large is refused before it is parsed", async () => {
  const padded = JSON.stringify({ observations: [obs()], padding: "x".repeat(70 * 1024) });
  assert.equal((await call("POST", "/v1/observations", { raw: padded })).status, 413);
  assert.equal((await stored()).length, 0);
});

test("a full batch of new failures on different machines is stored in one go", async () => {
  const batch = Array.from({ length: 50 }, (_, i) => failed(1000 + i));
  assert.deepEqual((await post(batch)).json, { accepted: 50, ignored: 0 });
  assert.equal((await stored()).length, 50);
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

test("a resend beside a new failure does not use up the new one's place", async () => {
  const old = failed(5);
  await post([old]);
  await db.prepare("UPDATE observations SET received_at = received_at - ?1").bind(3601 * 1000).run();
  // `old` is known already; the new failure for the same machine is past the spacing and must count.
  assert.deepEqual((await post([old, failed(5)])).json, { accepted: 1, ignored: 1 });
});

test("requests that race cannot get two failures for one machine past the spacing", async () => {
  const answers = await Promise.all(Array.from({ length: 8 }, () => post([failed(77)])));
  assert.deepEqual(answers.map((a) => a.status), Array(8).fill(200));
  assert.equal((await stored("machine_id = 77")).length, 1); // what is in the database, not what was answered
  assert.equal(answers.reduce((n, a) => n + a.json.accepted, 0), 1);
});

test("a source has a daily limit, also when its requests race", async () => {
  const stmt = db.prepare(
    "INSERT INTO observations (id, received_at, source, provider, machine_id, kind, policy_version)" +
      " VALUES (?1, ?2, 'laptop', 'vast', ?3, 'healthy', 1)",
  );
  const now = Date.now();
  for (let i = 0; i < 4; i++) {
    await db.batch(Array.from({ length: 100 }, (_, j) => stmt.bind(id(), now - 1000, i * 100 + j + 1)));
  }
  await db.batch(Array.from({ length: QUOTA - 406 }, (_, j) => stmt.bind(id(), now - 1000, 5000 + j)));
  // Six places are left. Six requests of three rows each arrive together, and
  // each of them, looked at alone, would fit.
  const answers = await Promise.all(
    Array.from({ length: 6 }, (_, k) => post([obs({ machine_id: 9000 + k }), obs({ machine_id: 9100 + k }), obs({ machine_id: 9200 + k })])),
  );
  assert.equal((await stored("source = 'laptop'")).length, QUOTA);
  assert.ok(answers.every((a) => a.status === 200 || a.status === 429));
  assert.equal((await post([obs()])).status, 429);
  assert.equal((await post([obs()], { token: CI })).status, 200); // per source
  // Old rows stop counting after a day.
  await db.prepare("UPDATE observations SET received_at = received_at - ?1 WHERE source = 'laptop'").bind(DAY + 1000).run();
  assert.equal((await post([obs()])).status, 200);
});

test("a resend is free even at the limit", async () => {
  const row = obs();
  await post([row]);
  const stmt = db.prepare(
    "INSERT INTO observations (id, received_at, source, provider, machine_id, kind, policy_version)" +
      " VALUES (?1, ?2, 'laptop', 'vast', ?3, 'healthy', 1)",
  );
  for (let i = 0; i < 5; i++) {
    await db.batch(Array.from({ length: 100 }, (_, j) => stmt.bind(id(), Date.now(), i * 100 + j + 1)));
  }
  assert.deepEqual(await post([row]), { status: 200, json: { accepted: 0, ignored: 1 } });
});

test("reads go back thirty days and no further", async () => {
  const now = Date.now();
  await seed({ at: now - 31 * DAY, machine: 1 });
  await seed({ at: now - 29 * DAY, machine: 2 });
  await seed({ at: now - 1000, machine: 3 });
  const { json } = await list();
  assert.deepEqual(json.observations.map((o) => o.machine_id), [2, 3]);
  assert.deepEqual((await list("&after=0")).json.observations.map((o) => o.machine_id), [2, 3]);
});

test("a healthy row is served only where it takes back the same source's failure", async () => {
  await seed({ machine: 1, kind: "healthy" }); // nobody complained about machine 1
  await seed({ machine: 2 }); // laptop: machine 2 failed
  await seed({ machine: 2, kind: "healthy", source: "quack_ci" }); // someone else's health: not served
  await seed({ machine: 2, kind: "healthy" }); // laptop's own later health: served
  await seed({ machine: 3, kind: "healthy" });
  await seed({ machine: 3 }); // a failure after the health: the health is not served
  const { json } = await list();
  assert.deepEqual(json.observations.map((o) => [o.machine_id, o.kind]), [
    [2, "failed_admission"], [2, "healthy"], [3, "failed_admission"],
  ]);
  assert.equal(new Set(json.observations.slice(0, 2).map((o) => o.src)).size, 1);
});

test("paging by sequence returns every row once and can be resumed", async () => {
  for (let i = 0; i < 25; i++) await seed({ machine: i + 1 });
  const seen = [];
  let after = 0;
  for (let guard = 0; guard < 20; guard++) {
    const { status, json } = await list(`&limit=7&after=${after}`);
    assert.equal(status, 200);
    assert.ok(json.observations.length <= 7);
    seen.push(...json.observations.map((o) => o.machine_id));
    if (json.next === null) break;
    assert.equal(json.next, json.observations.at(-1).seq);
    after = json.next;
  }
  assert.deepEqual(seen, Array.from({ length: 25 }, (_, i) => i + 1));
  // A row committed after a reader finished is found by resuming where it stopped.
  const last = (await list()).json.observations.at(-1).seq;
  await seed({ machine: 99 });
  assert.deepEqual((await list(`&after=${last}`)).json.observations.map((o) => o.machine_id), [99]);
  for (const bad of ["&after=x", "&after=-1", "&after=1.5", "&after=12:ab", "&limit=0", "&limit=1001", "&limit=abc"]) {
    assert.equal((await list(bad)).status, 400, bad);
  }
  assert.equal((await call("GET", "/v1/observations?provider=aws")).status, 400);
});

test("a revoked source can neither write nor be read, and readers are told", async () => {
  await post([failed(1)]);
  await post([failed(2)], { token: CI });
  const laptopSrc = (await list()).json.observations.find((o) => o.machine_id === 1).src;
  await db.prepare("INSERT INTO revoked_sources (source, revoked_at) VALUES ('laptop', ?1)").bind(Date.now()).run();
  assert.equal((await post([obs()])).status, 403);
  assert.equal((await list()).status, 403);
  const { json } = await list("", { token: CI });
  assert.deepEqual(json.observations.map((o) => o.machine_id), [2]);
  assert.deepEqual(json.revoked, [laptopSrc]); // so that clients drop what they kept of it
});

test("the daily job deletes rows older than ninety days", async () => {
  const now = Date.now();
  await seed({ at: now - 91 * DAY, machine: 1 });
  await seed({ at: now - 89 * DAY, machine: 2 });
  const worker = await mf.getWorker();
  await worker.scheduled({ cron: "17 4 * * *" });
  assert.deepEqual((await stored()).map((r) => r.machine_id), [2]);
});

test("the queries use their indexes", async () => {
  const plan = async (sql, ...args) =>
    (await db.prepare("EXPLAIN QUERY PLAN " + sql).bind(...args).all()).results.map((r) => r.detail).join(" | ");
  assert.match(await plan("DELETE FROM observations WHERE received_at < ?1", 0), /observations_by_age/);
  assert.match(await plan("SELECT COUNT(*) FROM observations WHERE source = ?1 AND received_at > ?2", "x", 0), /observations_by_source/);
  assert.match(await plan("SELECT * FROM observations WHERE provider = ?1 AND seq > ?2 ORDER BY seq LIMIT 5", "vast", 0), /observations_by_provider/);
});
