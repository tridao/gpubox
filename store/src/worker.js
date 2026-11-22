// gpubox reputation store: storage and validation, no policy.
//
//   POST /v1/observations   {"observations": [...]}   at most MAX_BATCH
//   GET  /v1/observations?provider=vast[&after=CURSOR][&limit=N]
//   GET  /v1/health
//
// Every client has its own bearer token, a Worker secret named TOKEN_<SOURCE>.
// The Worker decides `source` (from the token) and `received_at` (its own
// clock); whatever a client says about either is ignored. What clients do
// with the rows (the blocklist policy) lives in the Python package.
//
// Only the default export: the Workers runtime rejects an entry module that
// exports anything but handlers.

const MAX_BATCH = 50;
const MAX_PAGE = 1000;
const WINDOW_DAYS = 30; // how far back a read goes
const RETENTION_DAYS = 90; // rows older than this are deleted
const MAX_ROWS_PER_SOURCE_PER_DAY = 2000;
const FAILURE_SPACING_MS = 3600 * 1000; // one counted failure per source and machine

const DAY_MS = 86400 * 1000;
const KINDS = ["healthy", "failed_admission", "died_after_admission"];
const REASONS = [
  "busy_gpu", "unreachable", "no_network", "died",
  "runtime_error", "wrong_hardware", "cuda_init", "compute_mismatch",
];
const PROVIDERS = ["vast"];

const json = (status, body) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });

const isInt = (x, lo, hi) => Number.isInteger(x) && x >= lo && x <= hi;
const optional = (x, ok) => x === undefined || x === null || ok(x);

/** A row we are willing to store, or a string saying what is wrong with it. */
function validate(o) {
  if (typeof o !== "object" || o === null || Array.isArray(o)) return "not an object";
  if (typeof o.id !== "string" || !/^[0-9a-f]{16,64}$/.test(o.id)) return "id";
  if (!PROVIDERS.includes(o.provider)) return "provider";
  if (!isInt(o.machine_id, 1, 2 ** 40)) return "machine_id";
  if (!optional(o.host_id, (x) => isInt(x, 1, 2 ** 40))) return "host_id";
  if (!KINDS.includes(o.kind)) return "kind";
  const reason = o.reason_code ?? null;
  if (o.kind === "healthy" ? reason !== null : !REASONS.includes(reason)) return "reason_code";
  if (!optional(o.gpu, (x) => typeof x === "string" && /^[\w .+-]{1,64}$/.test(x))) return "gpu";
  if (!optional(o.num_gpus, (x) => isInt(x, 1, 64))) return "num_gpus";
  if (!optional(o.gate_version, (x) => isInt(x, 0, 1e6))) return "gate_version";
  if (!isInt(o.policy_version, 0, 1e6)) return "policy_version";
  if (!optional(o.secs_to_ready, (x) => typeof x === "number" && x >= 0 && x < 1e5)) {
    return "secs_to_ready";
  }
  return {
    id: o.id,
    provider: o.provider,
    machine_id: o.machine_id,
    host_id: o.host_id ?? null,
    kind: o.kind,
    reason_code: reason,
    gpu: o.gpu ?? null,
    num_gpus: o.num_gpus ?? null,
    gate_version: o.gate_version ?? null,
    policy_version: o.policy_version,
    secs_to_ready: o.secs_to_ready ?? null,
  };
}

async function digest(text) {
  return new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
}

/** The source a bearer token belongs to, or null. Compares digests in constant time. */
async function sourceOf(request, env) {
  const header = request.headers.get("authorization") || "";
  if (!header.startsWith("Bearer ") || header.length > 600) return null;
  const given = await digest(header.slice(7));
  let found = null;
  for (const [name, secret] of Object.entries(env)) {
    if (!name.startsWith("TOKEN_") || typeof secret !== "string" || secret.length < 32) continue;
    const want = await digest(secret);
    let diff = 0;
    for (let i = 0; i < want.length; i++) diff |= want[i] ^ given[i];
    if (diff === 0) found = name.slice(6).toLowerCase(); // no early exit
  }
  return found;
}

async function post(request, env, source, now) {
  let body;
  try {
    body = JSON.parse(await request.text());
  } catch {
    return json(400, { error: "body is not JSON" });
  }
  const rows = body && body.observations;
  if (!Array.isArray(rows) || rows.length === 0 || rows.length > MAX_BATCH) {
    return json(400, { error: `observations must be a list of 1 to ${MAX_BATCH}` });
  }
  const clean = [];
  for (const [i, raw] of rows.entries()) {
    const row = validate(raw);
    if (typeof row === "string") return json(400, { error: `observation ${i}: bad ${row}` });
    clean.push(row);
  }
  const today = await env.DB.prepare(
    "SELECT COUNT(*) AS n FROM observations WHERE source = ?1 AND received_at > ?2",
  )
    .bind(source, now - DAY_MS)
    .first("n");
  if (today + clean.length > MAX_ROWS_PER_SOURCE_PER_DAY) {
    return json(429, { error: "daily limit for this source reached" });
  }
  let accepted = 0;
  let ignored = 0;
  const failedInBatch = new Set();
  for (const row of clean) {
    if (row.kind !== "healthy") {
      // One source cannot pile strikes on a machine: one counted failure an hour.
      const key = `${row.provider}:${row.machine_id}`;
      const recent = await env.DB.prepare(
        "SELECT 1 AS hit FROM observations WHERE source = ?1 AND provider = ?2 AND machine_id = ?3" +
          " AND kind != 'healthy' AND received_at > ?4 AND id != ?5 LIMIT 1",
      )
        .bind(source, row.provider, row.machine_id, now - FAILURE_SPACING_MS, row.id)
        .first("hit");
      if (recent || failedInBatch.has(key)) {
        ignored++;
        continue;
      }
      failedInBatch.add(key);
    }
    const result = await env.DB.prepare(
      "INSERT OR IGNORE INTO observations (id, received_at, source, provider, machine_id," +
        " host_id, kind, reason_code, gpu, num_gpus, gate_version, policy_version, secs_to_ready)" +
        " VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
    )
      .bind(
        row.id, now, source, row.provider, row.machine_id, row.host_id, row.kind,
        row.reason_code, row.gpu, row.num_gpus, row.gate_version, row.policy_version,
        row.secs_to_ready,
      )
      .run();
    if (result.meta.changes) accepted++;
    else ignored++; // seen before: a resend
  }
  return json(200, { accepted, ignored });
}

async function get(url, env, now) {
  const provider = url.searchParams.get("provider");
  if (!PROVIDERS.includes(provider)) return json(400, { error: "provider" });
  const limit = Number(url.searchParams.get("limit") || MAX_PAGE);
  if (!isInt(limit, 1, MAX_PAGE)) return json(400, { error: "limit" });
  let afterTime = now - WINDOW_DAYS * DAY_MS;
  let afterId = "";
  const after = url.searchParams.get("after");
  if (after !== null) {
    const m = /^(\d{1,16}):([0-9a-f]{16,64})$/.exec(after);
    if (!m) return json(400, { error: "after" });
    afterTime = Math.max(afterTime, Number(m[1]));
    afterId = Number(m[1]) >= afterTime ? m[2] : "";
  }
  const { results } = await env.DB.prepare(
    "SELECT id, received_at, source, provider, machine_id, host_id, kind, reason_code, gpu," +
      " num_gpus, gate_version, policy_version, secs_to_ready FROM observations" +
      " WHERE provider = ?1 AND (received_at > ?2 OR (received_at = ?2 AND id > ?3))" +
      " AND source NOT IN (SELECT source FROM revoked_sources)" +
      " ORDER BY received_at, id LIMIT ?4",
  )
    .bind(provider, afterTime, afterId, limit + 1)
    .all();
  const page = results.slice(0, limit);
  const last = page[page.length - 1];
  return json(200, {
    // `ts` is when we received the row, in seconds: the only time clients may rely on.
    observations: page.map(({ received_at, ...row }) => ({ ...row, ts: received_at / 1000 })),
    next: results.length > limit ? `${last.received_at}:${last.id}` : null,
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    try {
      if (url.pathname === "/v1/health") return json(200, { ok: true });
      if (url.pathname !== "/v1/observations") return json(404, { error: "not found" });
      const source = await sourceOf(request, env);
      if (!source) return json(401, { error: "unauthorized" });
      const revoked = await env.DB.prepare("SELECT 1 AS hit FROM revoked_sources WHERE source = ?1")
        .bind(source)
        .first("hit");
      if (revoked) return json(403, { error: "this source is revoked" });
      const now = Date.now();
      if (request.method === "POST") return await post(request, env, source, now);
      if (request.method === "GET") return await get(url, env, now);
      return json(405, { error: "method" });
    } catch (err) {
      console.error(err && err.stack ? err.stack : String(err));
      return json(500, { error: "internal error" });
    }
  },

  async scheduled(event, env) {
    await env.DB.prepare("DELETE FROM observations WHERE received_at < ?1")
      .bind(Date.now() - RETENTION_DAYS * DAY_MS)
      .run();
  },
};
