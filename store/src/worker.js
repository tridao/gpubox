// gpubox reputation store: storage and validation, no policy.
//
//   POST /v1/observations   {"observations": [...]}   at most MAX_BATCH
//   GET  /v1/observations?provider=vast[&after=SEQ][&limit=N]
//   GET  /v1/health
//
// Every client has its own bearer token, a Worker secret named TOKEN_<SOURCE>.
// The Worker decides who wrote a row (from the token) and when (the database's
// clock); whatever a client says about either is ignored. What clients do
// with the rows (the blocklist policy) lives in the Python package.
//
// Only the default export: the Workers runtime rejects an entry module that
// exports anything but handlers.

const MAX_BATCH = 50;
const MAX_BODY_BYTES = 64 * 1024;
const MAX_PAGE = 1000;
const WINDOW_DAYS = 30; // how far back a read goes
const RETENTION_DAYS = 90; // rows older than this are deleted
const MAX_ROWS_PER_SOURCE_PER_DAY = 500;
const FAILURE_SPACING_MS = 3600 * 1000; // one counted failure per source and machine

const DAY_MS = 86400 * 1000;
const KINDS = ["healthy", "failed_admission", "died_after_admission"];
const REASONS = [
  "busy_gpu", "unreachable", "no_network", "died",
  "runtime_error", "wrong_hardware", "cuda_init", "compute_mismatch",
];
const PROVIDERS = ["vast"];
// The database's clock in ms, evaluated when the statement runs.
const NOW_SQL = "CAST((julianday('now') - 2440587.5) * 86400000 AS INTEGER)";

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

const hex = (bytes) => [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");

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

/** What readers see instead of a source's name: stable, and meaningless without the salt. */
async function opaque(source, env) {
  const salt = env.SOURCE_SALT; // checked in fetch(); must never change, or old ids stop matching
  return hex(await digest(`gpubox-source:${salt}:${source}`)).slice(0, 16);
}

/** The request body as text, or null if it is longer than `limit` bytes. Stops reading there. */
async function boundedText(request, limit) {
  if (Number(request.headers.get("content-length") || 0) > limit) return null;
  if (!request.body) return "";
  const reader = request.body.getReader();
  const chunks = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > limit) {
      await reader.cancel();
      return null;
    }
    chunks.push(value);
  }
  const all = new Uint8Array(size);
  let at = 0;
  for (const c of chunks) {
    all.set(c, at);
    at += c.byteLength;
  }
  return new TextDecoder().decode(all);
}

async function post(request, env, source) {
  const text = await boundedText(request, MAX_BODY_BYTES);
  if (text === null) return json(413, { error: `body larger than ${MAX_BODY_BYTES} bytes` });
  let body;
  try {
    body = JSON.parse(text);
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

  // Rows we have already (a resend) cost nothing and take nobody's place.
  const known = await env.DB.prepare(
    "SELECT id FROM observations WHERE id IN (SELECT value FROM json_each(?1))",
  )
    .bind(JSON.stringify(clean.map((r) => r.id)))
    .all();
  const have = new Set(known.results.map((r) => r.id));
  const fresh = [];
  const failed = new Set();
  for (const row of clean) {
    if (have.has(row.id)) continue;
    have.add(row.id); // the same id twice in one request counts once
    if (row.kind !== "healthy") {
      // One counted failure per machine in a request; the statement below
      // applies the same rule against what is already stored.
      const key = `${row.provider}:${row.machine_id}`;
      if (failed.has(key)) continue;
      failed.add(key);
    }
    fresh.push(row);
  }
  if (fresh.length === 0) return json(200, { accepted: 0, ignored: clean.length });

  // One statement, so that the limits hold however many requests race: the
  // database runs statements one at a time, and this one checks what is stored
  // at the moment it inserts. (A statement that reads the table it inserts
  // into reads it as it was before the statement.) The quota is all or
  // nothing for the request; the spacing rule drops single rows.
  const recentFailure =
    "EXISTS (SELECT 1 FROM observations f WHERE f.source = ?2" +
    " AND f.provider = r.value ->> 'provider' AND f.machine_id = r.value ->> 'machine_id'" +
    ` AND f.kind != 'healthy' AND f.received_at > ${NOW_SQL} - ?3)`;
  const payload = JSON.stringify(fresh);
  const result = await env.DB.prepare(
    "INSERT OR IGNORE INTO observations (id, received_at, source, provider, machine_id," +
      " host_id, kind, reason_code, gpu, num_gpus, gate_version, policy_version, secs_to_ready)" +
      ` SELECT r.value ->> 'id', ${NOW_SQL}, ?2, r.value ->> 'provider', r.value ->> 'machine_id',` +
      " r.value ->> 'host_id', r.value ->> 'kind', r.value ->> 'reason_code', r.value ->> 'gpu'," +
      " r.value ->> 'num_gpus', r.value ->> 'gate_version', r.value ->> 'policy_version'," +
      " r.value ->> 'secs_to_ready'" +
      " FROM json_each(?1) AS r" +
      " WHERE (SELECT COUNT(*) FROM observations q WHERE q.source = ?2" +
      `        AND q.received_at > ${NOW_SQL} - ?4) + json_array_length(?1) <= ?5` +
      `   AND (r.value ->> 'kind' = 'healthy' OR NOT ${recentFailure})`,
  )
    .bind(payload, source, FAILURE_SPACING_MS, DAY_MS, MAX_ROWS_PER_SOURCE_PER_DAY)
    .run();
  const accepted = result.meta.changes;
  if (accepted < fresh.length) {
    // Why were some not stored? Rows we have by now (a racing resend) and
    // failures inside the spacing are final. Anything else was the quota, and
    // the client must keep those rows and try again later.
    const explained = await env.DB.prepare(
      "SELECT COUNT(*) AS n FROM json_each(?1) AS r WHERE" +
        " EXISTS (SELECT 1 FROM observations o WHERE o.id = r.value ->> 'id')" +
        ` OR (r.value ->> 'kind' != 'healthy' AND ${recentFailure})`,
    )
      .bind(payload, source, FAILURE_SPACING_MS)
      .first("n");
    if (explained < fresh.length) return json(429, { error: "daily limit for this source reached" });
  }
  return json(200, { accepted, ignored: clean.length - accepted });
}

async function get(url, env) {
  const provider = url.searchParams.get("provider");
  if (!PROVIDERS.includes(provider)) return json(400, { error: "provider" });
  const limit = Number(url.searchParams.get("limit") || MAX_PAGE);
  if (!isInt(limit, 1, MAX_PAGE)) return json(400, { error: "limit" });
  const afterText = url.searchParams.get("after") ?? "0";
  if (!/^\d{1,15}$/.test(afterText)) return json(400, { error: "after" });
  const after = Number(afterText);
  // What a reader needs to decide whom to skip: every failure, and a healthy
  // row only where the same source had failed that machine before (it then
  // takes back that source's own soft verdict). Healthy rows of machines
  // nobody complained about, the bulk of the table, are not sent.
  const [page, revoked, meta] = await env.DB.batch([
    env.DB.prepare(
      "SELECT o.seq, o.id, o.received_at, o.source, o.provider, o.machine_id, o.kind," +
        " o.reason_code, o.policy_version FROM observations o" +
        ` WHERE o.provider = ?1 AND o.seq > ?2 AND o.received_at > ${NOW_SQL} - ?3` +
        " AND o.source NOT IN (SELECT source FROM revoked_sources)" +
        " AND (o.kind != 'healthy' OR EXISTS (SELECT 1 FROM observations f" +
        "      WHERE f.source = o.source AND f.provider = o.provider" +
        "        AND f.machine_id = o.machine_id AND f.kind != 'healthy' AND f.seq < o.seq" +
        `        AND f.received_at > ${NOW_SQL} - ?3))` +
        " ORDER BY o.seq LIMIT ?4",
    ).bind(provider, after, WINDOW_DAYS * DAY_MS, limit + 1),
    env.DB.prepare("SELECT source FROM revoked_sources"),
    env.DB.prepare("SELECT value FROM meta WHERE key = 'store_id'"),
  ]);
  const rows = page.results.slice(0, limit);
  const names = new Map();
  for (const name of [...rows.map((r) => r.source), ...revoked.results.map((r) => r.source)]) {
    if (!names.has(name)) names.set(name, await opaque(name, env));
  }
  return json(200, {
    // Which database these sequence numbers belong to: a client that sees
    // another value must forget what it read and start from 0.
    store: meta.results[0].value,
    observations: rows.map((r) => ({
      seq: r.seq,
      id: r.id,
      ts: r.received_at / 1000, // when we received it: the only time clients may rely on
      src: names.get(r.source),
      provider: r.provider,
      machine_id: r.machine_id,
      kind: r.kind,
      reason_code: r.reason_code,
      policy_version: r.policy_version,
    })),
    // Resume after this sequence number; null when there is nothing more for now.
    next: page.results.length > limit ? rows[rows.length - 1].seq : null,
    // Rows of these sources are no longer served; drop the ones you kept.
    revoked: revoked.results.map((r) => names.get(r.source)),
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    try {
      if (url.pathname === "/v1/health") return json(200, { ok: true });
      if (url.pathname !== "/v1/observations") return json(404, { error: "not found" });
      if (typeof env.SOURCE_SALT !== "string" || env.SOURCE_SALT.length < 16) {
        return json(500, { error: "the store is not configured (SOURCE_SALT)" });
      }
      const source = await sourceOf(request, env);
      if (!source) return json(401, { error: "unauthorized" });
      const revoked = await env.DB.prepare("SELECT 1 AS hit FROM revoked_sources WHERE source = ?1")
        .bind(source)
        .first("hit");
      if (revoked) return json(403, { error: "this source is revoked" });
      if (request.method === "POST") return await post(request, env, source);
      if (request.method === "GET") return await get(url, env);
      return json(405, { error: "method" });
    } catch (err) {
      console.error(err && err.stack ? err.stack : String(err));
      return json(500, { error: "internal error" });
    }
  },

  async scheduled(event, env) {
    await env.DB.prepare(`DELETE FROM observations WHERE received_at < ${NOW_SQL} - ?1`)
      .bind(RETENTION_DAYS * DAY_MS)
      .run();
  },
};
