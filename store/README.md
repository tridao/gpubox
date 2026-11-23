# gpubox reputation store

A Cloudflare Worker in front of a D1 database. Clients post what they observed
about rented machines and read back what everyone observed; the blocklist
policy stays in the Python package. The store is advisory: nothing depends on
it for cleanup, and a launch never fails because of it.

```
POST /v1/observations   {"observations": [...]}   at most 50 rows and 64 KB
GET  /v1/observations?provider=vast[&after=SEQ][&limit=N]
GET  /v1/health
```

- Each client has its own bearer token, a Worker secret named `TOKEN_<SOURCE>`
  (at least 32 characters; names must differ in more than case). The Worker
  decides who wrote a row, from the token, and when, from the database's
  clock; what a client says about either is ignored.
- A row is stored once (its id is unique), so resending a stored row is safe
  and free.
- Per source: 500 new rows a day, and one counted failure per machine per
  hour. Both are part of the insert statement itself, so requests that race
  cannot get past them. A request that does not fit the daily limit is
  refused whole with 429 and can be sent again later; a failure inside the
  hour is dropped for good and reported as `ignored` (if that answer is lost
  and the row is sent again more than an hour later, it is stored then).
- Reads are by sequence number, in commit order: ask for what comes after the
  last number you have. Rows of the last 30 days are served. Every answer
  carries `store`, the identity of the database; a client that sees it change
  must forget what it read.
- Readers get reason codes only: `seq, id, ts, src, provider, machine_id,
  kind, reason_code, policy_version`. `src` is an opaque id of the source
  (a salted hash), not its name. A `healthy` row is served only where the same
  source had reported a failure of that machine before; the healthy rows of
  machines nobody complained about stay in the database.
- Rows are deleted after 90 days by a daily scheduled run.
- To cut a source off: `INSERT INTO revoked_sources (source, revoked_at) VALUES
  ('laptop', <ms>)`. It can then neither write nor read, its rows are no
  longer served, and every read lists the revoked sources' opaque ids so that
  clients drop what they had kept.

What one valid token can still do: mark up to 500 machines a day as failed
for everyone, for as long as the policy blocks on one source's word (two
hours to 14 days on a first report). It cannot unblock a machine for anyone
else, and revoking it undoes its rows. Requests are not rate-limited beyond
that; add a Cloudflare rate-limiting rule if that matters.

## Develop

```bash
npm ci               # wrangler (build), cf (deploy), miniflare (tests); all pinned
npm test             # the Worker against a local D1; no network
```

## Deploy

Uses the `cf` CLI pinned in `package.json` (`npx cf ...`), because building
needs a newer one than 0.15. Set `CLOUDFLARE_ACCOUNT_ID` first.

```bash
npx cf d1 create --name gpubox-reputation            # once; put the uuid in cloudflare.config.ts
npx cf d1 migrations apply <database-uuid>
npx cf deploy --secrets-file ~/.config/gpubox/store-secrets.json
```

The secrets file is JSON and stays out of the repository:

```json
{"TOKEN_LAPTOP": "...", "TOKEN_QUACK_CI": "...", "SOURCE_SALT": "..."}
```

`SOURCE_SALT` (16 characters or more) is required, and must never change:
the opaque source ids are derived from it, and clients remember revoked ids.

The Worker's entry module must export nothing but the default handler object;
the runtime rejects any other export (the tests catch this, `cf deploy
--dry-run` does not).
