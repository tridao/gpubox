# gpubox reputation store

A Cloudflare Worker in front of a D1 database. Clients post what they observed
about rented machines and read back what everyone observed; the blocklist
policy stays in the Python package. The store is advisory: nothing depends on
it for cleanup, and a launch never fails because of it.

```
POST /v1/observations   {"observations": [...]}   at most 50 per request
GET  /v1/observations?provider=vast[&after=CURSOR][&limit=N]   last 30 days, paged
GET  /v1/health
```

- Each client has its own bearer token, a Worker secret named `TOKEN_<SOURCE>`
  (at least 32 characters). The Worker sets `source` from the token and the
  time from its own clock; what a client says about either is ignored.
- A row is stored once (`INSERT OR IGNORE` on its id), so resending is safe.
- Per source: 2,000 rows a day, and one counted failure per machine per hour.
- Rows are deleted after 90 days by a daily scheduled run.
- To cut a source off: `INSERT INTO revoked_sources (source, revoked_at) VALUES
  ('laptop', <ms>)`. It can then neither write nor read, and its rows are no
  longer served.

## Develop

```bash
npm install          # wrangler (build), cf (deploy), miniflare (tests); all pinned
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

The secrets file is JSON, `{"TOKEN_LAPTOP": "...", "TOKEN_QUACK_CI": "..."}`,
and stays out of the repository.

The Worker's entry module must export nothing but the default handler object;
the runtime rejects any other export (the tests catch this, `cf deploy
--dry-run` does not).
