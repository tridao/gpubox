# gpubox reputation store

A Cloudflare Worker in front of a D1 database, at
`https://gpubox-reputation.<subdomain>.workers.dev`. Clients
post what they observed about rented machines and read back what everyone
observed; the blocklist policy stays in the Python package. The store is
advisory: nothing depends on it for cleanup, and a launch never fails because
of it.

```
POST /v1/observations   {"observations": [...]}   at most 50 rows and 64 KB
GET  /v1/observations?provider=vast              the last 30 days, in one answer
GET  /v1/health
```

- Each client has its own bearer token, a Worker secret named `TOKEN_<SOURCE>`
  (at least 32 characters; names must differ in more than case, since the
  source is the lowercased name). The Worker decides who wrote a row, from
  the token, and when, from the database's clock; what a client says about
  either is ignored. So a row sent late counts from when it arrived (the
  client does not send failures whose block is already over).
- A row is stored once (its id is unique), so resending a stored row is safe
  and free.
- Per source: 500 new rows a day, and one counted failure per machine per
  hour. Both are part of the insert statement itself, so requests that race
  cannot get past them. A request that does not fit the daily limit is
  refused whole with 429 and can be sent again later; a failure inside the
  hour is dropped for good and reported as `ignored` (if that answer is lost
  and the row is sent again more than an hour later, it is stored then).
- A read returns `{"observations": [...], "truncated": bool}`: rows of the
  last 30 days in commit order, as `seq, id, ts, src, provider, machine_id,
  kind, reason_code, policy_version`, where `src` is the source's name. A
  `healthy` row is served only where the same source had reported a failure
  of that machine before; the healthy rows of machines nobody complained
  about stay in the database. At most 5,000 rows, the newest.
- Rows are deleted after 90 days by a daily scheduled run.

Every token holder is trusted (DESIGN.md, A6). What one token can do: mark up
to 500 machines a day as failed for everyone, for as long as the policy
blocks on one source's word (two hours to 14 days). It cannot unblock a
machine for anyone else. To cut a source off (say `laptop`):

```bash
npx cf workers secrets delete TOKEN_LAPTOP --worker gpubox-reputation
# check: a GET with the old token now answers 401
npx cf d1 query <database-uuid> --sql "DELETE FROM observations WHERE source = 'laptop'"
```

and remove it from the secrets file. Removing it from the file alone does
nothing: `cf deploy` keeps secrets the file does not mention.

## Develop

```bash
npm ci               # wrangler (build), cf (deploy), miniflare (tests); all pinned
npm test             # the Worker against a local D1; no network
```

## Deploy

Uses the `cf` CLI pinned in `package.json` (`npx cf ...`), because building
needs a newer one than 0.15.

```bash
export CLOUDFLARE_ACCOUNT_ID=<your-account-id>
npx cf d1 migrations apply <database-uuid>
npx cf deploy --secrets-file ~/.config/gpubox/store-secrets.json
```

The database was created once with `npx cf d1 create --name gpubox-reputation`
(its uuid is in `cloudflare.config.ts`). The account's workers.dev subdomain,
`gpubox`, was registered once through the API (`PUT
/accounts/{id}/workers/subdomain`); `cf` has no command for it.

The secrets file is JSON, mode 600, and stays out of the repository:

```json
{"TOKEN_LAPTOP": "...", "TOKEN_QUACK_CI": "..."}
```

A client is configured with `GPUBOX_STORE_URL` and `GPUBOX_STORE_TOKEN`; for
CI, put them in the repository's secrets.

The Worker's entry module must export nothing but the default handler object;
the runtime rejects any other export (the tests catch this, `cf deploy
--dry-run` does not).
