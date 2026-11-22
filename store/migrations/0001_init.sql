-- Observations of rented machines, shared between gpubox clients.
-- The Worker is the only writer. It sets `received_at` and `source` itself.
CREATE TABLE observations (
  id             TEXT PRIMARY KEY,        -- chosen by the client; makes a resend a no-op
  received_at    INTEGER NOT NULL,        -- ms since epoch, the Worker's clock
  source         TEXT NOT NULL,           -- which token wrote it
  provider       TEXT NOT NULL,
  machine_id     INTEGER NOT NULL,
  host_id        INTEGER,
  kind           TEXT NOT NULL,
  reason_code    TEXT,
  gpu            TEXT,
  num_gpus       INTEGER,
  gate_version   INTEGER,
  policy_version INTEGER NOT NULL,
  secs_to_ready  REAL
);
CREATE INDEX observations_by_time ON observations (provider, received_at, id);
CREATE INDEX observations_by_machine ON observations (source, provider, machine_id, received_at);
CREATE INDEX observations_by_source ON observations (source, received_at);

-- A source whose token leaked: its rows are no longer served and it may not write.
CREATE TABLE revoked_sources (
  source     TEXT PRIMARY KEY,
  revoked_at INTEGER NOT NULL
);
