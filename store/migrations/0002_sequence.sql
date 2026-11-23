-- Rows get a sequence number in commit order, so that a reader can resume
-- exactly where it stopped. (Paging by receive time could miss a row that
-- was stamped before, and committed after, the reader passed.)
-- Nothing has been stored yet: the table is recreated.
DROP TABLE observations;

CREATE TABLE observations (
  seq            INTEGER PRIMARY KEY AUTOINCREMENT,
  id             TEXT NOT NULL UNIQUE,    -- chosen by the client; makes a resend a no-op
  received_at    INTEGER NOT NULL,        -- ms since epoch, the database's clock
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
-- reads: one provider, in sequence
CREATE INDEX observations_by_provider ON observations (provider, seq);
-- "has this source already failed this machine recently" and "did it fail before this healthy"
CREATE INDEX observations_by_machine ON observations (source, provider, machine_id, kind, received_at);
-- the daily quota of a source
CREATE INDEX observations_by_source ON observations (source, received_at);
-- retention
CREATE INDEX observations_by_age ON observations (received_at);
