-- An identity for this database, so that a client can tell when the sequence
-- numbers it remembers belong to another (recreated or different) store.
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT INTO meta (key, value) VALUES ('store_id', lower(hex(randomblob(8))));
