-- Readers now get the whole 30-day window in one answer and sources by name,
-- so the store identity (for resuming reads) and the revocation list (a
-- source is cut off by deleting its token secret) are no longer used.
DROP TABLE meta;
DROP TABLE revoked_sources;
