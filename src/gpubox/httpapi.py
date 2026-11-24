"""A JSON-over-HTTP client for the providers' REST APIs. Standard library only.

One client, configured per provider with its base URL, its authentication headers and
the error class its calls raise (a provider.ProviderError with the HTTP status; 0 when
there was no answer). Answers are parsed, never trusted: callers check their shape.

`retries` is only for calls that can safely run twice: a create passes 0. A 429 or a
5xx, a lost connection or an unreadable body is tried again; any other status is the
answer.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request


def describe_error(text: str) -> str:
    """The useful part of an error body: the fields APIs put their reason in, else the
    start of the body."""
    try:
        err = json.loads(text)
    except ValueError:
        err = None
    if isinstance(err, dict):
        parts = [
            str(err[k]) for k in ("title", "detail", "message", "error", "error_reason")
            if isinstance(err.get(k), (str, int)) and str(err[k])
        ]  # fmt: skip
        extra = err.get("errors")
        if isinstance(extra, list):
            parts += [str(e) for e in extra[:3]]
        if parts:
            return " ".join(": ".join(parts).split())[:300]
    return " ".join(text.split())[:300] or "no details"


class Client:
    def __init__(self, base: str, headers: dict, error, timeout: float = 30):
        self.base, self.headers, self.error, self.timeout = (
            base.rstrip("/"),
            headers,
            error,
            timeout,
        )

    def call(self, method, path, *, body=None, query=None, retries=2, timeout=None):
        """The parsed JSON answer (None for an empty one), or `error(status, message)`."""
        timeout = self.timeout if timeout is None else max(0.5, min(self.timeout, timeout))
        url = f"{self.base}/{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query, doseq=True)
        headers = {"Accept": "application/json", "User-Agent": "gpubox", **self.headers}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read()
                    return json.loads(raw) if raw.strip() else None
            except urllib.error.HTTPError as exc:
                try:
                    text = exc.read().decode(errors="replace")
                except Exception:
                    text = ""
                if (exc.code == 429 or exc.code >= 500) and attempt < retries:
                    time.sleep(2**attempt)
                    continue
                msg = f"{method} /{path}: HTTP {exc.code}: {describe_error(text)}"
                raise self.error(exc.code, msg) from None
            except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
                if attempt < retries:
                    time.sleep(2**attempt)
                    continue
                raise self.error(0, f"{method} /{path}: {exc}") from None
        raise AssertionError("unreachable")
