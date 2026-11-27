"""A JSON-over-HTTP client for the providers' REST APIs. Standard library only.

One client, configured per provider with its base URL, its authentication headers and
the error class its calls raise (a provider.ProviderError with the HTTP status; 0 when
there was no answer; and `code`, the API's own name for the error where its body states
one). Answers are parsed, never trusted: callers check their shape.

`retries` is only for calls that can safely run twice: a create passes 0, unless the
API collapses repeats of it (an idempotency key in `headers`). A 429 or a 5xx, a lost
connection or an unreadable body is tried again, with the same body; any other status
is the answer.
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
    if isinstance(err, dict) and isinstance(err.get("error"), dict):
        # {"error": {"code": ..., "message": ...}}: its message, then its code.
        err = {**err, **err["error"], "error": err["error"].get("code")}
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


def error_code(text: str) -> str | None:
    """The API's own name for an error (`{"error": {"code": ...}}` or `{"code": ...}`), or
    None where the body states none."""
    try:
        err = json.loads(text)
    except ValueError:
        return None
    if not isinstance(err, dict):
        return None
    code = err["error"].get("code") if isinstance(err.get("error"), dict) else err.get("code")
    return code if isinstance(code, str) and code else None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect would carry the credential elsewhere, and its answer is not the API's."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


OPENER = urllib.request.build_opener(_NoRedirect)


class Client:
    def __init__(self, base: str, headers: dict, error, timeout: float = 30):
        self.base, self.headers, self.error, self.timeout = (
            base.rstrip("/"),
            headers,
            error,
            timeout,
        )

    def call(
        self, method, path, *, body=None, query=None, retries=2, timeout=None, headers=None,
        as_text=False,
    ):  # fmt: skip
        """The parsed JSON answer (None for an empty one), or `error(status, message)`.
        `headers` are added to this call's (an idempotency key). `as_text`: the answer as
        text, for an API that answers some calls with a bare value (Verda's ids)."""
        timeout = self.timeout if timeout is None else max(0.5, min(self.timeout, timeout))
        url = f"{self.base}/{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query, doseq=True)
        headers = {
            "Accept": "application/json", "User-Agent": "gpubox", **self.headers, **(headers or {})
        }  # fmt: skip
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with OPENER.open(req, timeout=timeout) as resp:
                    raw = resp.read()
                    if as_text:
                        return raw.decode()
                    return json.loads(raw) if raw.strip() else None
            except urllib.error.HTTPError as exc:
                try:
                    body_text = exc.read().decode(errors="replace")
                except Exception:
                    body_text = ""
                if (exc.code == 429 or exc.code >= 500) and attempt < retries:
                    time.sleep(2**attempt)
                    continue
                msg = f"{method} /{path}: HTTP {exc.code}: {describe_error(body_text)}"
                err = self.error(exc.code, msg)
                err.code = error_code(body_text)
                raise err from None
            except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
                if attempt < retries:
                    time.sleep(2**attempt)
                    continue
                raise self.error(0, f"{method} /{path}: {exc}") from None
        raise AssertionError("unreachable")
