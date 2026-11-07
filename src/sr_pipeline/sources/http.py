"""HTTP access for the database connectors: disk cache, retries, rate limits.

Three jobs, all in one place so every connector behaves the same way:

1. **Cache.** Every successful GET is stored under ``.cache/http/`` keyed on
   sha256(URL + sorted parameters). Re-running a search returns exactly the same
   records, for free, without touching the network: the search is reproducible.
   The flip side: a cached search never sees papers published since. Use
   ``refresh=True`` to re-query the databases (and note the new search date).
   Secret parameters (API keys) are left out of the key, so the cache is shared
   whether or not a key is set, and keys never end up in a cache file.
2. **Retries.** 429 (too many requests), 5xx and network errors are retried
   with exponential backoff, honouring the server's ``Retry-After`` header.
   Other errors (400, 404) are raised at once: retrying will not fix a bad query.
3. **Rate limit.** A minimum interval between two requests *to the same host*
   (PubMed: 3/s without key, arXiv: one every 3 s). Cache hits do not count.

``get_bytes`` is the binary twin of ``get_text`` (open-access PDFs): body cached
as ``<key>.bin`` with its content type in ``<key>.meta.json``.

``transport``, ``sleep`` and ``clock`` are injectable so tests run with recorded
responses (``httpx.MockTransport``), no network and no real waiting.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import httpx

SECRET_PARAMS = {"api_key", "apikey", "key"}
RETRY_STATUS = {429, 500, 502, 503, 504}
USER_AGENT = "sr-pipeline/0.1 (systematic review research tool)"


class HttpError(RuntimeError):
    def __init__(self, url: str, status: int | None, detail: str):
        super().__init__(f"{url}: {status or 'network error'}: {detail}")
        self.url, self.status = url, status


def cache_key(url: str, params: dict | None) -> str:
    public = {k: str(v) for k, v in (params or {}).items()
              if k not in SECRET_PARAMS and v is not None}
    payload = json.dumps({"url": url, "params": public}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class HttpClient:
    def __init__(self, cache_dir: str | Path | None = None, *, max_retries: int = 4,
                 backoff: float = 1.0, timeout: float = 30.0,
                 min_interval: dict[str, float] | None = None, refresh: bool = False,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries
        self.backoff = backoff
        self.min_interval = dict(min_interval or {})  # host -> seconds
        self.refresh = refresh
        self.sleep, self.clock = sleep, clock
        self._last_request: dict[str, float] = {}
        self.network_calls = 0  # real requests sent (cache hits excluded)
        self._client = httpx.Client(transport=transport, timeout=timeout, follow_redirects=True,
                                    headers={"User-Agent": USER_AGENT})

    def set_min_interval(self, host: str, seconds: float) -> None:
        self.min_interval[host] = max(seconds, self.min_interval.get(host, 0.0))

    # --- public API ----------------------------------------------------------
    def get_text(self, url: str, params: dict | None = None) -> str:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        path = self.cache_dir / f"{cache_key(url, params)}.json" if self.cache_dir else None
        if path is not None and path.exists() and not self.refresh:
            return json.loads(path.read_text())["body"]

        body = self._fetch(url, params).text
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            public = {k: v for k, v in params.items() if k not in SECRET_PARAMS}
            tmp = path.with_suffix(".tmp")  # atomic write: never a half-written cache file
            tmp.write_text(json.dumps({"url": url, "params": public, "body": body}))
            tmp.replace(path)
        return body

    def get_json(self, url: str, params: dict | None = None) -> dict:
        return json.loads(self.get_text(url, params))

    def get_bytes(self, url: str, params: dict | None = None) -> tuple[bytes, str]:
        """Binary GET (e.g. a PDF): ``(body, content_type)``, cached like ``get_text``."""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        key = cache_key(url, params)
        body_path = self.cache_dir / f"{key}.bin" if self.cache_dir else None
        meta_path = self.cache_dir / f"{key}.meta.json" if self.cache_dir else None
        if body_path is not None and body_path.exists() and meta_path.exists() and not self.refresh:
            return body_path.read_bytes(), json.loads(meta_path.read_text())["content_type"]

        resp = self._fetch(url, params)
        body, ctype = resp.content, resp.headers.get("content-type", "")
        if body_path is not None:
            body_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = body_path.with_suffix(".tmp")
            tmp.write_bytes(body)
            tmp.replace(body_path)
            meta_path.write_text(json.dumps({"url": url, "content_type": ctype,
                                             "final_url": str(resp.url)}))
        return body, ctype

    # --- internals -------------------------------------------------------------
    def _wait_turn(self, host: str) -> None:
        interval = self.min_interval.get(host, 0.0)
        last = self._last_request.get(host)
        if interval and last is not None:
            wait = interval - (self.clock() - last)
            if wait > 0:
                self.sleep(wait)
        self._last_request[host] = self.clock()

    def _fetch(self, url: str, params: dict) -> httpx.Response:
        host = urlsplit(url).netloc
        last_error = ""
        status = None
        for attempt in range(self.max_retries + 1):
            self._wait_turn(host)
            self.network_calls += 1
            try:
                resp = self._client.get(url, params=params)
            except httpx.TransportError as exc:
                status, last_error, retry_after = None, f"{type(exc).__name__}: {exc}", None
            else:
                if resp.status_code == 200:
                    return resp
                status, last_error = resp.status_code, resp.text[:200]
                if status not in RETRY_STATUS:
                    raise HttpError(url, status, last_error)
                retry_after = _retry_after(resp)
            if attempt < self.max_retries:
                self.sleep(retry_after if retry_after is not None
                           else self.backoff * 2 ** attempt)
        raise HttpError(url, status, f"gave up after {self.max_retries + 1} attempts: {last_error}")


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    try:
        return min(float(value), 60.0) if value is not None else None
    except ValueError:
        return None
