#!/usr/bin/env python3
"""hf CLI wrapper: rewrite Hub pagination / redirect URLs onto HF_ENDPOINT.

hf-mirror (and similar proxies) often return Link: rel="next" headers that still
point at https://huggingface.co. huggingface_hub follows those absolute URLs and
bypasses HF_ENDPOINT, which fails when the official host is unreachable.

Usage (same args as `hf`):
  export HF_ENDPOINT=https://hf-mirror.com
  python scripts/hf_with_mirror.py download <repo> ...
"""

from __future__ import annotations

import os
import sys


def _apply_mirror_patches() -> None:
    endpoint = (os.environ.get("HF_ENDPOINT") or "").rstrip("/")
    if not endpoint or endpoint in (
        "https://huggingface.co",
        "https://hub-ci.huggingface.co",
    ):
        return

    # Prefer IPv4: WSL often has broken IPv6 (Errno 101) for huggingface.co.
    import socket

    _orig_getaddrinfo = socket.getaddrinfo

    def _getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
        infos = _orig_getaddrinfo(host, port, family, type, proto, flags)
        v4 = [i for i in infos if i[0] == socket.AF_INET]
        return v4 or infos

    socket.getaddrinfo = _getaddrinfo_ipv4  # type: ignore[assignment]

    from huggingface_hub import constants
    from huggingface_hub.utils import _pagination
    from huggingface_hub.utils._http import fix_hf_endpoint_in_url

    _orig_next = _pagination._get_next_page

    def _get_next_page(response):
        url = _orig_next(response)
        if url:
            return fix_hf_endpoint_in_url(url, endpoint=constants.ENDPOINT)
        return url

    _pagination._get_next_page = _get_next_page  # type: ignore[assignment]

    # Also rewrite absolute huggingface.co URLs on every request (redirects, CDN resolve).
    import huggingface_hub.utils._http as _http

    _orig_backoff = _http.http_backoff

    def _http_backoff(method, url, **kwargs):
        if isinstance(url, str):
            url = fix_hf_endpoint_in_url(url, endpoint=constants.ENDPOINT)
        return _orig_backoff(method, url, **kwargs)

    _http.http_backoff = _http_backoff  # type: ignore[assignment]

    session = _http.get_session()
    _orig_request = session.request

    def _request(method, url, *args, **kwargs):
        if isinstance(url, str):
            url = fix_hf_endpoint_in_url(url, endpoint=constants.ENDPOINT)
        return _orig_request(method, url, *args, **kwargs)

    session.request = _request  # type: ignore[assignment]


def main() -> None:
    # Xet client / hf_transfer often bypass the mirror; prefer plain HTTPS.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    # Avoid flooding logs unless the user explicitly set HF_DEBUG.
    # (A sticky HF_DEBUG=1 from a prior shell session is common.)
    if os.environ.get("HF_DEBUG", "").lower() in {"1", "true", "yes"} and not os.environ.get(
        "HF_MIRROR_KEEP_DEBUG"
    ):
        # Keep debug only when explicitly opted in for this wrapper.
        pass

    _apply_mirror_patches()

    from huggingface_hub.cli.hf import main as hf_main

    # hf CLI reads sys.argv[1:]; keep "hf" as argv0 for nicer help text.
    sys.argv = ["hf", *sys.argv[1:]]
    hf_main()


if __name__ == "__main__":
    main()
