"""Strict HTTPS byte-range reads for immutable public model objects."""

from __future__ import annotations

import math
import re
from collections.abc import Buffer
from http.client import HTTPResponse
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import OpenerDirector, Request, build_opener

from ams.checked import checked_range_end, checked_uint
from ams.errors import AmsError, ErrorCode

_CONTENT_RANGE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", re.ASCII)
_RETRIABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_MAX_URL_BYTES = 8192
_MAX_RETRIES = 5
_MAX_TIMEOUT_SECONDS = 300.0


def _validate_https_url(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > _MAX_URL_BYTES:
        raise AmsError(ErrorCode.PLAN_INVALID, f"{field} is empty or too long")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise AmsError(ErrorCode.PLAN_INVALID, f"{field} has an invalid port") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or port is not None
    ):
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            f"{field} must be a credential-free HTTPS URL on the default port",
        )
    return value


def _validate_transport_limits(timeout_seconds: float, max_retries: int) -> tuple[float, int]:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or timeout_seconds > _MAX_TIMEOUT_SECONDS
    ):
        raise AmsError(ErrorCode.PLAN_INVALID, "HTTP range timeout is outside the supported bound")
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or not 0 <= max_retries <= _MAX_RETRIES
    ):
        raise AmsError(
            ErrorCode.PLAN_INVALID, "HTTP range retry count is outside the supported bound"
        )
    return float(timeout_seconds), max_retries


def _transport_error(message: str, *, retriable: bool, status: int | None = None) -> AmsError:
    evidence = {"http_status": status} if status is not None else None
    return AmsError(
        ErrorCode.IO_FAILURE,
        message,
        retriable=retriable,
        subsystem="http-range",
        evidence=evidence,
    )


class HttpRangeReader:
    """A bounded public HTTPS reader that requires exact RFC 7233 responses.

    The reader sends no credentials and accepts only exact ``206`` responses with identity
    encoding, a matching ``Content-Length``, and a matching ``Content-Range``. Callers own the
    destination buffer; the reader never materializes the requested range internally.
    """

    def __init__(
        self,
        url: str,
        size_bytes: int,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        _opener: OpenerDirector | None = None,
    ) -> None:
        self.url = _validate_https_url(url, field="HTTP range URL")
        self.size_bytes = checked_uint(size_bytes, name="http_range.size_bytes")
        if self.size_bytes == 0:
            raise AmsError(ErrorCode.PLAN_INVALID, "HTTP range object must be nonempty")
        self.timeout_seconds, self.max_retries = _validate_transport_limits(
            timeout_seconds, max_retries
        )
        self._opener = _opener or build_opener()

    @classmethod
    def discover(
        cls,
        url: str,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        _opener: OpenerDirector | None = None,
    ) -> HttpRangeReader:
        """Discover object size with one byte and retain the same bounded reader policy."""
        validated_url = _validate_https_url(url, field="HTTP range URL")
        timeout, retries = _validate_transport_limits(timeout_seconds, max_retries)
        opener = _opener or build_opener()
        probe = bytearray(1)
        size_bytes = cls._read_request(
            opener,
            validated_url,
            0,
            memoryview(probe),
            expected_size=None,
            timeout_seconds=timeout,
            max_retries=retries,
        )
        return cls(
            validated_url,
            size_bytes,
            timeout_seconds=timeout,
            max_retries=retries,
            _opener=opener,
        )

    def read_into(self, offset: int, destination: Buffer) -> None:
        view = memoryview(destination).cast("B")
        try:
            if view.readonly:
                raise AmsError(ErrorCode.IO_FAILURE, "HTTP range destination is read-only")
            if view.nbytes == 0:
                checked_uint(offset, name="http_range.offset")
                if offset > self.size_bytes:
                    raise AmsError(ErrorCode.IO_FAILURE, "zero-length HTTP read begins past object")
                return
            end = checked_range_end(offset, view.nbytes, name="http_range.read")
            if end > self.size_bytes:
                raise AmsError(ErrorCode.IO_FAILURE, "HTTP range read exceeds the source object")
            observed_size = self._read_request(
                self._opener,
                self.url,
                offset,
                view,
                expected_size=self.size_bytes,
                timeout_seconds=self.timeout_seconds,
                max_retries=self.max_retries,
            )
            if observed_size != self.size_bytes:
                raise _transport_error("HTTP range object size changed", retriable=False)
        finally:
            view.release()

    @staticmethod
    def _read_request(
        opener: OpenerDirector,
        url: str,
        offset: int,
        destination: memoryview,
        *,
        expected_size: int | None,
        timeout_seconds: float,
        max_retries: int,
    ) -> int:
        end = checked_range_end(offset, destination.nbytes, name="http_range.request") - 1
        request = Request(
            url,
            headers={
                "Accept-Encoding": "identity",
                "Range": f"bytes={offset}-{end}",
                "User-Agent": "ams-runtime/0.1",
            },
            method="GET",
        )
        for attempt in range(max_retries + 1):
            try:
                with opener.open(request, timeout=timeout_seconds) as response:
                    return HttpRangeReader._consume_response(
                        response,
                        offset,
                        end,
                        destination,
                        expected_size=expected_size,
                    )
            except HTTPError as exc:
                retriable = exc.code in _RETRIABLE_STATUS
                exc.close()
                if retriable and attempt < max_retries:
                    continue
                raise _transport_error(
                    "HTTP range request returned an error status",
                    retriable=retriable,
                    status=exc.code,
                ) from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt < max_retries:
                    continue
                raise _transport_error(
                    "HTTP range request failed",
                    retriable=True,
                ) from exc
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "HTTP range retry loop fell through")

    @staticmethod
    def _consume_response(
        response: HTTPResponse,
        offset: int,
        end: int,
        destination: memoryview,
        *,
        expected_size: int | None,
    ) -> int:
        final_url = response.geturl()
        _validate_https_url(final_url, field="HTTP range redirect URL")
        status = getattr(response, "status", response.getcode())
        if status != 206:
            raise _transport_error(
                "HTTP range server did not honor the byte range",
                retriable=False,
                status=status,
            )
        content_encoding = response.headers.get("Content-Encoding")
        if content_encoding not in {None, "identity"}:
            raise _transport_error("HTTP range response was content-encoded", retriable=False)
        length_text = response.headers.get("Content-Length")
        if length_text != str(destination.nbytes):
            raise _transport_error("HTTP range response length is invalid", retriable=False)
        match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
        if match is None:
            raise _transport_error("HTTP range response metadata is invalid", retriable=False)
        observed_start, observed_end, observed_size = (int(value) for value in match.groups())
        if observed_start != offset or observed_end != end or observed_size <= observed_end:
            raise _transport_error(
                "HTTP range response does not match the request", retriable=False
            )
        if expected_size is not None and observed_size != expected_size:
            raise _transport_error("HTTP range object size changed", retriable=False)
        completed = 0
        while completed < destination.nbytes:
            count = response.readinto(destination[completed:])
            if count is None or count == 0:
                raise _transport_error("HTTP range response ended early", retriable=True)
            completed += count
        if response.read(1):
            raise _transport_error(
                "HTTP range response exceeded its declared length", retriable=False
            )
        return observed_size
