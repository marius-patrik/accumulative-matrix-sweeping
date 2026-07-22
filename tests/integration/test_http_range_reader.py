from __future__ import annotations

from urllib.error import HTTPError, URLError

import pytest

from ams.errors import AmsError, ErrorCode
from ams.storage import HttpRangeReader


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        content_range: str,
        content_length: int | None = None,
        status: int = 206,
        final_url: str = "https://cdn.example.test/object",
    ) -> None:
        self.payload = payload
        self.position = 0
        self.status = status
        self.final_url = final_url
        self.headers = {
            "Content-Length": str(len(payload) if content_length is None else content_length),
            "Content-Range": content_range,
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def geturl(self) -> str:
        return self.final_url

    def getcode(self) -> int:
        return self.status

    def readinto(self, destination) -> int:
        view = memoryview(destination).cast("B")
        available = min(view.nbytes, len(self.payload) - self.position)
        if available <= 0:
            return 0
        view[:available] = self.payload[self.position : self.position + available]
        self.position += available
        return available

    def read(self, length: int) -> bytes:
        result = self.payload[self.position : self.position + length]
        self.position += len(result)
        return result


class FakeOpener:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.requests = []

    def open(self, request, *, timeout: float):
        self.requests.append((request, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_https_reader_discovers_size_and_reads_the_exact_requested_range() -> None:
    opener = FakeOpener(
        [
            FakeResponse(b"a", content_range="bytes 0-0/6"),
            FakeResponse(b"cde", content_range="bytes 2-4/6"),
        ]
    )
    reader = HttpRangeReader.discover(
        "https://models.example.test/object",
        max_retries=0,
        _opener=opener,
    )
    destination = bytearray(3)
    reader.read_into(2, destination)

    assert reader.size_bytes == 6
    assert destination == b"cde"
    assert [request.get_header("Range") for request, _ in opener.requests] == [
        "bytes=0-0",
        "bytes=2-4",
    ]
    assert all(
        request.get_header("Accept-encoding") == "identity" for request, _ in opener.requests
    )


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (FakeResponse(b"abc", content_range="bytes 2-4/6", status=200), "honor"),
        (FakeResponse(b"abc", content_range="bytes 1-3/6"), "match"),
        (
            FakeResponse(
                b"abc",
                content_range="bytes 2-4/6",
                final_url="http://cdn.example.test/object",
            ),
            "credential-free HTTPS",
        ),
    ],
)
def test_https_reader_rejects_status_range_and_redirect_drift_before_payload_use(
    response: FakeResponse,
    message: str,
) -> None:
    opener = FakeOpener([response])
    reader = HttpRangeReader(
        "https://models.example.test/object",
        6,
        max_retries=0,
        _opener=opener,
    )
    destination = bytearray(b"XYZ")
    with pytest.raises(AmsError, match=message):
        reader.read_into(2, destination)
    assert destination == b"XYZ"


def test_https_reader_retries_only_bounded_transport_failures() -> None:
    transient = URLError("temporary")
    opener = FakeOpener(
        [
            transient,
            FakeResponse(b"a", content_range="bytes 0-1/2", content_length=2),
            FakeResponse(b"ab", content_range="bytes 0-1/2"),
        ]
    )
    reader = HttpRangeReader(
        "https://models.example.test/object",
        2,
        max_retries=2,
        _opener=opener,
    )
    destination = bytearray(2)
    reader.read_into(0, destination)
    assert destination == b"ab"
    assert len(opener.requests) == 3

    permanent = HTTPError(
        "https://models.example.test/object",
        404,
        "not found",
        {},
        None,
    )
    failing = FakeOpener([permanent])
    reader = HttpRangeReader(
        "https://models.example.test/object",
        2,
        max_retries=5,
        _opener=failing,
    )
    with pytest.raises(AmsError) as caught:
        reader.read_into(0, bytearray(2))
    assert caught.value.code is ErrorCode.IO_FAILURE
    assert caught.value.retriable is False
    assert caught.value.evidence == {"http_status": 404}
    assert len(failing.requests) == 1
