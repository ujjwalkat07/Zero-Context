"""Tests for high performance serialization and validation."""

from __future__ import annotations

import msgspec
import pytest

from zero_context import (
    APIResponse,
    SerializationError,
    StreamChunk,
    decode_json,
    decode_json_raw,
    encode_json,
)


class SampleItem(msgspec.Struct, frozen=True):
    id: str
    count: int
    active: bool = True


def test_encode_and_decode_struct() -> None:
    item = SampleItem(id="item-42", count=100)
    data = encode_json(item)
    assert isinstance(data, bytes)
    assert b"item-42" in data

    decoded = decode_json(data, SampleItem)
    assert decoded.id == "item-42"
    assert decoded.count == 100
    assert decoded.active is True


def test_decode_generic_api_response() -> None:
    raw = b'{"data":{"id":"chunk-1","event":"update","data":"hello"},"status":"ok"}'
    resp = decode_json(raw, APIResponse[StreamChunk])
    assert resp.status == "ok"
    assert resp.data.id == "chunk-1"
    assert resp.data.event == "update"
    assert resp.data.data == "hello"


def test_decode_invalid_json_raises_serialization_error() -> None:
    with pytest.raises(SerializationError):
        decode_json(b"{bad json", SampleItem)


def test_decode_schema_mismatch_raises_serialization_error() -> None:
    # count is expected to be int, passing string
    with pytest.raises(SerializationError):
        decode_json(b'{"id":"item-1","count":"not-a-number"}', SampleItem)


def test_decode_json_raw() -> None:
    raw = b'{"foo":"bar","nums":[1,2,3]}'
    decoded = decode_json_raw(raw)
    assert decoded == {"foo": "bar", "nums": [1, 2, 3]}
