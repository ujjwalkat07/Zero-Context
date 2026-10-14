"""Ultra-fast JSON and binary serialization layer powered by msgspec."""

from __future__ import annotations

from typing import Any, TypeVar

import msgspec

from zero_context._exceptions import SerializationError

T = TypeVar("T")

# Pre-allocated default encoder instance
_DEFAULT_ENCODER = msgspec.json.Encoder()


def encode_json(value: Any) -> bytes:
    """Encode an arbitrary object or msgspec.Struct into UTF-8 JSON bytes.

    Raises:
        SerializationError: If the value cannot be serialized into JSON.
    """
    try:
        return _DEFAULT_ENCODER.encode(value)
    except msgspec.EncodeError as err:
        raise SerializationError(f"Failed to serialize payload to JSON: {err}") from err


def decode_json(data: bytes, target_type: type[T]) -> T:
    """Zero-copy decode UTF-8 JSON bytes into a typed struct or model.

    Args:
        data: Raw JSON payload bytes.
        target_type: Expected msgspec Struct, primitive, or dataclass.

    Raises:
        SerializationError: If decoding fails or schema validation rejects the structure.
    """
    try:
        return msgspec.json.decode(data, type=target_type)
    except (msgspec.DecodeError, msgspec.ValidationError) as err:
        raise SerializationError(
            f"Failed to decode response as {target_type.__name__}: {err}"
        ) from err


def decode_json_raw(data: bytes) -> Any:
    """Decode raw JSON bytes into native Python dict/list types."""
    try:
        return msgspec.json.decode(data)
    except msgspec.DecodeError as err:
        raise SerializationError(f"Failed to decode raw JSON: {err}") from err
