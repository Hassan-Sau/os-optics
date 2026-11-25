"""Minimal deterministic PNG encoder (8-bit RGB or RGBA, no filtering, zlib): the same pixels always give the same bytes."""

from __future__ import annotations

import struct
import zlib

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
BIT_DEPTH = 8
COLOR_TYPE_RGB = 2
COLOR_TYPE_RGBA = 6
CHANNELS = {COLOR_TYPE_RGB: 3, COLOR_TYPE_RGBA: 4}
ZLIB_LEVEL = 9
CRC_MASK = 0xFFFFFFFF
CHUNK_OVERHEAD = 12  # length, type and CRC fields
COLOR_TYPE_OFFSET = 9  # of the colour type byte in the IHDR body


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & CRC_MASK)


def png_bytes(rgb: np.ndarray) -> bytes:
    """``rgb``: (height, width, 3) uint8, or (height, width, 4) with straight (not premultiplied) alpha."""

    height, width, channels = rgb.shape
    color_type = COLOR_TYPE_RGBA if channels == CHANNELS[COLOR_TYPE_RGBA] else COLOR_TYPE_RGB
    rows = np.concatenate([np.zeros((height, 1), dtype=np.uint8),
                           np.ascontiguousarray(rgb, dtype=np.uint8).reshape(height, width * channels)], axis=1)
    header = struct.pack(">IIBBBBB", width, height, BIT_DEPTH, color_type, 0, 0, 0)
    return PNG_SIGNATURE + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(rows.tobytes(), ZLIB_LEVEL)) + _chunk(b"IEND", b"")


def read_png(data: bytes) -> np.ndarray:
    """Decode a PNG written by ``png_bytes``: (height, width, 3) for RGB, (height, width, 4) for RGBA (tests and
    self-checks)."""

    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG")
    pos, width, height, channels, idat = len(PNG_SIGNATURE), 0, 0, 3, b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
            channels = CHANNELS[body[COLOR_TYPE_OFFSET]]
        elif kind == b"IDAT":
            idat += body
        pos += length + CHUNK_OVERHEAD
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(height, width * channels + 1)
    return raw[:, 1:].reshape(height, width, channels)


def read_png_rgb(data: bytes) -> np.ndarray:
    """Decode a PNG written by ``png_bytes`` (its colour channels; ``read_png`` keeps an alpha channel)."""

    return read_png(data)[:, :, :3]
