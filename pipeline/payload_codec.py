# pipeline/payload_codec.py
# Packs/Unpacks payload according to SoundSafe_Payload_Current.md (6-bit alphabet, length-prefixed fields, etc.)

from __future__ import annotations
from typing import Dict, List, Tuple

ALPHABET = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_.:/"
# Example 64-char alphabet; adjust to your exact 6-bit set from the Markdown

def text_to_6bit(s: str) -> List[int]:
    idx = {c:i for i,c in enumerate(ALPHABET)}
    return [idx[c] for c in s]

def sixbit_to_text(vals: List[int]) -> str:
    return "".join(ALPHABET[v % 64] for v in vals)

def pack_fields(fields: Dict[str, str]) -> bytes:
    """
    Example packer using length-prefixed fields in 6-bit alphabet.
    Returns raw bytes (8-bit) which you will feed into RS.
    """
    bits: List[int] = []
    for key, value in fields.items():
        vals = text_to_6bit(value)
        length = len(vals)
        # 1 byte length (0..255), then 6-bit symbols
        bits.extend([(length >> i) & 1 for i in range(8)])  # little-endian bits
        for v in vals:
            bits.extend([(v >> i) & 1 for i in range(6)])

    # pad to byte
    while len(bits) % 8 != 0:
        bits.append(0)
    by = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        for k in range(8):
            b |= (bits[i+k] << k)
        by.append(b)
    return bytes(by)

def unpack_fields(payload_bytes: bytes, field_order: List[str]) -> Dict[str, str]:
    bits = []
    for b in payload_bytes:
        for k in range(8):
            bits.append((b >> k) & 1)

    out: Dict[str, str] = {}
    cursor = 0
    for key in field_order:
        if cursor + 8 > len(bits): break
        # read length byte
        L = 0
        for k in range(8):
            L |= (bits[cursor+k] << k)
        cursor += 8
        vals: List[int] = []
        for _ in range(L):
            if cursor + 6 > len(bits): break
            v = 0
            for k in range(6):
                v |= (bits[cursor+k] << k)
            cursor += 6
            vals.append(v)
        out[key] = sixbit_to_text(vals)
    return out
