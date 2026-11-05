#!/usr/bin/env python3
"""Dump a GGUF file without gguf-py or numpy.

Prints the metadata, the tensor directory and per-tensor value stats. This is
the reader side of make_demo_gguf.py, useful to see how the file is laid out.
"""

import argparse
import array
import struct
import sys
from pathlib import Path

GGML_TYPE_F32 = 0

GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

TYPE_NAMES = {
    GGUF_TYPE_UINT8: "u8", GGUF_TYPE_INT8: "i8",
    GGUF_TYPE_UINT16: "u16", GGUF_TYPE_INT16: "i16",
    GGUF_TYPE_UINT32: "u32", GGUF_TYPE_INT32: "i32",
    GGUF_TYPE_FLOAT32: "f32", GGUF_TYPE_BOOL: "bool",
    GGUF_TYPE_STRING: "str", GGUF_TYPE_ARRAY: "array",
    GGUF_TYPE_UINT64: "u64", GGUF_TYPE_INT64: "i64",
    GGUF_TYPE_FLOAT64: "f64",
}

# ggml tensor types, not the gguf metadata types above
GGML_TYPE_NAMES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
    8: "q8_0", 9: "q8_1", 10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k",
    14: "q6_k", 15: "q8_k", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs",
    19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs",
    24: "i8", 25: "i16", 26: "i32", 27: "i64", 28: "f64", 29: "iq1_m",
    30: "bf16",
}

SCALAR_FMT = {
    GGUF_TYPE_UINT8: "<B", GGUF_TYPE_INT8: "<b",
    GGUF_TYPE_UINT16: "<H", GGUF_TYPE_INT16: "<h",
    GGUF_TYPE_UINT32: "<I", GGUF_TYPE_INT32: "<i",
    GGUF_TYPE_FLOAT32: "<f", GGUF_TYPE_BOOL: "<B",
    GGUF_TYPE_UINT64: "<Q", GGUF_TYPE_INT64: "<q",
    GGUF_TYPE_FLOAT64: "<d",
}


def align(x, a):
    return (x + a - 1) // a * a


class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("unexpected end of file")
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        return self.read(self.u64()).decode("utf-8", "replace")

    def scalar(self, t):
        return struct.unpack(SCALAR_FMT[t], self.read(struct.calcsize(SCALAR_FMT[t])))[0]

    def value(self, t):
        if t == GGUF_TYPE_STRING:
            return self.string()
        if t == GGUF_TYPE_ARRAY:
            et = self.u32()
            n = self.u64()
            return (et, [self.value(et) for _ in range(n)])
        return self.scalar(t)


def fmt_scalar(v):
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def fmt_value(t, v):
    if t == GGUF_TYPE_ARRAY:
        et, items = v
        head = ", ".join(fmt_scalar(x) for x in items[:5])
        more = ", ..." if len(items) > 5 else ""
        return f"array<{TYPE_NAMES[et]}> n={len(items)} [{head}{more}]"
    return fmt_scalar(v)


def read_f32(reader, data_start, offset, n):
    reader.f.seek(data_start + offset)
    buf = array.array("f")
    buf.frombytes(reader.read(n * 4))
    if sys.byteorder != "little":
        buf.byteswap()
    return buf


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path, nargs="?", default=Path(__file__).with_name("demo-llama.gguf"))
    ap.add_argument("--tensor", help="only show this tensor")
    ap.add_argument("--values", type=int, default=0, help="print the first N values of the selected tensor")
    ap.add_argument("--no-stats", action="store_true")
    args = ap.parse_args()

    r = Reader(args.model)

    if r.read(4) != b"GGUF":
        raise SystemExit(f"{args.model}: not a GGUF file")

    version      = r.u32()
    n_tensors    = r.u64()
    n_kv         = r.u64()

    meta = []
    for _ in range(n_kv):
        key = r.string()
        t = r.u32()
        meta.append((key, t, r.value(t)))

    tensors = []
    for _ in range(n_tensors):
        name    = r.string()
        n_dims  = r.u32()
        dims    = [r.u64() for _ in range(n_dims)]
        ttype   = r.u32()
        offset  = r.u64()
        tensors.append((name, dims, ttype, offset))

    alignment = 32
    for key, _, value in meta:
        if key == "general.alignment":
            alignment = value

    data_start = align(r.f.tell(), alignment)

    print(f"file      : {args.model}")
    print(f"version   : {version}")
    print(f"tensors   : {n_tensors}")
    print(f"metadata  : {n_kv}")
    print(f"alignment : {alignment}")
    print(f"data at   : {data_start}")

    print("\nmetadata:")
    for key, t, value in meta:
        print(f"  {key} = {fmt_value(t, value)}")

    print("\ntensors (ne is ggml order, torch is reversed):")
    for i, (name, dims, ttype, offset) in enumerate(tensors):
        if args.tensor and name != args.tensor:
            continue

        n = 1
        for d in dims:
            n *= d
        torch = list(reversed(dims))
        line = f"  [{i:2d}] {name:<26} {GGML_TYPE_NAMES.get(ttype, f'type{ttype}')} ne={dims} torch={torch} n={n} offset={offset}"

        if ttype == GGML_TYPE_F32 and not args.no_stats:
            data = read_f32(r, data_start, offset, n)
            mean = sum(data) / n
            line += f" min={min(data):.6g} max={max(data):.6g} mean={mean:.6g}"

        print(line)

        if args.values and name == args.tensor:
            vals = read_f32(r, data_start, offset, min(n, args.values))
            print("       values:", ", ".join(f"{v:.6g}" for v in vals))


if __name__ == "__main__":
    main()
