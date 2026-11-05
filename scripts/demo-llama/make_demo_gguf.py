#!/usr/bin/env python3
"""Write a tiny single-layer demollama GGUF from scratch.

The writer is pure Python on purpose: no gguf-py and no numpy, so the file
layout stays visible. See dump_gguf.py for the reader side.
"""

import argparse
import array
import random
import struct
import sys
from pathlib import Path

ALIGNMENT = 32

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

N_EMBD    = 64
N_HEAD    = 4
N_HEAD_KV = 4
N_FF      = 128
N_LAYER   = 1
N_CTX     = 128
ROPE_DIM  = N_EMBD // N_HEAD
N_EMBD_KV = N_EMBD // N_HEAD * N_HEAD_KV

GGUF_MAGIC   = b"GGUF"
GGUF_VERSION = 3


def align(x, a=ALIGNMENT):
    return (x + a - 1) // a * a


def write_string(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def write_scalar(f, t, v):
    if t == GGUF_TYPE_UINT8:
        f.write(struct.pack("<B", v))
    elif t == GGUF_TYPE_INT8:
        f.write(struct.pack("<b", v))
    elif t == GGUF_TYPE_UINT16:
        f.write(struct.pack("<H", v))
    elif t == GGUF_TYPE_INT16:
        f.write(struct.pack("<h", v))
    elif t == GGUF_TYPE_UINT32:
        f.write(struct.pack("<I", v))
    elif t == GGUF_TYPE_INT32:
        f.write(struct.pack("<i", v))
    elif t == GGUF_TYPE_FLOAT32:
        f.write(struct.pack("<f", v))
    elif t == GGUF_TYPE_BOOL:
        f.write(struct.pack("<B", 1 if v else 0))
    elif t == GGUF_TYPE_STRING:
        write_string(f, v)
    elif t == GGUF_TYPE_UINT64:
        f.write(struct.pack("<Q", v))
    elif t == GGUF_TYPE_INT64:
        f.write(struct.pack("<q", v))
    elif t == GGUF_TYPE_FLOAT64:
        f.write(struct.pack("<d", v))
    else:
        raise ValueError(f"unsupported gguf type {t}")


def write_kv(f, key, value):
    write_string(f, key)

    if isinstance(value, bool):
        f.write(struct.pack("<I", GGUF_TYPE_BOOL))
        write_scalar(f, GGUF_TYPE_BOOL, value)
    elif isinstance(value, str):
        f.write(struct.pack("<I", GGUF_TYPE_STRING))
        write_scalar(f, GGUF_TYPE_STRING, value)
    elif isinstance(value, float):
        f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
        write_scalar(f, GGUF_TYPE_FLOAT32, value)
    elif isinstance(value, int):
        f.write(struct.pack("<I", GGUF_TYPE_UINT32))
        write_scalar(f, GGUF_TYPE_UINT32, value)
    elif isinstance(value, list):
        if not value:
            raise ValueError(f"empty array for {key}")
        if isinstance(value[0], str):
            et = GGUF_TYPE_STRING
        elif isinstance(value[0], bool):
            et = GGUF_TYPE_BOOL
        elif isinstance(value[0], int):
            et = GGUF_TYPE_INT32
        elif isinstance(value[0], float):
            et = GGUF_TYPE_FLOAT32
        else:
            raise ValueError(f"unsupported array element for {key}")

        f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
        f.write(struct.pack("<I", et))
        f.write(struct.pack("<Q", len(value)))
        for v in value:
            write_scalar(f, et, v)
    else:
        raise ValueError(f"unsupported value for {key}")


def build_vocab():
    # standard llama SPM layout: unknown, bos, eos, then 256 byte tokens
    tokens = ["<unk>", "<s>", "</s>"]
    types  = [2, 3, 3]

    for i in range(256):
        tokens.append(f"<0x{i:02X}>")
        types.append(6)

    words = [
        "the", "a", "an", "cat", "dog", "runs", "sits", "on", "mat",
        "hello", "world", "this", "is", "demo", "llama", "model",
        "one", "layer", "token", "tensor", "weight", "small", "test",
        "de", "mo", "ll", "ama", "wor", "ld", "he", "llo",
    ]
    seen = set(tokens)
    for w in words:
        piece = "\u2581" + w
        if piece in seen:
            continue
        seen.add(piece)
        tokens.append(piece)
        types.append(1)

    scores = [0.0] * len(tokens)
    return tokens, scores, types


def tensor_data(name, n, rng):
    if name.endswith("norm.weight"):
        return [1.0] * n
    return [rng.gauss(0.0, 0.02) for _ in range(n)]


def make_tensors(n_vocab, rng, tie_output):
    dims_by_name = [
        ("token_embd.weight",              [N_EMBD, n_vocab]),
        ("output_norm.weight",             [N_EMBD]),
    ]
    if not tie_output:
        dims_by_name.append(("output.weight", [N_EMBD, n_vocab]))

    for i in range(N_LAYER):
        dims_by_name += [
            (f"blk.{i}.attn_norm.weight",      [N_EMBD]),
            (f"blk.{i}.attn_q.weight",         [N_EMBD, N_EMBD]),
            (f"blk.{i}.attn_k.weight",         [N_EMBD, N_EMBD_KV]),
            (f"blk.{i}.attn_v.weight",         [N_EMBD, N_EMBD_KV]),
            (f"blk.{i}.attn_output.weight",    [N_EMBD, N_EMBD]),
            (f"blk.{i}.ffn_norm.weight",       [N_EMBD]),
            (f"blk.{i}.ffn_gate.weight",       [N_EMBD, N_FF]),
            (f"blk.{i}.ffn_down.weight",       [N_FF, N_EMBD]),
            (f"blk.{i}.ffn_up.weight",         [N_EMBD, N_FF]),
        ]

    tensors = []
    for name, dims in dims_by_name:
        n = 1
        for d in dims:
            n *= d
        tensors.append((name, dims, tensor_data(name, n, rng)))
    return tensors


def write_gguf(path, meta, tensors):
    # tensor offsets are relative to the start of the data section
    offsets = []
    off = 0
    for _, _, data in tensors:
        offsets.append(off)
        off = align(off + len(data) * 4)
    data_size = off

    with open(path, "wb") as f:
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(meta)))

        for key, value in meta:
            write_kv(f, key, value)

        for (name, dims, _), off in zip(tensors, offsets):
            write_string(f, name)
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", GGML_TYPE_F32))
            f.write(struct.pack("<Q", off))

        pad = align(f.tell()) - f.tell()
        f.write(b"\x00" * pad)
        data_start = f.tell()

        for (_, _, data), tensor_off in zip(tensors, offsets):
            assert f.tell() == data_start + tensor_off
            buf = array.array("f", data)
            if sys.byteorder != "little":
                buf.byteswap()
            f.write(buf.tobytes())
            f.write(b"\x00" * (align(f.tell() - data_start) - (f.tell() - data_start)))

        return data_start, data_size


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("demo-llama.gguf"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tie-output", action="store_true",
                    help="omit output.weight, the loader reuses token_embd.weight")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    tokens, scores, types = build_vocab()
    n_vocab = len(tokens)

    meta = [
        ("general.architecture", "demollama"),
        ("general.name",         "demollama tiny 1-layer"),
        ("general.file_type",    0),

        ("demollama.context_length",                    N_CTX),
        ("demollama.embedding_length",                  N_EMBD),
        ("demollama.block_count",                       N_LAYER),
        ("demollama.feed_forward_length",               N_FF),
        ("demollama.attention.head_count",              N_HEAD),
        ("demollama.attention.head_count_kv",           N_HEAD_KV),
        ("demollama.attention.layer_norm_rms_epsilon",  1e-5),
        ("demollama.attention.causal",                  True),
        ("demollama.rope.dimension_count",              ROPE_DIM),
        ("demollama.rope.freq_base",                    10000.0),
        ("demollama.vocab_size",                        n_vocab),

        ("tokenizer.ggml.model",           "llama"),
        ("tokenizer.ggml.tokens",          tokens),
        ("tokenizer.ggml.scores",          scores),
        ("tokenizer.ggml.token_type",      types),
        ("tokenizer.ggml.bos_token_id",    1),
        ("tokenizer.ggml.eos_token_id",    2),
        ("tokenizer.ggml.unknown_token_id", 0),
        ("tokenizer.ggml.add_bos_token",   True),
        ("tokenizer.ggml.add_eos_token",   False),
        ("tokenizer.ggml.add_space_prefix", True),
    ]

    tensors = make_tensors(n_vocab, rng, args.tie_output)
    data_start, data_size = write_gguf(args.out, meta, tensors)

    n_params = sum(len(data) for _, _, data in tensors)
    print(f"wrote {args.out}")
    print(f"  tensors : {len(tensors)}")
    print(f"  vocab   : {n_vocab}")
    print(f"  params  : {n_params}")
    print(f"  data    : {data_size} bytes at offset {data_start}")
    print(f"  file    : {args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
