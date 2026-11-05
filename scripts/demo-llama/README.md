# demollama: 单层 llama 教学模型

这个目录生成一个很小的 GGUF，并配套一个最小架构实现，用来学习：

- GGUF 文件结构：metadata + tensor 目录 + 权重数据
- llama.cpp 如何把 GGUF 映射到 `src/models/demollama.cpp`
- 单层前向图：embedding -> RMSNorm -> QKV -> RoPE -> causal attention -> residual -> RMSNorm -> SwiGLU -> residual -> RMSNorm -> lm_head

权重是随机数，输出没有语义，只用来跑通流程和观察中间结果。

## 生成 GGUF

```sh
python scripts/demo-llama/make_demo_gguf.py
```

默认写入 `scripts/demo-llama/demo-llama.gguf`。加 `--tie-output` 则不写 `output.weight`，加载时复用 `token_embd.weight`。

生成脚本只用 Python 标准库，直接按 GGUF v3 格式写文件，方便对照格式规范。

## 运行

`llama-simple` 是最短路径，不做 chat template：

```sh
build-x64-windows-llvm-release/bin/llama-simple.exe \
  -m scripts/demo-llama/demo-llama.gguf -n 8 "hello world"
```

`llama-cli` 也可以加载同一个文件。这个版本的 `llama-cli` 会进入对话模式，加 `-st` 只跑一轮：

```sh
build-x64-windows-llvm-release/bin/llama-cli.exe \
  -m scripts/demo-llama/demo-llama.gguf -p "hello world" -n 8 --temp 0 -c 128 -st
```

## 查看 GGUF

```sh
python scripts/demo-llama/dump_gguf.py
python scripts/demo-llama/dump_gguf.py --tensor blk.0.attn_q.weight --values 8
python scripts/demo-llama/dump_gguf.py --no-stats
```

输出包含 metadata、每个 tensor 的 `ne`（ggml 顺序）、对应的 torch shape、元素个数、偏移和数值统计。

`ne` 是反过来的 PyTorch shape：`ne=[64, 290]` 对应 PyTorch `[290, 64]`。

仓库里还有官方工具，可以按需使用：

```sh
# 官方 Python 工具，支持 JSON / Markdown，需要 numpy
python -m pip install "numpy~=2.4.6"
python gguf-py/gguf/scripts/gguf_dump.py scripts/demo-llama/demo-llama.gguf --json
python gguf-py/gguf/scripts/gguf_dump.py scripts/demo-llama/demo-llama.gguf --markdown
python gguf-py/gguf/scripts/gguf_dump.py scripts/demo-llama/demo-llama.gguf --json --json-array
```

`--json` 只导出 metadata 和 tensor 目录（shape/type/offset），不读取权重值；`--json-array` 才会把 tokenizer 数组这类内容完整写出。

C++ 侧有个 `examples/gguf` 读写示例：

```sh
build-x64-windows-llvm-release/bin/llama-gguf.exe scripts/demo-llama/demo-llama.gguf r n
```

最后的 `n` 表示跳过数据校验。它会打印 metadata、tensor 信息和每个 tensor 前 10 个 float。

本目录的 `dump_gguf.py` 不依赖 numpy，额外提供 min/max/mean 统计和 `--values` 抽样，方便在没有 numpy 的环境里直接看文件。

## 查看中间激活

`llama-eval-callback` 会打印计算图每个节点的张量。`demollama.cpp` 里的 `cb(...)` 就是这些名字的来源：

```sh
build-x64-windows-llvm-release/bin/llama-eval-callback.exe \
  -m scripts/demo-llama/demo-llama.gguf -p "hello" -n 1 --temp 0
```

重点看 `attn_norm-0`、`Qcur-0`、`Kcur-0`、`Vcur-0`、`kqv_out-0`、`ffn_norm-0`、`l_out-0`、`result_norm`、`result_output`。

## 模型参数

| 项 | 值 |
| --- | --- |
| n_layer | 1 |
| n_embd | 64 |
| n_head | 4 |
| n_head_kv | 4 |
| head_dim | 16 |
| n_ff | 128 |
| n_ctx_train | 128 |
| vocab | 290 |

## 张量表

| 名字 | ggml ne | 说明 |
| --- | --- | --- |
| token_embd.weight | [64, 290] | token embedding |
| output_norm.weight | [64] | 最终 RMSNorm |
| output.weight | [64, 290] | lm_head，可选 |
| blk.0.attn_norm.weight | [64] | attention 前 RMSNorm |
| blk.0.attn_q.weight | [64, 64] | Q 投影 |
| blk.0.attn_k.weight | [64, 64] | K 投影 |
| blk.0.attn_v.weight | [64, 64] | V 投影 |
| blk.0.attn_output.weight | [64, 64] | attention 输出投影 |
| blk.0.ffn_norm.weight | [64] | FFN 前 RMSNorm |
| blk.0.ffn_gate.weight | [64, 128] | SwiGLU gate |
| blk.0.ffn_down.weight | [128, 64] | FFN down |
| blk.0.ffn_up.weight | [64, 128] | FFN up |

## 新增架构的注册点

新增 `demollama` 架构只改这些地方：

1. `src/llama-arch.h`：加 `LLM_ARCH_DEMOLLAMA`
2. `src/llama-arch.cpp`：架构名 `demollama`
3. `src/llama-model.cpp`：`llama_model_mapping` 里创建模型，`llama_model_rope_type` 里返回 NORM
4. `src/models/models.h`：声明 `llama_model_demollama`
5. `src/models/demollama.cpp`：加载超参、张量和构建前向图

新增 `.cpp` 后需要重新跑一次 CMake configure，因为 `src/CMakeLists.txt` 用 `file(GLOB ...)` 收集源文件。

## 可以继续改的地方

- 改 `N_LAYER = 2`，看每层如何循环
- 加 `--tie-output`，看权重共享
- 把 `N_HEAD_KV` 改成 2，看 GQA 里 K/V 的维度变化
- 用 `llama-quantize` 把 F32 转成 Q8_0，再 dump 对比张量类型和大小
