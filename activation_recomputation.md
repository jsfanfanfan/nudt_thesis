# Megatron-LM 激活重计算调研

## 1. 概述

激活重计算通过在前向时少存中间激活、在反向时按需重计算，以**显存换算力**，从而支持更大 batch、更长序列或更大模型训练。Megatron-LM 支持两种粒度（**full** / **selective**）和两种层级划分方式（**uniform** / **block**），并可对指定子模块做细粒度重计算。

---

## 2. 配置参数（TransformerConfig / 训练参数）

所有与激活重计算相关的配置定义在 `TransformerConfig`（`megatron/core/transformer/transformer_config.py`）中。

| 参数 | 类型 | 说明 |
|------|------|------|
| **recompute_granularity** | `None` \| `'full'` \| `'selective'` | 重计算粒度。`None` 表示不重计算；`full` 对整个 Transformer 层做重计算；`selective` 只对 `recompute_modules` 中的指定的子模块做重计算。 |
| **recompute_method** | `None` \| `'uniform'` \| `'block'` | 仅用于 **full** 粒度：如何划分“哪些层”做 checkpoint。`uniform` 均匀分块；`block` 只对每个 pipeline stage 的前若干层做 checkpoint。**selective 时必须为 None**。 |
| **recompute_num_layers** | `None` \| int | 仅用于 **full** 粒度。`uniform` 时表示每个 chunk 的层数；`block` 时表示每个 stage 内做 重计算的层数。**selective 时必须为 None**。 |
| **distribute_saved_activations** | bool | 若为 True，将 checkpoint 保存的激活在 tensor parallel 组内分片存储，进一步省显存；仅 **full** 粒度且非 sequence parallel 时有效。 |
| **recompute_modules** | list[str] | 仅用于 **selective**。要重计算的子模块名列表，见下节。默认 `["core_attn"]`。 |

**兼容与废弃**：

- `--checkpoint-activations` 已废弃，需改用 `--recompute-activations` 或 `--recompute-granularity` / `--recompute-method`。
- `--recompute-activations` 等价于设置 `recompute_granularity='selective'`（不指定时即默认 `recompute_modules=["core_attn"]`）。

---

## 3. 重计算粒度（recompute_granularity）

### 3.1 `full`(整层 checkpoint)

- **含义**：以“层”为单位做激活重计算；前向只保存每个 chunk/block 的输入，反向时重新执行该段前向再算梯度。
- **要求**：必须同时设置 `recompute_method`（`uniform` 或 `block`）和 `recompute_num_layers`。
- **入口**：`TransformerBlock.forward()` 在 `recompute_granularity == 'full'` 且 `training` 时调用 `_checkpointed_forward()`，否则逐层正常前向。
- **实现位置**：`megatron/core/transformer/transformer_block.py` 中 `_checkpointed_forward()`，内部按 `recompute_method` 调用 `checkpoint_handler(custom(...))`，底层为 `tensor_parallel.checkpoint` 或 `te_checkpoint`（FP8/FP4 时）。

### 3.2 `selective`(子模块级 checkpoint)

- **含义**：只对 `recompute_modules` 中列出的子模块做重计算，其余部分正常存激活。
- **要求**：`recompute_method` 必须为 `None`，`recompute_num_layers` 必须为 `None`；selective 始终作用于**所有层**。
- **实现位置**：各子模块在各自 forward 中根据 `config.recompute_granularity == 'selective'` 和 `config.recompute_modules` 决定是否用 checkpoint / output-discarding checkpoint。

---

## 4. 层级划分方式（recompute_method，仅 full）

在 **full** 粒度下，用 `recompute_method` 和 `recompute_num_layers` 决定“哪些层”被包在一个 checkpoint 里。

### 4.1 `uniform`（均匀分块）

- **语义**：将当前 pipeline stage 内的所有层均匀分成多个 chunk，每个 chunk 包含 `recompute_num_layers` 层；每个 chunk 的**输入**被 checkpoint，chunk 内前向一次执行，反向时整块重算再反传。
- **特点**：checkpoint 数量少，显存占用更小，但每次反向重算的层数较多。
- **代码**：`transformer_block.py` 中 `_checkpointed_forward()`，`layer_idx` 按 `recompute_num_layers` 递增，每次 `checkpoint_handler(custom(layer_idx, layer_idx + recompute_num_layers))`。

### 4.2 `block`（仅前 N 层）

- **语义**：在当前 pipeline stage 内，只对**前 `recompute_num_layers` 层**做 checkpoint；后续层不做 checkpoint，正常存激活。
- **特点**：在显存允许的情况下可减少重计算量，更充分利用显存。
- **特殊**：FP8/FP4 且当前 `hidden_states.requires_grad == False` 时，会跳过若干层不重算（`recompute_skip_num_layers`），以满足重入 autograd 对至少一个带梯度的输入的要求。
- **代码**：`transformer_block.py` 中按 `layer_idx` 循环，若 `recompute_skip_num_layers <= layer_idx < recompute_skip_num_layers + recompute_num_layers` 则对该层调用 `checkpoint_handler(custom(layer_idx, layer_idx + 1))`，否则直接执行 `custom(layer_idx, layer_idx + 1)(...)`。

---

## 5. 子模块重计算（recompute_modules，仅 selective）

`recompute_modules` 指定在 **selective** 粒度下要对哪些子模块做重计算。不同子模块使用的 checkpoint 机制不同：

- **标准 checkpoint**：保存输入，反向时重新前向再反传（如 PyTorch `checkpoint` / TE `checkpoint`）。
- **Output-discarding checkpoint**：前向完成后丢弃该子模块输出占用的显存，在反向时通过 grad hook 再算一遍并把结果写回原张量元数据，进一步省显存。

下表给出各选项、对应实现位置及 checkpoint 类型。

| 模块名 | 含义 | 实现位置 | Checkpoint 类型 |
|--------|------|----------|------------------|
| **core_attn** | 核心注意力（QKV→Attention→输出） | `attention.py` / `multi_latent_attention.py`，`checkpoint_core_attention` 时调用 `_checkpointed_attention_forward()` | 标准 checkpoint |
| **mlp** | 稠密 MLP（非 MoE） | `transformer_layer.py`，`recompute_mlp` 时对 `self.mlp` 做 `tensor_parallel.checkpoint` 或 `te_checkpoint` | 标准 checkpoint |
| **moe** | 整块 MoE 层（router + experts + combine） | `moe_layer.py`，`moe_layer_recompute` 时对整段 forward 做 checkpoint；或 `MoETransformerLayer` 中 partial CUDA graph 时对 `_forward_mlp_partial_cudagraphs` 做 checkpoint | 标准 checkpoint |
| **shared_experts** | MoE 的共享专家 | `moe_layer.py` / `shared_experts.py`，`shared_experts_recompute` 时对 `shared_experts` 的 forward 做 checkpoint | 标准 checkpoint |
| **layernorm** | 输入 LayerNorm + MLP 前 LayerNorm | `transformer_layer.py`，`recompute_input_layernorm` / `recompute_pre_mlp_layernorm`，使用 `CheckpointWithoutOutput` | Output-discarding |
| **moe_act** | MoE 专家内部激活函数（如 SiLU）* 专家权重 | `moe/experts.py`（GroupedMLP / TEGroupedMLP），`activation_recompute` 时对激活部分做 output-discarding | Output-discarding |
| **mla_up_proj** | MLA 的 QKV 上投影与 RoPE | `multi_latent_attention.py`，`recompute_up_proj` 时对 `qkv_up` 输出用 `CheckpointWithoutOutput` | Output-discarding |

**默认**：若未指定 `recompute_modules`，则设为 `["core_attn"]`。

**约束与说明**：

- **core_attn**：使用 TE 实现时可能是融合算子，不一定需要再对 core_attn 做 recompute，配置时会打 warning。
- **moe_act**：仅在与 `moe_grouped_gemm` 同时开启时支持。
- **mla_up_proj**：仅在 `multi_latent_attention` 开启时有效。
- **shared_experts**：与 `moe_shared_expert_overlap` 互斥。
- **layernorm / moe_act** 在 FP8 下若使用 delayed scaling，需 TE ≥ 2.6.0dev0，且 delayed scaling 与 moe_act/layernorm recompute 不兼容（会报错）。

---

## 6. 底层 Checkpoint 实现

### 6.1 标准 Checkpoint（整段前向重算）

- **PyTorch 路径**：`megatron/core/tensor_parallel/random.py` 中 `checkpoint()` → `CheckpointFunction`。前向保存输入与 RNG 状态，不保存中间激活；反向时用保存的输入和 RNG 重新跑一遍前向再 `torch.autograd.backward`。
- **Transformer Engine 路径**：`megatron/core/extensions/transformer_engine.py` 中 `te_checkpoint()`，内部调用 `transformer_engine.pytorch.distributed.checkpoint`，支持 FP8/FP4 与 `distribute_saved_activations`。
- **使用场景**：full 整层、selective 的 core_attn / mlp / moe / shared_experts。

### 6.2 Output-discarding Checkpoint（丢弃输出再重算）

- **实现**：`megatron/core/tensor_parallel/random.py` 中 `CheckpointWithoutOutput`。
  - 前向：`CheckpointWithoutOutputFunction` 在 `no_grad` 下执行，只保存输入到 `ctx`，输出不保留在计算图里；调用方随后可对输出调用 `discard_output_and_register_recompute(hook_tensor)`，将输出张量 storage resize 为 0，并在 `hook_tensor` 上注册 backward hook。
  - 反向：当 `hook_tensor` 需要梯度时触发 hook，在 hook 里用保存的输入重新前向，得到新输出后把数据写回原先输出张量的 storage（保持元数据一致），供后续 backward 使用。
- **使用场景**：selective 的 layernorm、moe_act、mla_up_proj；与 fine-grained activation offloading 配合时可进一步省显存。

---

## 7. 与 FP8 / FP4 的交互

- **full 粒度**：`TransformerBlock._checkpointed_forward()` 在 FP8/FP4 时通过 `te_checkpoint` 做整层 checkpoint，并可在每层使用 `get_fp8_context` / `get_fp4_context`。
- **selective 子模块**：  
  - core_attn / mlp / moe / shared_experts 在 FP8/FP4 下使用 `te_checkpoint`。  
  - layernorm / moe_act 使用 `CheckpointWithoutOutput`，其内部在 recompute 阶段会进入 `activation_recompute_forward(..., recompute_phase=True)` 和 `fp8_autocast`，以保持数值一致。
- **限制**：  
  - CPU offloading 与任意激活重计算互斥（config 校验会报错）。  
  - FP8 delayed scaling 与 moe_act、layernorm recompute 不兼容。  
  - 部分路径在 FP4 下尚未完全支持（代码中留有 TODO）。

---

## 8. 与 MoE 的交互

- **moe**：整层 MoE 可用 standard checkpoint 包住整段 forward（router + dispatch + experts + combine）；在 partial CUDA graph（`cuda_graph_scope` 含 moe_router/moe_preprocess 且 `cuda_graph_impl == "local"`）时，由 `MoETransformerLayer` 对 `_forward_mlp_partial_cudagraphs` 做 checkpoint。
- **shared_experts**：非 overlap 模式下，`shared_experts` 的 forward 可单独被 checkpoint；与 `moe_shared_expert_overlap` 互斥。
- **moe_act**：专家内部激活用 output-discarding，需 `moe_grouped_gemm`；legacy GroupedMLP 在 FP8/FP4 下不支持 moe_act recompute。
- **layernorm**：`recompute_pre_mlp_layernorm` 在 MoE + CUDA Graph（router/preprocess scope）下有一系列条件限制（如 alltoall token dispatcher 或 latent MoE 等），不满足时会被禁用并打 warning。

## 11. 配置校验摘要（TransformerConfig）

- `recompute_granularity` ∈ `{None, 'full', 'selective'}`。
- **full** 时：`recompute_method` ∈ `{'uniform','block'}`，`recompute_num_layers` 为 1 到当前 stage 层数之间的整数。
- **selective** 时：`recompute_method` 与 `recompute_num_layers` 必须为 `None`；`recompute_modules` 只能包含规定的子模块集合；moe_act / mla_up_proj / shared_experts 等有额外前置条件。
- **distribute_saved_activations** 仅允许在 full 且非 sequence parallel 时为 True。
- CPU offloading 与 recompute 不能同时开启。

---

## 12. 举例：32 层、PP=8、Dense 模型

假设：**总层数 32**，**PP=8**，**dense 模型**（无 MoE）。则每个 pipeline stage 上有 **4 层**（32÷8=4），即每个 rank 的 `TransformerBlock` 只包含 4 个 Transformer 层（例如对应全局层 0–3、4–7、…、28–31）。下面按不同重计算配置说明“每个 stage 内”的行为。

### 12.1 不重计算（baseline）

```bash
# 不设置 recompute_granularity，或 recompute_granularity=None
```
- 每个 stage 内 4 层正常前向，**所有中间激活**都保留，反向直接使用。
- **显存最大**，**计算量最小**（无重算）。

---

### 12.2 full + uniform（按层数均匀分块）

**思路**：把本 stage 的 4 层切成若干“块”，每块做一次 checkpoint（只存块入口的输入，块内前向不存中间激活，反向时整块重算）。

| 配置 | 每 stage 内的划分 | 每 stage checkpoint 次数 | 反向时一次重算的层数 | 显存 | 计算 |
|------|-------------------|---------------------------|------------------------|------|------|
| `recompute_num_layers=1` | 4 块：[L0], [L1], [L2], [L3] | 4 | 1 | 最小 | 最大 |
| `recompute_num_layers=2` | 2 块：[L0,L1], [L2,L3] | 2 | 2 | 中 | 中 |
| `recompute_num_layers=4` | 1 块：[L0,L1,L2,L3] | 1 | 4 | 大（只存 stage 输入） | 最小 |

**示例命令**：

```bash
# 每 1 层一块 → 最省显存，重算最多
--recompute-granularity full --recompute-method uniform --recompute-num-layers 1

# 每 2 层一块 → 折中
--recompute-granularity full --recompute-method uniform --recompute-num-layers 2

# 整 stage 一块 → 只存 stage 入口，反向重算本 stage 全部 4 层
--recompute-granularity full --recompute-method uniform --recompute-num-layers 4
```

注意：`recompute_num_layers` 不能超过本 stage 层数 4，否则会越界或校验报错。

---

### 12.3 full + block（只对“前 N 层”做 checkpoint）

**思路**：在本 stage 内，只对**前 `recompute_num_layers` 层**逐层做 checkpoint；后面的层**不做** checkpoint，正常存激活。

| 配置 | 本 stage 内行为 | 显存 | 计算 |
|------|-----------------|------|------|
| `recompute_num_layers=1` | 只有第 0 层 checkpoint；第 1、2、3 层正常存激活 | 较大 | 只重算 1 层 |
| `recompute_num_layers=2` | 第 0、1 层 checkpoint；第 2、3 层正常存激活 | 中 | 重算 2 层 |
| `recompute_num_layers=4` | 第 0、1、2、3 层都 checkpoint；无“不 checkpoint”的层 | 最小（与 uniform 4 类似） | 重算 4 层 |

**示例命令**：

```bash
# 只对每 stage 的前 1 层做 checkpoint，后 3 层不重算
--recompute-granularity full --recompute-method block --recompute-num-layers 1

# 前 2 层 checkpoint，后 2 层正常
--recompute-granularity full --recompute-method block --recompute-num-layers 2

# 本 stage 全部 4 层都做 checkpoint（显存最省，重算 4 层）
--recompute-granularity full --recompute-method block --recompute-num-layers 4
```

**和 uniform 的对比**（本 stage 4 层）：

- **uniform, num_layers=2**：2 个 checkpoint，每个包 2 层；反向时每次重算 2 层。
- **block, num_layers=2**：只有前 2 层各自 1 个 checkpoint；后 2 层不 checkpoint，显存比 uniform 2 略大，但少重算后 2 层。

---

### 12.4 selective（子模块级，所有层都参与）

**思路**：不按“层”分块，而是**每一层**里只对指定子模块做重计算（如 core_attn、mlp、layernorm），其余部分照常存激活。对 32 层、PP=8 而言，**每个 stage 的 4 层**都会应用同一套 selective 规则。

| 配置 | 每层内重算内容 | 显存 | 计算 |
|------|----------------|------|------|
| `recompute_modules=[core_attn]`（默认） | 只重算 core attention | 省一部分 | 只多算 attention |
| `[core_attn, mlp]` | 重算 attention + MLP | 更省 | 多算 attn + MLP |
| `[core_attn, mlp, layernorm]` | 重算 attention + MLP + 两个 LayerNorm | 最省（在 selective 内） | 最多 |

**示例命令**：

```bash
# 仅 core attention 重算（等价于 --recompute-activations）
--recompute-granularity selective --recompute-modules "[core_attn]"

# attention + MLP 重算
--recompute-granularity selective --recompute-modules "[core_attn, mlp]"

# attention + MLP + layernorm 重算（dense 下常用“最省显存”组合）
--recompute-granularity selective --recompute-modules "[core_attn, mlp, layernorm]"
```

注意：selective 下**不能**设置 `recompute_method` / `recompute_num_layers`，所有层（包括本 stage 的 4 层）都按同一套 `recompute_modules` 处理。

---

### 12.5 简要对比(单 stage 4 层视角)

| 用法 | 每 stage 存多少 | 每 stage 重算多少 | 适用场景 |
|------|-----------------|-------------------|----------|
| 无重算 | 4 层全部激活 | 0 | 显存充足 |
| full + uniform, num=1 | 4 个入口 | 4 次 × 1 层 | 显存紧张，可接受较多重算 |
| full + uniform, num=4 | 1 个入口 | 1 次 × 4 层 | 希望少 checkpoint、少重算次数 |
| full + block, num=2 | 前 2 层入口 + 后 2 层激活 | 2 层 | 折中：略省显存、控制重算量 |
| selective, [core_attn,mlp,layernorm] | 每层只存非重算子模块激活 | 每层 attn+mlp+ln 各 1 次 | 细粒度省显存，dense 常用 |

## 14. 相关文件索引

| 功能 | 文件路径 |
|------|----------|
| 配置定义与校验 | `megatron/core/transformer/transformer_config.py` |
| full 粒度、uniform/block | `megatron/core/transformer/transformer_block.py` |
| selective：layernorm / mlp | `megatron/core/transformer/transformer_layer.py` |
| selective：core_attn | `megatron/core/transformer/attention.py`，`multi_latent_attention.py` |
| selective：moe / shared_experts / moe_act | `megatron/core/transformer/moe/moe_layer.py`，`experts.py`，`shared_experts.py` |
| 标准 checkpoint | `megatron/core/tensor_parallel/random.py`（`checkpoint` / `CheckpointFunction`） |
| Output-discarding | `megatron/core/tensor_parallel/random.py`（`CheckpointWithoutOutput`） |
| TE checkpoint | `megatron/core/extensions/transformer_engine.py`（`te_checkpoint`） |
| 参数解析与兼容 | `megatron/training/arguments.py`，`yaml_arguments.py` |

---

以上即为 Megatron-LM 中激活重计算策略的完整梳理，便于按显存与算力需求选择 `recompute_granularity`、`recompute_method` 和 `recompute_modules` 组合。