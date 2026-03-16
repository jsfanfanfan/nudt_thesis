请将开题报告 PDF 中的下列图导出为 PDF 或 PNG，放入本目录（figures/），并按下表命名（扩展名 .pdf 或 .png 均可，LaTeX 会自动识别）：

  fig01_mllm_components   — 图1 多模态大语言模型的核心组成部分
  fig02_megatron_mllm     — 图2 使用 Megatron 训练多模态大语言模型
  fig03_data_parallel     — 图3 数据并行（正文 2.2 现用 TikZ 绘制）
  fig04_tensor_parallel   — 图4 张量并行（正文 2.2 现用 TikZ，风格参考 Megatron-LM）
  fig05_pipeline_parallel — 图5 流水线并行（正文 2.2 现用 TikZ，风格参考 GPipe/Megatron）
  fig06_zero              — 图6 零冗余优化器 ZeRO

若暂缺某图，可先放一张空白图或同尺寸占位图，避免编译报错。

图源参考（若需从论文替换为原图）：
  - 张量并行 / 序列并行：Megatron-LM (arxiv.org/abs/1909.08053)、Megatron 2023 长序列 (Context/Sequence Parallelism)
  - 流水线并行：Megatron GPU Clusters (arxiv.org/abs/2104.04473)、NVIDIA 博客 “Scaling LM Training to a Trillion Parameters”
  - 上下文并行（CP）：Megatron Core 文档、Huggingface Context Parallelism 文档
  - 专家并行（EP）：GShard (Google)、MoE 综述中的 token 路由与专家分布图
