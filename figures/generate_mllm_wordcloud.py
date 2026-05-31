#!/usr/bin/env python3
"""从 MLLM 发展时间线图提取模型名称，生成词云图。"""

from pathlib import Path

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 从图中提取的模型/技术名称（按出现频次加权：完整名称计 1，系列名在词云中自然聚合）
MODEL_NAMES = [
    # 2023 早期
    "OFA", "Flamingo", "CLIP", "PaLM-E", "LLaVA", "BLIP-2", "LLaMA-Adapter",
    "Mini-Gemini", "KOSMOS-1", "GPT-4", "DeepSeek VL", "mPLUG-Owl", "VideoChat",
    "EmbodiedGPT", "InstructBLIP", "MiniGPT-4", "MultiModal-GPT", "Pengi",
    "LLaMA-Adapter V2", "LTU", "PandaGPT", "PMC-VQA", "VideoLLM",
    # 2023 中后期
    "Video-LLaMA", "SegPoint", "Video-ChatGPT", "Shikra", "MotionGPT", "RT-2",
    "Kosmos-2", "Med-Flamingo", "MIMIC-IT", "3D-LLM", "LLaVA-Med", "AudioPaLM",
    "SEED", "InternVL", "mPLUG-Owl2", "MiniGPT-v2", "Honeybee", "LMDrive",
    "GSVA", "VILA", "Chat-UniVi", "Merlin", "SALMONN", "ViP-LLaVA", "ManipLLM",
    "Griffon", "CogAgent", "SmartEdit", "LLaVA-1.5", "LLaFS", "Osprey", "GeoChat",
    "X-InstructBLIP", "Ferret", "CogVLM", "SPHINX", "ShareGPT4V", "DRESS", "GLaMM",
    "Monkey", "SEED-LLaMA", "LION", "TimeChat", "LLaMA-VID", "RoboFlamingo",
    # 2024 早期
    "InternVL 1.5", "Morph-Tokens", "PlausiVL", "SNIFFER", "Groma", "MA-LMM",
    "MoVA", "MoMA", "LM-XComposer2-4KHD", "NExT-GPT", "MGIE", "DreamLLM",
    "BLIVA", "MMICL", "AnomalyGPT", "LISA", "EAVL", "Qwen-VL", "Idefics",
    "PointLLM", "Kosmos-2.5", "InternLM-XComposer",
    # 2024 中期
    "SkyEyeGPT", "V2T Tokenizer", "MoE-LLaVA", "MM1", "Idefics2", "ModaVerse",
    "LHRS-Bot", "InternLM-XComposer2", "LGVI", "GROUNDHOG", "LLaVA-UHD", "FILM",
    "DoCo",
    # 2024 后期
    "LongLLaVA", "CALVIN", "LongVILA", "Qwen2-VL", "mPLUG-Owl3", "EMMA",
    "DeeR-VLA", "DPE-CLIP", "TransAgent", "DeepSeek VL2", "InternVL 2.5",
    "VideoLLM-online", "ShareGPT4Video", "UniAudio 1.5", "Cambrian-1",
    "VisionLLM v2", "ControlMLLM", "StimuVAR", "MoME", "Emotion-LLaMA",
    # 2025
    "InternVL 3", "Ola", "SEAL", "Qwen2.5-VL", "Qwen2.5-Omni",
]

# 系列/家族加权，使词云突出主流技术脉络
FAMILY_BOOST = {
    "LLaVA": 8,
    "GPT": 6,
    "InternVL": 7,
    "Qwen": 6,
    "MiniGPT": 5,
    "Kosmos": 4,
    "KOSMOS": 4,
    "BLIP": 4,
    "Flamingo": 3,
    "CLIP": 3,
    "LLaMA": 5,
    "VL": 4,
    "Video": 4,
    "mPLUG": 4,
    "Cog": 3,
    "SEED": 3,
    "XComposer": 3,
    "Idefics": 3,
    "MoE": 2,
    "DeepSeek": 4,
}


def build_frequencies(names: list[str]) -> dict[str, float]:
    """构建词频字典：每个词仅出现一次，通过权重体现重要性（避免重复排版）。"""
    freq: dict[str, float] = {}
    for name in names:
        weight = 1.0
        for token, boost in FAMILY_BOOST.items():
            if token.lower() in name.lower():
                weight += boost * 0.25
        freq[name] = freq.get(name, 0.0) + weight
    return freq


def find_font() -> str | None:
    """查找可用于英文+数字的字体。"""
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    frequencies = build_frequencies(MODEL_NAMES)
    font_path = find_font()

    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        max_words=200,
        colormap="viridis",
        prefer_horizontal=0.85,
        min_font_size=10,
        max_font_size=120,
        relative_scaling=0.5,
        font_path=font_path,
        margin=10,
        random_state=42,
        collocations=False,  # 禁止把相邻重复识别为二元组并排绘制
    ).generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        "Multimodal LLM Landscape (2023–2025)",
        fontsize=14,
        pad=12,
    )
    plt.tight_layout()

    png_path = out_dir / "mllm_timeline_wordcloud.png"
    pdf_path = out_dir / "mllm_timeline_wordcloud.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 同时导出纯文本清单
    txt_path = out_dir / "mllm_timeline_extracted_names.txt"
    unique = sorted(set(MODEL_NAMES), key=str.lower)
    txt_path.write_text("\n".join(unique), encoding="utf-8")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {txt_path} ({len(unique)} unique names)")


if __name__ == "__main__":
    main()
