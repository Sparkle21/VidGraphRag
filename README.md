<div align="center">

# VidGraphRAG

**一个「黑屏」的全模态视频 RAG 引擎 —— 无 GUI、纯命令行 / HTTP 服务，专为超长视频的理解与检索而设计。**

</div>

---

VidGraphRAG 是一个 **headless（无界面）** 的视频检索增强生成引擎。给它一批长视频，它会做分镜切分、多模态编码、语音转写、视觉语言描述、知识图谱构建，然后你就可以用**文字、图片、音频、甚至一段视频片段**去检索并问答。

它**不依赖 Electron / 前端 / 桌面应用**，一条 `pip install` 加一个 CLI 就能在你自己的机器上跑起来。核心是纯 Python 库 `vidgraphrag/`，可嵌入、可命令行、可作为 HTTP 服务。

## 核心特性

- **真·全模态检索**：视觉、声学、文本（VLM/ASR）、词法四路并行召回并加权融合。
  - 声学通道使用 **ImageBind 音频模态**对原始音频波形直接编码（1024 维），而非"拿 ASR 文本冒充音频"，实现真正的"以音搜视频"。
- **查询侧全模态输入**：`query_by_image` / `query_by_audio` / `query_by_video_clip`——用一张图、一段声音、一小段视频去搜整个库。
- **Shot–Segment 双层架构**：底层按镜头（shot）保留细粒度时空特征，上层做语义聚合（segment）与知识图谱，兼顾精度与长程语义。
- **知识图谱 + Wiki 编译层**：实体抽取、关系构建、社区发现，并沉淀为可检索的 Markdown 知识库。
- **优雅降级**：无 GPU / 未装重依赖时，自动退回"纯文本 embedding + BM25 词法"通路，核心问答仍可用。
- **多 LLM 后端**：阿里云百炼 DashScope、OpenAI、Azure、Ollama、DeepSeek 任选。

## 安装

```bash
# 克隆仓库
git clone https://github.com/<your-name>/VidGraphRAG.git
cd VidGraphRAG

# 方式一（推荐）：一次性装齐视频处理 + ImageBind + ASR + HNSW + HTTP 服务
pip install -e ".[full]"

# 方式二：按需装最小集
pip install -e .                 # 核心：纯文本 + BM25 + 知识图谱通路即可跑
pip install -e ".[video]"        # 分镜检测 / 切片 / 关键帧（moviepy, scenedetect）
pip install -e ".[imagebind]"    # 视觉与真声学通道（torch + ImageBind，需 GPU 或大内存）
pip install -e ".[asr]"          # 云端 ASR（DashScope fun-asr）
pip install -e ".[hnsw]"         # 多模态向量索引
pip install -e ".[server]"       # vidgraphrag serve 所需的 FastAPI/uvicorn
```

> 建议 Python ≥ 3.10。需要完整多模态能力时优先装 `[full]`；只想快速体验文本问答，装核心即可。

## 配置

引擎通过 `--llm` 选择后端，**API Key 一律从环境变量读取**，代码里不内嵌任何凭据。

| 后端 (`--llm`) | 所需环境变量 | 说明 |
| --- | --- | --- |
| `dashscope` | `DASHSCOPE_API_KEY` | 阿里云百炼，默认对话 `qwen3-8b`、嵌入 `text-embedding-v2`、画面描述 `qwen3-vl-8b-instruct`（**全模态能力最完整，推荐**） |
| `openai` / `openai-mini` | `OPENAI_API_KEY` | OpenAI |
| `azure` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` | Azure OpenAI |
| `ollama` | 无（本地服务） | 本地 Ollama |
| `deepseek` | `DEEPSEEK_API_KEY`, `SILICONFLOW_API_KEY` | DeepSeek + 硅基流动嵌入 |

以 DashScope 为例：

```bash
export DASHSCOPE_API_KEY='sk-xxxxxxxxxxxxxxxx'
```

> **ImageBind 权重**：视觉/声学通道需要 `imagebind_huge.pth`（约 4.8GB）。引擎会依次尝试 `IMAGEBIND_CHECKPOINT` 环境变量、当前目录 `.checkpoints/`、仓库内桌面端目录；均找不到时才联网下载。建议离线预放：
> ```bash
> export IMAGEBIND_CHECKPOINT=/path/to/imagebind_huge.pth
> ```

## 快速开始

**1. 索引视频**（构建双层存储 + 知识图谱）

```bash
vidgraphrag ingest --llm dashscope --workdir ./my-videos path/to/long_video.mp4
```

输出示例：

```
Indexing 1 video(s) into './my-videos' (mode=fusion) ...
✓ Ingest complete.
```

**2. 文本问答**

```bash
vidgraphrag query --llm dashscope --workdir ./my-videos "视频里出现了哪些主要人物？他们在做什么？"
```

**3. 以图 / 以音 / 以视频片段搜视频**（需 ImageBind）

```bash
vidgraphrag query --llm dashscope --workdir ./my-videos --image  query_frame.jpg
vidgraphrag query --llm dashscope --workdir ./my-videos --audio  sound_clip.wav
vidgraphrag query --llm dashscope --workdir ./my-videos --video  short_clip.mp4 --topk 5
```

**4. 作为 HTTP 服务**（需 `[server]` extra）

```bash
vidgraphrag serve --llm dashscope --workdir ./my-videos --host 127.0.0.1 --port 8000
```

## CLI 参考

```
vidgraphrag [-v|-vv] <子命令> [参数]

子命令:
  ingest <video...>     索引视频（可多个），构建双层存储与知识图谱
  query [问题]          文本问答；用 --image/--audio/--video 时问题可为空
  serve                 启动 HTTP 服务（需 [server] extra）

公共参数:
  --workdir DIR         索引/存储工作目录 (默认 ./videorag-workdir)
  --llm NAME            LLM 后端 (默认 openai；见上表)
  --legacy              关闭 Shot 双层架构，退回固定时长分段

query 专有:
  --image PATH          以图搜视频
  --audio PATH          以音搜视频（音频 wav/mp3 等）
  --video PATH          以视频片段搜视频
  --topk N              返回条数 (默认 5)
  --no-reference        答案不附带视频片段引用
  --with-imagebind      文本查询也启用视觉通路（需加载 ImageBind）
```

## 架构概览

```
           ┌──────────── 视频 ────────────┐
           │  分镜检测 → Shot（细粒度）    │
           │  语义聚合 → Segment（长程）   │
           └──────────────┬───────────────┘
                          │  多模态编码
   ┌──────────┬───────────┼───────────┬──────────┐
 视觉        声学         文本        词法       知识图谱
ImageBind   ImageBind   VLM/ASR     BM25      实体/关系/社区
 (vision)   (audio)     embedding   稀疏检索   + Wiki 编译层
   └──────────┴───────────┴─────┬─────┴──────────┘
                                │  多路加权融合召回
                              答案 / Top-K 片段
```

四路召回按权重融合（视觉 / 声学 / 文本 / 词法），任一通道因缺依赖不可用时自动跳过并降级，不影响其余通路。

## 目录结构

```
vidgraphrag/          # 核心纯 Python 引擎（库本体，零前端依赖）
├── _videoutil/       # 分镜 / 转写 / VLM 描述 / 多模态特征编码
├── _storage/         # 向量库 / 图库 / KV / BM25 等存储后端
├── cli.py            # headless 命令行入口
├── _server.py        # HTTP 服务层（可选）
├── videorag.py       # 引擎主类 VideoRAG
└── ...
scripts/dev/          # 开发与验证脚本
tests/                # 测试
docs/                 # 设计与架构文档
```

## 许可

本项目采用 [`LICENSE`](LICENSE) 中描述的双许可：框架架构部分为 MIT；因集成 ImageBind（NonCommercial 许可），**当前完整实现仅限非商业用途**。若替换为商业友好的模型，则框架架构部分仍按 MIT 使用。请在使用前阅读许可全文。

## 致谢

本引擎在以下开源工作之上构建，特此致谢并保留相应出处：

- **VideoRAG** — *VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos*（双层架构与视频 RAG 思路）。
- **nano-graphrag** — 核心存储与检索骨架。
- **Microsoft GraphRAG** — 部分提示词与图算法（详见 `vidgraphrag/prompt.py`、`_storage/gdb_networkx.py` 内注释）。
- **ImageBind**（Meta）— 视觉与声学统一嵌入空间。
