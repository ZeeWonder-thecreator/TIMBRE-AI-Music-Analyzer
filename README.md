# TIMBRE — 音乐制作解构引擎

> 上传一首歌，理解它的声音是如何被做出来的。
> TIMBRE helps everyday listeners understand how their favorite songs are made.

TIMBRE 是一个基于真实音频测量的音乐制作分析系统。

它并不依赖纯 AI 猜测，而是通过 DSP（数字信号处理）与音频模型，对歌曲中的声音结构进行拆解，再利用大语言模型将这些技术结果翻译成普通人能够理解的语言。

相比传统“AI 音乐分析工具”容易出现的幻觉与不可信描述，TIMBRE 更强调：

- 可解释性
- 可验证性
- 真实物理测量
- 面向普通用户的理解体验

---

# 为什么做 TIMBRE

很多人都有过这种体验：

> 听到一首歌里的某个声音特别惊艳，  
> 但完全不知道它是什么、怎么做出来的，甚至不知道应该搜索什么关键词。

现有工具通常只有两个方向：

- 专业 DAW / 音频软件（门槛过高）
- 纯 AI 生成描述（可信度低）

TIMBRE 的目标，是在两者之间建立桥梁。

它尝试把复杂的音乐制作信息，转化成普通听众也能理解的解释：

- 这首歌用了什么乐器
- 哪些是合成器音色
- 混响和 Delay 如何影响氛围
- 为什么这首歌会有这种情绪感

---

# 核心理念

## 先做真实测量，再让 AI 解释

TIMBRE 会明确区分两个层级：

| 层级 | 作用 |
|---|---|
| DSP / 音频模型 | 测量真实声音特征 |
| LLM | 将技术数据翻译成人类语言 |

这是整个项目最重要的架构决策。

早期版本曾尝试直接让 AI 分析音频，但实际测试发现：

- AI 会幻觉不存在的声音
- 描述无法验证
- 结果不稳定

因此 TIMBRE 最终采用：

> “物理测量 + AI 解读” 分离架构。

也就是说：

- BPM 来自真实节奏检测
- Attack 来自瞬态测量
- Reverb 来自衰减分析
- Delay 来自自相关检测
- Tremolo / LFO 来自 FFT 调制分析

AI 不会凭空生成底层数据。

---

# 核心功能

## 多音轨分离

基于 Demucs 对上传音频进行 Stem Separation：

- 人声
- 贝斯
- 鼓组
- 吉他
- 钢琴
- 合成器 / Other

---

## 基于 DSP 的真实音色分析

每个轨道都会进行独立声音分析。

---

## 音色类型识别

根据 Attack（起音时间）判断声音类型：

- Pad
- Lead
- Pluck

不是 AI 分类，而是通过真实瞬态特征测量实现。

---

## 混响检测

通过衰减时间分析空间效果：

- 干声
- 短混响
- 长混响

---

## Delay 检测

通过 RMS 包络自相关检测重复回声结构。

---

## Tremolo / LFO 检测

对 RMS 包络进行 FFT 分析，检测低频调制行为。

---

## 音符提取

使用 Spotify Basic Pitch 提取主要音高。

---

## BPM 检测

混合 BPM 检测 Pipeline：

1. madmom Beat Tracking
2. librosa tempogram fallback

用于降低现代流行 / K-pop 常见的 half-time BPM 歧义。

---

## 人类可读的制作报告

将复杂音频特征转化为：

- 普通听众能理解
- 音乐初学者能学习
- 制作人能参考

的自然语言解释。

---

## 内置学习系统

分析报告中的专业术语会在首次出现时自动高亮标注。

用户点击即可展开详细释义卡片，无需离开页面。

当前已覆盖 20+ 核心音乐制作术语，包括：

- 效果器类（混响、Delay、Ping-Pong Delay、门限混响、预延迟等）
- 合成器类（振荡器、滤波器、锯齿波、包络、LFO 等）
- 混音类（压缩器、侧链、EQ、失真等）
- 基础概念（BPM、Attack、干声、谐波、泛音、频谱等）

同时提供独立的 Glossary 词典页面，支持通过释义卡片一键跳转。

TIMBRE 不只是分析工具。

它也是一个降低音乐制作门槛的学习系统。

---

# 技术架构

## 分析 Pipeline

```text
音频上传
    ↓
Demucs 音轨分离
    ↓
各轨 DSP 分析
    ├── Attack 检测
    ├── 频谱分析
    ├── 混响分析
    ├── Delay 检测
    ├── 调制检测
    ├── 音符提取
    ↓
BPM 检测
    ↓
结构化声音特征
    ↓
LLM 解读层
    ↓
自然语言制作分析报告
```

---

# 工程亮点

## 流式分析系统（SSE）

TIMBRE 使用 Server-Sent Events 实现实时分析反馈。

用户在分析过程中可以实时看到：

- 音轨分离进度
- BPM 检测状态
- 效果分析状态
- 制作建议生成状态

避免传统长时间推理任务中的“页面卡死感”。

---

## 并行轨道分析

所有 Stem 分析通过：

```python
ThreadPoolExecutor
```

并行执行。

包括：

- onset 分析
- pitch 提取
- effects 检测
- modulation 检测

显著降低整体分析耗时。

---

## 高鲁棒性 Attack 检测

TIMBRE 并未使用简单峰值检测。

当前算法包含：

- onset backtracking
- 局部 RMS 峰值估计
- 80% rise-threshold
- percentile robust aggregation

用于降低：

- Demucs bleed
- 噪声瞬态
- 分轨污染

对结果的影响。

---

## DSP 导向的效果器分析

### Delay

通过：

- RMS 自相关
- 周期峰值检测

识别回声结构。

### Tremolo / LFO

通过：

- FFT
- 低频调制能量分析

检测 Tremolo / 调制行为。

整个系统优先采用物理测量，而不是 AI 分类。

---

# 技术栈

| 模块 | 技术 |
|---|---|
| 音轨分离 | Demucs htdemucs_6s |
| 音高提取 | Spotify Basic Pitch |
| BPM 检测 | madmom + librosa |
| DSP 分析 | librosa + NumPy |
| AI 解读 | DeepSeek API |
| 后端 | Flask |
| 流式通信 | SSE |
| 并发 | concurrent.futures |
| 前端 | HTML / CSS / JavaScript |

---

# 产品定位

| 产品 | 核心目标 |
|---|---|
| Shazam | 识别歌曲 |
| Sonoteller | AI 标签分类 |
| Audio Jam | 扒谱练习 |
| TIMBRE | 音乐制作解构 |

TIMBRE 不负责告诉你：

> “这首歌叫什么。”

它试图回答的是：

> “为什么它会听起来像这样。”

---

# 当前技术边界

## 无法识别具体合成器型号

目前系统无法识别：

- Serum
- Vital
- Omnisphere

等具体合成器品牌与预设。

这需要专门训练的监督分类模型，是未来迭代方向。

---

## 分轨泄漏问题

Demucs 无法做到绝对隔离。

不同轨道间的 bleed 是当前 Source Separation 模型共有上限。

---

## Half-Time BPM 歧义

现代流行 / K-pop 常存在 half-time 节奏感知问题。

机器检测到的 BPM 与人类音乐标注 BPM 可能存在 2 倍关系歧义。

---

# 未来方向

## 近期

- 用户反馈系统
- 更好的结果页可视化
- 更快的推理 Pipeline
- 更多效果器识别

---

## 中期

- 用户修正数据收集
- 基于反馈的数据校准
- 更精细的 modulation 分类

---

## 长期

- 自研音乐效果器识别模型
- 合成器架构分类
- 参数级声音复原
- 风格迁移辅助声音重建

---

# 安装方式

## 克隆仓库

```bash
git clone https://github.com/yourname/timbre.git
cd timbre
```

---

## 安装依赖

```bash
pip install flask librosa numpy torch demucs basic-pitch openai soundfile markdown madmom
```

---

## 环境变量

创建 `.env`：

```env
DEEPSEEK_API_KEY=your_api_key
```

---

## 运行

```bash
python app.py
```

---

# 支持格式

- MP3
- WAV
- FLAC
- AAC

---

# 项目结构

```text
TIMBRE/
├── app.py
├── templates/
│   ├── index.html
│   ├── loading.html
│   ├── result.html
│   └── glossary.html
├── static/
│   └── stems/
└── README.md
```

---

# 项目背景

TIMBRE 由一名非计算机专业背景的开发者在约两个月内，借助 AI 辅助工具独立完成。

开发者从 Python for 循环阶段开始自学，逐步完成整个多模型音乐分析 Pipeline。

项目过程中经历了多次架构重构：

- 从全 AI 分析
- 到 DSP + AI 混合系统
- 从简单 transient peak
- 到 robust onset-rise 分析
- 从外链词典
- 到内置悬浮学习系统

每一次调整，都来自真实测试中发现的问题。

---

# 愿景

TIMBRE 最终想做的事情，是降低音乐制作知识的理解门槛。

不是把音乐变简单。

而是让那些原本隐藏在声音背后的制作逻辑，第一次真正能被普通人看见。

---

*TIMBRE — 听见声音背后的设计。*

