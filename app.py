import os
import re
import torch
import soundfile as sf
import numpy as np
from flask import Flask, request, render_template, Response, stream_with_context, redirect, url_for
import librosa
from openai import OpenAI
import markdown
from demucs.pretrained import get_model
from demucs.apply import apply_model
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import concurrent.futures
import json

client = OpenAI(
  api_key=os.environ.get("DEEPSEEK_API_KEY"),
  base_url="https://api.deepseek.com"
)

app = Flask(__name__)
results = {}  # 临时存储分析结果，key=filename

print("加载分轨模型...")
model = get_model('htdemucs_6s')
model.eval()

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

TRACK_ZH = {
    'vocals': '人声', 'bass': '贝斯', 'drums': '鼓组',
    'guitar': '吉他', 'piano': '钢琴', 'other': '合成器 / 电子音色',
}
ATTACK_ZH = {
    'pluck': '起音干脆，弹拨 / 打击类音色',
    'lead':  '起音较快，合成铅音类型',
    'pad':   '起音缓慢，铺底型音色（Pad）',
}
REVERB_ZH = {
    'dry':          '干声，无空间混响',
    'short_reverb': '带短混响，有轻微空间感',
    'long_reverb':  '带长混响，有明显残响尾音',
}
TERMS_DB = {
    'BPM': (
        '每分钟节拍数，是衡量音乐速度的基本单位。120 BPM 相当于每秒两拍，是流行音乐最常见的速度范围。',
        'BPM（Beats Per Minute）决定一首歌的律动快慢，所有乐器和效果器的时间参数都需要与其对齐才能产生和谐律动感。怎么听：用脚踩拍，数一分钟踩多少下。常见范围：慢歌 60–90，流行/R&B 90–130，EDM 128–160。BPM 越快整体能量感越强，编曲空间越紧凑。',
        'BPM',
    ),
    'Attack': (
        '声音从起始到达到最大响度所需的时间。Attack 短，声音起头干脆；Attack 长，声音缓缓淡入。',
        'Attack 是 ADSR 包络的第一阶段，描述声音"冲出来"的速度。调节 Attack 可改变音色的侵略感：短 Attack 带来冲击力（如鼓、拨弦），长 Attack 让声音柔和渐入（如弦乐、Pad）。怎么听：判断声音开头是"噗的一下"还是"慢慢浮上来"。参数范围：合成器和压缩器的 Attack 通常在 0ms–500ms 可调。',
        'ADSR',
    ),
    'Pad': (
        '合成器中一种起音慢、持续时间长的铺底音色。常用于在歌曲背景中营造氛围、填充空间感。',
        'Pad 是合成器音色分类之一，特征是 Attack 极长（通常超过 100ms）、持续饱满、衰减缓慢，像一层"声音雾气"悬浮在歌曲中。为什么用：Pad 撑起背景和声框架，让编曲更丰满，避免过于干涩。怎么听：没有明显起音冲击的持续声音层，通常比主旋律低 6–12dB。常加混响和合唱效果，是流行、电子、R&B 编曲中最常见的背景层。',
        'pad合成器',
    ),
    'Lead': (
        '合成器中承担主旋律的音色，起音较快、音色明亮、穿透力强。是听众注意力最集中的声部。',
        'Lead 合成器负责演奏旋律线条，位于混音最突出的位置。为什么用：相比真实乐器，合成器 Lead 能创造更具未来感的旋律音色，也可模拟小提琴、笛子等。怎么听：跟着旋律哼唱，你在哼的那条线通常就是 Lead 声部。参数：Attack 短（10–50ms），Filter 截止频率高以增加亮度，常加少量 Reverb 但不过多以免模糊旋律。Lead 的音色选择直接决定歌曲的风格气质。',
        '主音合成器',
    ),
    'Pluck': (
        '弹拨类音色，起音瞬间响亮、随后快速衰减，如吉他拨弦或竖琴。节奏感强，适合强调律动。',
        'Pluck 来源于真实弹拨乐器的物理特性：弦被拨动时产生瞬间爆发，振动迅速消散。为什么用：节奏感强，适合强调律动、装饰性旋律线或分解和弦。怎么听：听到"噗"或"叮"的一下然后迅速消失就是 Pluck 类音色。参数：Attack 极短（通常 <10ms），Decay 短，Sustain 低，可用 Filter 调节音色明暗。常见于 lo-fi、tropical house、K-pop 编曲的律动层。',
        '拨弦音色',
    ),
    '混响': (
        '模拟声音在空间中反射产生的残响效果，让声音有房间感或大厅感。衰减时间越长，空间越大。',
        '混响（Reverb）通过叠加大量快速反射波来模拟不同大小的空间。为什么用：干声听起来像贴耳录音，加混响能把各声部"放进"同一虚拟空间，让编曲更自然统一。怎么听：声音结束后听是否有"尾巴"——短尾是小房间，长尾是大厅或空旷空间。参数：Decay 0.2s–5s+，Pre-delay 控制距离感，Wet/Dry 控制混响量。现代流行音乐人声通常使用短混响（0.8–1.5s）以保持清晰度。',
        '混响',
    ),
    'Delay': (
        '将原始声音延迟后重复播放，产生回声叠加效果。可以是一次回声，也可以是多次逐渐消失的重复。',
        'Delay 通过将音频信号延迟一段时间后叠加回原声产生回声效果。为什么用：既能增加空间感，当 Delay 时间与 BPM 对齐时，回声会卡在拍子上形成丰富的律动层次。怎么听：唱完一句歌词后听是否有"影子"在重复，或在安静段落听到渐弱的回声。参数：Delay Time（常以几分音符设定），Feedback（重复次数），Wet/Dry。Slapback Delay（30–100ms 单次）让声音更厚实，过多 Feedback 会使声音混乱。',
        'delay效果器',
    ),
    'Tremolo': (
        '通过周期性改变音量产生"颤动"效果的调制类效果器，听起来像音量在规律地忽大忽小。',
        'Tremolo 是振幅调制效果，使用 LFO 周期性地改变信号音量。为什么用：产生独特的律动感和有机质感，可以是细腻的"气息感"也可以是明显的"机械颤动"，常用于吉他、合成器 Pad 和人声效果。怎么听：某个持续的声音如果音量在有规律地"起伏"就是 Tremolo 效果。参数：Rate（LFO 频率，1–10Hz 常见），Depth（控制音量变化幅度）。低速 Tremolo（1–3Hz）增加有机律动感，高速（6Hz+）产生类似颤音的紧张感。',
        'tremolo效果器',
    ),
    'LFO': (
        '低频振荡器，合成器中产生缓慢周期性变化信号的模块，频率通常在 0.1–20Hz。常用来驱动 Tremolo、Vibrato 等调制效果。',
        'LFO（Low Frequency Oscillator）本身不发声，而是作为控制信号周期性地改变其他参数（音量、音高、滤波器截止频率等）。为什么用：使静态的合成器参数"活起来"，产生自然的律动和变化感，是电子音乐中表现生命力的关键工具。怎么听：任何周期性的声音变化（忽大忽小、忽高忽低、忽亮忽暗）背后通常都有 LFO 在驱动。参数：Rate（频率），Waveform（正弦波最平滑，方波最突兀），Depth（调制深度）。LFO 与 BPM 同步时产生强烈律动感，不同步时更自由有机。',
        'LFO低频振荡器',
    ),
    '合成器': (
        '用电子信号合成声音的乐器，能模拟真实乐器或创造自然界不存在的声音。是流行、电子音乐编曲的核心工具。',
        '合成器（Synthesizer）通过振荡器产生原始波形，再经滤波器、放大器和调制模块塑造声音特征。为什么用：声音参数完全可控，能制作任何风格所需的音色，从模拟钢琴到科幻音效均可实现，且便于录制和重现。怎么听：电子感强、非常"纯净"或非常"奇特"的声音通常来自合成器，真实乐器常带有自然泛音和演奏噪声。常见类型：减法合成（最经典）、FM 合成（金属感、铃声）、采样合成。合成器的音色选择直接定义一首歌的风格，是区分流行、lo-fi、EDM 等风格的核心元素。',
        '合成器',
    ),
    '谐波': (
        '基频的整数倍频率成分，决定音色的明暗和丰富度。谐波越多，声音越亮越复杂。',
        '谐波（Harmonics）是声音频谱中基频之上的整数倍频率分量。为什么重要：同一个音高用不同乐器演奏听起来不同，就是因为谐波结构不同。怎么听：同一个音，钢琴比笛子听起来更"丰满"，是因为钢琴的高次谐波更丰富。应用：合成器通过调节滤波器截止频率来增减谐波，从而改变音色明暗；失真效果器通过产生额外谐波让声音更"脏"更有攻击性。偶次谐波听感温暖，奇次谐波听感尖锐。',
        '谐波 harmonics',
    ),
    '干声': (
        '未经任何空间效果（混响、Delay 等）处理的原始声音信号，听起来贴耳、直接、无空间感。',
        '干声（Dry Signal）指完全没有经过混响、延迟等空间效果处理的纯净音频信号。为什么重要：干声是混音的起点，所有空间效果都是在干声基础上叠加的；干声越干净，后期处理空间越大。怎么听：声音像直接贴在耳边说话，没有任何"尾巴"或回声，就是干声。应用：录音时通常录制干声，后期再加效果；混音中 Dry/Wet 旋钮控制干声与效果声的比例。人声录音追求干净的干声，以便后期灵活处理。',
        '干声 dry signal',
    ),
    '泛音': (
        '乐器发声时伴随基频产生的高频成分，是谐波的自然表现形式，决定了每种乐器独特的音色特征。',
        '泛音（Overtones）是乐器振动时自然产生的高于基频的频率成分。为什么重要：泛音结构是区分不同乐器音色的关键——同样弹一个 A4（440Hz），吉他和钢琴听起来不同就是因为泛音比例不同。怎么听：用手指轻触吉他弦的中点再拨弦，听到的清脆高音就是泛音。应用：合成器通过叠加不同比例的泛音来模拟真实乐器；EQ 调节本质上就是在增减特定频段的泛音。真实乐器的泛音丰富且不规则，合成器的泛音精确且可控。',
        '泛音 overtones',
    ),
    '频谱': (
        '声音中各频率成分的分布图，横轴是频率（低音到高音），纵轴是强度。是分析音色的核心工具。',
        '频谱（Spectrum）将声音分解为各个频率分量并显示其强度分布。为什么重要：频谱让你"看见"声音的构成——哪些频段突出、哪些缺失，是混音和音色设计的基础参考。怎么听：低频多的声音听起来"厚重"，高频多的听起来"明亮"或"刺耳"。应用：EQ 均衡器就是在频谱上做加减法；频谱分析仪是混音师判断频率冲突的主要工具。一首混音良好的歌，频谱分布应该均匀平滑，没有某个频段过度突出。',
        '频谱分析',
    ),
    '门限混响': (
        '一种混响尾音被突然截断的特殊混响效果，声音先爆发再戛然而止，极具冲击力。80年代标志性鼓声效果。',
        '门限混响（Gated Reverb）是将混响信号通过噪声门处理，当混响音量降到门限以下时立即静音，产生"爆发→突然消失"的效果。为什么用：普通混响尾音会模糊节奏感，门限混响保留了混响的爆发力但不拖泥带水，让鼓声既有空间感又干脆有力。怎么听：鼓声听起来很"大"但没有拖尾，像在一个大房间里但声音被突然切断。经典案例：Phil Collins "In the Air Tonight" 的鼓声。常用于军鼓和 Tom 鼓，是 80s 流行和现代 synthwave 的标志音色。',
        '门限混响 gated reverb',
    ),
    '预延迟': (
        '混响效果中，原始声音结束到混响开始之间的时间间隔。用来制造声源与反射墙壁之间的距离感。',
        '预延迟（Pre-delay）控制干声与混响之间的时间差。为什么用：没有预延迟时，混响会"糊住"原声，让声音变得模糊；加入适当预延迟，人耳能先听清原声再感受空间，既保持清晰度又有空间感。怎么听：人声是否听起来"在空间里但很清晰"——如果是，通常有 20–60ms 的预延迟。参数范围：10–100ms，越长距离感越远。流行人声常用 30–50ms，让人声"浮在"混响前面而不被淹没。',
        '预延迟 pre-delay',
    ),
    'Ping-Pong Delay': (
        '一种左右声道交替出现回声的延迟效果，声音像乒乓球一样在左右耳之间弹跳，增加立体感和空间宽度。',
        'Ping-Pong Delay 将回声信号交替发送到左右声道，产生声音在两耳之间"弹跳"的效果。为什么用：普通 Delay 的回声在中间，听感平面；Ping-Pong 让回声在立体声场中移动，极大增加宽度和趣味性，适合填充编曲空间。怎么听：戴耳机听，回声是否在左右耳交替出现。常用于合成器旋律、人声尾音、吉他 Solo。参数：Delay Time 通常与 BPM 同步（如 1/8 音符），Feedback 控制弹跳次数，Spread 控制左右分离程度。',
        'ping-pong delay',
    ),
    '滤波器': (
        '只允许特定频率范围通过、削减其他频率的音频处理工具。是合成器音色塑造和混音 EQ 的核心组件。',
        '滤波器（Filter）通过削减特定频段来改变声音的音色特征。为什么用：原始振荡器波形通常太亮太刺耳，滤波器削掉多余高频让声音变得柔和可控，是减法合成的核心步骤。怎么听：声音从"明亮刺耳"变"闷暗柔和"的过程就是低通滤波器在工作。常见类型：低通（削高频，最常用）、高通（削低频）、带通（只留中间频段）。参数：Cutoff（截止频率）、Resonance（共振，在截止点产生尖锐峰值）。合成器中用 LFO 调制 Cutoff 可产生经典的"哇哇"效果。',
        '滤波器 filter',
    ),
    '锯齿波': (
        '合成器基础波形之一，含有全部谐波成分，音色明亮饱满。是制作 Lead、Pad、Bass 最常用的起始波形。',
        '锯齿波（Sawtooth Wave）包含基频及所有整数倍谐波，且谐波强度按 1/n 递减，因此音色最丰富最明亮。为什么用：因为谐波最全，用滤波器削减后可以塑造出几乎任何音色，是合成器音色设计的万能起点。怎么听：未经处理的锯齿波听起来像"嗡嗡"的蜂鸣声，带有明显的"锋利感"。应用：叠加多个略微失谐的锯齿波产生 Supersaw（经典 Trance/EDM Lead 音色）；低通滤波后变成温暖的 Pad 或 Bass。与方波（只含奇次谐波，更空洞）形成对比。',
        '锯齿波 sawtooth',
    ),
    '包络': (
        '描述声音从出现到消失的音量变化曲线，通常用 ADSR 四个阶段表示：起音、衰减、持续、释放。',
        '包络（Envelope / ADSR）定义声音随时间的变化轨迹。A（Attack）= 起音时间，D（Decay）= 到达持续音量的衰减时间，S（Sustain）= 按住键时的持续音量，R（Release）= 松键后消失的时间。为什么用：包络决定声音的"形状"——同一个波形，短包络是 Pluck，长包络是 Pad。怎么听：声音是"叮"一下就没（短 Decay + 低 Sustain）还是一直持续（高 Sustain）。应用：不仅控制音量，还可控制滤波器（音色随时间变亮变暗）、音高（弯音效果）等任何参数。',
        '包络 envelope ADSR',
    ),
    '压缩器': (
        '自动减小声音动态范围的处理器——让最响的部分变小，使整体音量更均匀、更有力。混音中最重要的工具之一。',
        '压缩器（Compressor）当信号超过设定门限时自动降低增益，缩小响与轻之间的差距。为什么用：未压缩的人声忽大忽小难以在混音中稳定存在，压缩后音量一致性提高，听感更专业更有力。怎么听：人声每个字的音量都很均匀、鼓声听起来很"紧实有力"通常是压缩的结果。参数：Threshold（门限）、Ratio（压缩比）、Attack、Release。轻压缩（2:1–4:1）让声音自然稳定，重压缩（10:1+）产生明显的"泵感"效果。侧链压缩让贝斯随底鼓"呼吸"是 EDM 标志性技巧。',
        '压缩器 compressor',
    ),
    '侧链': (
        '用一个信号（通常是底鼓）去控制另一个信号（通常是贝斯或 Pad）的音量，产生节奏性的"呼吸"起伏效果。',
        '侧链（Sidechain）是一种路由技巧，让压缩器不根据自身信号而是根据外部信号来触发压缩。为什么用：当底鼓响起时自动压低贝斯/Pad 的音量，既避免低频打架，又产生标志性的"泵感"律动。怎么听：贝斯或背景 Pad 的音量随着底鼓节奏在有规律地"一起一伏"。应用：EDM/House 中几乎必用；也可用于人声（让伴奏在人声出现时自动让位）。参数：通过压缩器的 Attack 和 Release 控制"泵"的形状——快 Release 产生短促弹跳，慢 Release 产生平滑起伏。',
        '侧链 sidechain',
    ),
    'EQ': (
        '均衡器，用来增强或削减特定频段的音量，调整声音的明暗、厚薄。混音中用来让各乐器互不打架。',
        'EQ（Equalizer）是频率层面的音量控制器，可以精确地提升或削减任意频段。为什么用：每个乐器都占据一定频率范围，当多个乐器频率重叠时会互相遮蔽（"打架"），EQ 通过给每个乐器"让出空间"来保持混音清晰。怎么听：人声听起来"闷"是因为高频被削了，听起来"薄"是因为中低频被削了。常见操作：高通滤波去掉不需要的低频隆隆声、提升 3–5kHz 增加人声存在感、削减 200–400Hz 减少"浑浊感"。参数型 EQ 可调频率、增益、Q 值（带宽）。',
        'EQ 均衡器',
    ),
    '失真': (
        '故意让信号过载产生额外谐波的效果，让声音变得粗糙、有攻击性。从轻微温暖到极端暴力都可调节。',
        '失真（Distortion）通过将信号推过放大器或算法的承受极限，产生削波和额外谐波。为什么用：适度失真增加温暖感和存在感（如电子管过载），重度失真带来攻击性和能量感（如电吉他失真）。怎么听：声音是否有"毛刺感""沙沙声"或"撕裂感"。类型：Overdrive（轻微过载，温暖）、Distortion（中度，摇滚吉他）、Fuzz（极端，模糊厚重）、Bitcrusher（数字失真，lo-fi 质感）。在电子音乐中常用于贝斯增加存在感，或用于人声制造特殊效果。',
        '失真 distortion',
    ),
    '振荡器': (
        '合成器中产生原始声音波形的核心模块，是所有合成音色的起点。不同波形（锯齿、方波、正弦等）决定基础音色。',
        '振荡器（Oscillator / OSC）是合成器的"声带"，产生周期性电信号作为声音的原始素材。为什么用：振荡器决定音色的基础特征，后续所有处理（滤波、包络、效果）都建立在振荡器波形之上。常见波形：正弦波（纯净，只有基频）、锯齿波（明亮，全谐波）、方波（空洞，奇次谐波）、噪声（无音高，用于打击和特效）。现代合成器通常有 2–3 个振荡器可叠加，通过微调音高差（Detune）产生厚实感。Wavetable 合成器可在不同波形间平滑过渡。',
        '振荡器 oscillator',
    ),
}
MODULATION_TRACKS = {'other', 'piano'}

def extract_notes(stem_path):
    _, _, note_events = predict(stem_path)
    if not note_events:
        return []
    counts = {}
    for event in note_events:
        midi_pitch = int(event[2])
        note = NOTE_NAMES[midi_pitch % 12]
        counts[note] = counts.get(note, 0) + 1
    sorted_notes = sorted(counts, key=lambda n: counts[n], reverse=True)
    return sorted_notes[:3]


def analyze_attack(stem_path):
    """
        基于 RMS 包络分析计算音色 Attack 时间。
        使用 onset 检测 + 25th 百分位抗噪方案。
        核心实现不在此处展示。
        """
    pass


def analyze_effects(stem_path):
    """
    检测音轨效果器处理特征。
    包含混响衰减测量、自相关 Delay 检测、LFO 调制频谱分析。
    核心实现不在此处展示。
    """
    pass


def separate_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=model.samplerate, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])

    wav = torch.tensor(y, dtype=torch.float32)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()

    with torch.no_grad():
        sources = apply_model(model, wav[None], device='cpu', shifts=0, split=True, overlap=0.1)[0]

    sources = sources * ref.std() + ref.mean()

    os.makedirs('static/stems', exist_ok=True)
    stems = {}
    per_stem = {}  # 先存路径和基础数据，后面并行分析

    for i, name in enumerate(model.sources):
        path = f"static/stems/{name}.wav"
        audio_data = sources[i].T.numpy()
        sf.write(path, audio_data, model.samplerate)
        stems[name] = name + '.wav'

        y_stem = audio_data.mean(axis=1)
        rms = float(np.sqrt(np.mean(y_stem ** 2)))
        centroid = float(librosa.feature.spectral_centroid(y=y_stem, sr=model.samplerate).mean())
        flatness = float(librosa.feature.spectral_flatness(y=y_stem).mean())
        per_stem[name] = {'path': path, 'rms': rms, 'centroid': centroid, 'flatness': flatness}

    def analyze_stem(name):
        d = per_stem[name]
        path, rms = d['path'], d['rms']
        if name == 'drums':
            attack_info = {'attack_ms': 5.0, 'sound_type': 'pluck'}
            effects_info = analyze_effects(path) if rms > 0.01 else {}
            effects_info['delay'] = False
            effects_info['modulation'] = None
        elif name == 'bass':
            attack_info = analyze_attack(path) if rms > 0.01 else {}
            effects_info = analyze_effects(path) if rms > 0.01 else {}
            effects_info['delay'] = False
        else:
            attack_info = analyze_attack(path) if rms > 0.01 else {}
            effects_info = analyze_effects(path) if rms > 0.01 else {}
        notes = extract_notes(path) if rms > 0.01 and name not in ('vocals', 'drums') else []
        return name, {
            'rms': round(rms, 4),
            'centroid': round(d['centroid'], 1),
            'flatness': round(d['flatness'], 6),
            'notes': notes,
            'attack_ms': attack_info.get('attack_ms'),
            'sound_type': attack_info.get('sound_type'),
            'reverb': effects_info.get('reverb'),
            'delay': effects_info.get('delay'),
            'modulation': effects_info.get('modulation'),
        }

    stem_analysis = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for name, result in executor.map(analyze_stem, model.sources):
            stem_analysis[name] = result

    return stems, stem_analysis


def describe_track_sec1(name, data):
    parts = []
    centroid = data.get('centroid', 0)
    flatness = data.get('flatness', 0)
    sound_type = data.get('sound_type')
    reverb = data.get('reverb')
    attack_ms = data.get('attack_ms')

    if name == 'vocals':
        if centroid > 5000:
            parts.append('音色非常明亮，高频突出')
        elif centroid > 3500:
            parts.append('音色偏亮')
        elif centroid > 2500:
            parts.append('音色中性偏亮')
        elif centroid > 2000:
            parts.append('音色中性')
        else:
            parts.append('音色偏暗，低频感强')
        if reverb == 'dry':
            parts.append('干声处理，贴耳感强')
        elif reverb == 'short_reverb':
            parts.append('加了短混响，有轻微空间感')
        elif reverb == 'long_reverb':
            parts.append('加了长混响，空间感明显')
        if data.get('delay'):
            parts.append('有 Delay 回声')
        if data.get('modulation'):
            parts.append('有调制效果')

    elif name == 'bass':
        if centroid < 300:
            parts.append('极低频为主，Sub Bass / 808特征')
        elif centroid < 500 and flatness < 0.05:
            parts.append('低频浑厚，偏真实拨弦贝斯')
        elif centroid < 500 and flatness >= 0.05:
            parts.append('低中频为主，偏合成器贝斯（808风格）')
        elif centroid < 700:
            parts.append('低中频为主，合成器贝斯可能性较高')
        elif centroid < 1200:
            parts.append('谐波成分较多，偏合成器贝斯')
        else:
            parts.append('高频谐波丰富，失真或 Reese Bass 特征')
        if attack_ms is not None:
            if attack_ms < 10:
                parts.append('起音极快，弹拨感强')
            elif attack_ms < 30:
                parts.append('起音干脆')
            elif attack_ms > 80:
                parts.append('起音缓慢，滑入感')
        if reverb == 'short_reverb':
            parts.append('轻微空间感')
        elif reverb == 'long_reverb':
            parts.append('带混响处理')
        else:
            parts.append('干声')

    elif name == 'drums':
        if centroid > 4000:
            parts.append('整体偏亮，镲片/高频打击突出')
        elif centroid > 2500:
            parts.append('高低频均衡')
        else:
            parts.append('偏低频，底鼓为主')
        if reverb == 'dry':
            parts.append('干声，无空间处理')
        elif reverb == 'short_reverb':
            parts.append('带短混响，有房间感')
        elif reverb == 'long_reverb':
            parts.append('带大量混响，有明显空间感')


    elif name == 'guitar':
        flatness = data.get('flatness', 0)
        if centroid > 3500 or (centroid > 2000 and flatness > 0.12):
            parts.append('高频能量集中，疑似失真电吉他')
        elif centroid > 2000:
            parts.append('音色偏亮，偏电吉他质感')
        elif centroid > 1200:
            parts.append('音色中性，真实吉他特征')
        else:
            parts.append('音色温暖，偏木吉他质感')
        if sound_type == 'pluck':
            parts.append('弹拨感明显')
        elif sound_type == 'lead':
            parts.append('持续音为主，可能是滑弦或延音奏法')
        elif sound_type == 'pad':
            parts.append('持续音，可能有大量延音处理')
        if reverb == 'short_reverb':
            parts.append('带短混响')
        elif reverb == 'long_reverb':
            parts.append('带长混响')
        if data.get('delay'):
            parts.append('有 Delay')

    elif name == 'piano':
        if centroid > 3500:
            parts.append('音色明亮，高音区为主')
        elif centroid > 2500:
            parts.append('音色偏亮')
        elif centroid > 1200:
            parts.append('音色中性')
        else:
            parts.append('音色低沉，低音区为主')
        if reverb == 'short_reverb':
            parts.append('带短混响')
        elif reverb == 'long_reverb':
            parts.append('带长混响，空间感强')
        elif reverb == 'dry':
            parts.append('干声')
        if data.get('delay'):
            parts.append('有 Delay 处理')

    elif name == 'other':
        if centroid > 5000:
            parts.append('音色非常明亮，高频能量集中')
        elif centroid > 4000:
            parts.append('音色明亮')
        elif centroid > 2500:
            parts.append('音色中性偏亮')
        elif centroid > 1500:
            parts.append('音色中性偏暗')
        else:
            parts.append('音色低沉温暖')
        type_zh = {
            'pad':   '起音缓慢，铺底型音色（Pad）',
            'lead':  '起音较快，主旋律型音色（Lead）',
            'pluck': '起音干脆，弹拨型音色（Pluck）',
        }
        if sound_type in type_zh:
            parts.append(type_zh[sound_type])
        if attack_ms is not None and sound_type == 'pluck':
            if attack_ms < 5:
                parts.append('瞬态极短，打击感强')
        if reverb == 'short_reverb':
            parts.append('带短混响')
        elif reverb == 'long_reverb':
            parts.append('带长混响')
        if data.get('delay'):
            parts.append('有 Delay')
        if data.get('modulation'):
            parts.append('有调制效果')

    return '，'.join(parts) if parts else '信号较弱，特征不明显'


def build_report(stem_analysis, tempo):
    used_terms = set()
    active = {k: v for k, v in stem_analysis.items() if v['rms'] > 0.01}

    # --- 第一段：乐器使用 ---
    sorted_tracks = sorted(active.items(), key=lambda x: x[1]['rms'], reverse=True)
    main_tracks = [(k, v) for k, v in sorted_tracks if v['rms'] >= 0.05]
    side_tracks = [(k, v) for k, v in sorted_tracks if 0.01 < v['rms'] < 0.05]

    sec1 = '## 这首歌用了哪些乐器\n\n'
    for section_label, track_list in [('主要声部：\n\n', main_tracks), ('辅助声部（音量较低）：\n\n', side_tracks)]:
        if not track_list:
            continue
        sec1 += section_label
        for name, data in track_list:
            zh = TRACK_ZH[name]
            if name == 'other':
                used_terms.add('合成器')
            sound_type = data.get('sound_type')
            if sound_type:
                used_terms.add('Attack')
                if sound_type == 'pad':
                    used_terms.add('Pad')
                elif sound_type == 'lead':
                    used_terms.add('Lead')
                elif sound_type == 'pluck' and name != 'drums':
                    used_terms.add('Pluck')
            if data.get('reverb') and data['reverb'] != 'dry':
                used_terms.add('混响')
            if data.get('delay'):
                used_terms.add('Delay')
            if data.get('modulation') and name in MODULATION_TRACKS:
                used_terms.add('Tremolo')
                used_terms.add('LFO')
            desc = describe_track_sec1(name, data)
            sec1 += f'**{zh}**：{desc}。\n\n'
            sec1 += '\n> 注：弦乐、铜管等管弦乐器暂不在识别范围内，如有此类乐器可能显示为合成器/电子音色。\n'

    # --- 第二段：音高与节奏 ---
    used_terms.add('BPM')
    sec2 = '## 音高与节奏\n\n'
    sec2 += f'整首歌速度为 {round(float(tempo), 1)} BPM，属于{"快速" if tempo > 130 else "中速" if tempo > 90 else "慢速"}节奏。\n\n'

    for name, data in sorted_tracks:
        if name in ('vocals', 'drums'):
            continue
        zh = TRACK_ZH[name]
        lines = []

        notes = data.get('notes', [])
        if notes:
            lines.append(f'音符：{" / ".join(notes)}')

        if name == 'other':
            attack_ms = data.get('attack_ms')
            if attack_ms is not None:
                lines.append(f'起音时间：{attack_ms}ms')

            mod = data.get('modulation')
            if mod:
                freq_str = mod.split('(')[1].replace('Hz)', '') if '(' in mod else ''
                if freq_str:
                    lines.append(f'调制效果：Tremolo，LFO 约 {freq_str}Hz')
                    used_terms.add('Tremolo')
                    used_terms.add('LFO')

        if lines:
            sec2 += f'**{zh}**：{"；".join(lines)}。\n\n'

    return sec1, sec2, used_terms


def inject_term_tags(html):
    terms_sorted = sorted(TERMS_DB.keys(), key=len, reverse=True)
    pattern = re.compile('(' + '|'.join(re.escape(t) for t in terms_sorted) + ')')
    seen = set()
    parts = re.split(r'(<strong>.*?</strong>|<[^>]+>)', html)
    for i, part in enumerate(parts):
        if not part.startswith('<'):
            def replace_first(m):
                term = m.group(1)
                if term in seen:
                    return term
                seen.add(term)
                return f'<strong>{term}</strong>'
            parts[i] = pattern.sub(replace_first, part)
    return ''.join(parts)


def detect_bpm(filename):
    tempo = None
    try:
        import collections, collections.abc
        for _attr in ('MutableSequence', 'MutableMapping', 'MutableSet', 'Callable', 'Sequence'):
            if not hasattr(collections, _attr):
                setattr(collections, _attr, getattr(collections.abc, _attr))
        if not hasattr(np, 'float'):
            np.float = float
        if not hasattr(np, 'complex'):
            np.complex = complex
        if not hasattr(np, 'int'):
            np.int = int
        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
        act = RNNBeatProcessor()(filename)
        beats = DBNBeatTrackingProcessor(fps=100)(act)
        if len(beats) > 4:
            ibi = np.diff(beats)
            tempo = 60.0 / float(np.median(ibi))
            if tempo < 70:
                # 可能是慢歌，保留原始值不强制倍速
                pass
            while tempo > 240:
                tempo /= 2
    except Exception as e:
        print(f"[BPM] madmom error: {e}")

    if tempo is None:
        y, sr = librosa.load(filename)
        hop_length = 256
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        avg_tg = tempogram.mean(axis=1)
        tempo_axis = librosa.tempo_frequencies(avg_tg.shape[0], sr=sr, hop_length=hop_length)
        for lo, hi in [(100, 185), (60, 220)]:
            mask = (tempo_axis >= lo) & (tempo_axis <= hi)
            if mask.any():
                tempo = float(tempo_axis[mask][np.argmax(avg_tg[mask])])
                break
        else:
            tempo = float(tempo_axis[np.argmax(avg_tg)])
    return tempo


def generate_sec3(stem_analysis, tempo):
    active = sorted(
        [(k, v) for k, v in stem_analysis.items() if v['rms'] > 0.01],
        key=lambda x: x[1]['rms'], reverse=True
    )
    track_desc = []
    for name, data in active:
        zh = TRACK_ZH[name]
        parts = []
        if data.get('sound_type') and data['sound_type'] in ATTACK_ZH:
            parts.append(ATTACK_ZH[data['sound_type']])
        if data.get('reverb') and data['reverb'] in REVERB_ZH:
            parts.append(REVERB_ZH[data['reverb']])
        if data.get('delay'):
            parts.append('有 Delay 效果')
        if data.get('modulation') and name in MODULATION_TRACKS:
            parts.append(f'有调制效果（{data["modulation"]}）')
        if data.get('notes'):
            parts.append(f'常用音符：{"、".join(data["notes"])}')
        if parts:
            track_desc.append(f'{zh}：{"；".join(parts)}')

    prompt = f"""你是一个音乐制作顾问。根据以下这首歌的实测声音特征，给出3-4条具体的制作方向，帮助用户复现这首歌的声音风格。

歌曲 BPM：{round(float(tempo), 1)}
各轨测量结果：
{chr(10).join(track_desc)}

输出要求：
- 只输出正文内容，不要输出任何标题行
- 每条建议单独成段，语言具体可操作
- 聚焦在如何用合成器 / 效果器复现这首歌的声音特色
- 不要出现原始数字（RMS / Hz / ms 数值一律不写）
- 不要开场白，直接给建议"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return '## 如果你想做出类似的声音\n\n' + response.choices[0].message.content.strip()


def recommend_tutorials(stem_analysis):
    keywords = set()
    for name, data in stem_analysis.items():
        if data.get('rms', 0) <= 0.01:
            continue
        if name == 'other':
            st = data.get('sound_type')
            if st == 'pad':
                keywords.add('合成器Pad音色设计')
            elif st == 'lead':
                keywords.add('合成器Lead音色')
            elif st == 'pluck':
                keywords.add('合成器Pluck音色')
            else:
                keywords.add('合成器音色设计')
        if data.get('reverb') == 'long_reverb':
            keywords.add('混响效果器教程')
        if data.get('delay'):
            keywords.add('Delay效果器使用')
        if data.get('modulation') and name in MODULATION_TRACKS:
            keywords.add('LFO调制效果')
        if name == 'bass':
            if data.get('centroid', 0) > 700:
                keywords.add('合成器贝斯音色设计')
            else:
                keywords.add('贝斯编曲')
    if not keywords:
        keywords.add('编曲混音教程')
    return [
        {'title': q, 'url': 'https://search.bilibili.com/all?keyword=' + q}
        for q in list(keywords)[:5]
    ]


def sse_event(progress, step, done=False):
    data = json.dumps({'progress': progress, 'step': step, 'done': done}, ensure_ascii=False)
    return f'data: {data}\n\n'


@app.route('/', methods=['GET', 'POST'])
def upload():
  if request.method == 'POST':
      file = request.files['audio']
      if not file.filename.lower().endswith(('.mp3', '.wav', '.aac', '.flac')):
          return '请上传音频文件'
      file.save(file.filename)
      return redirect(url_for('loading', filename=file.filename))
  return render_template('index.html')


@app.route('/loading/<filename>')
def loading(filename):
    return render_template('loading.html', filename=filename)


@app.route('/stream/<filename>')
def stream(filename):
    def generate():
        yield sse_event(5, '文件接收完成')

        yield sse_event(10, 'Demucs 音轨分离开始...')
        stems, stem_analysis = separate_audio(filename)
        yield sse_event(40, '音轨分离完成')

        tempo = detect_bpm(filename)
        yield sse_event(50, 'BPM 检测完成')

        sec1, sec2, used_terms = build_report(stem_analysis, tempo)
        yield sse_event(70, '各轨道效果分析完成')

        sec3 = generate_sec3(stem_analysis, tempo)
        yield sse_event(90, '制作建议生成完成')

        tutorials = recommend_tutorials(stem_analysis)

        full_report = sec1 + '\n' + sec2 + '\n' + sec3
        html_content = markdown.markdown(full_report)
        html_content = inject_term_tags(html_content)
        terms_json = json.dumps(
            {k: {'short': v[0], 'long': v[1]} for k, v in TERMS_DB.items()},
            ensure_ascii=False
        )
        results[filename] = {
            'content': html_content,
            'stems': stems,
            'terms_json': terms_json,
            'bilibili_results': tutorials,
        }
        yield sse_event(100, '分析完成', done=True)

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/result/<filename>')
def result(filename):
    data = results.pop(filename, None)
    if not data:
        return redirect('/')
    return render_template('result.html', **data)

@app.route('/glossary')
def glossary():
    terms = {k: v[1] for k, v in TERMS_DB.items()}
    return render_template('glossary.html', terms=terms)

app.run(debug=True, threaded=True)