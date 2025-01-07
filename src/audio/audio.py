from typing import Dict, List
import os, json
from tqdm import tqdm

from ..utils.ffmpeg import create_silence, concat
from ..utils.chat_model import ChatModel

from .lib.CosyVoice.cosyvoice.utils.file_utils import load_wav
from .tta import TTAModel

def get_wav_secs(wav_file: str):
    wav = load_wav(wav_file, 16000)
    return wav.shape[1] // 16000

def gen_audio_desc(
    dialogs: List[Dict],
    intervals: List[Dict],
    speech_source_dir: str,
    model=ChatModel,
):
    """
    生成音效描述
    Args:
        dialogs (List[Dict]): 对话列表
        intervals (List[Dict]): 对话之间的时间间隔
        speech_source_dir (str): 音频目录
        model (str): openai 模型
    Returns:
        [{
            "audio_desc": 音频描述,
            "start": 开始时间（以秒为单位）,
            "end": 结束时间（以秒为单位）,
            "explaination": 插入这段音效的解释说明
        }]
    """

    new_dialogs = []
    start_secs = 0
    for i, dialog in enumerate(dialogs):
        speech_file = os.path.join(speech_source_dir, f"{dialog['role']}_{i}.wav")
        speech = load_wav(speech_file, 16000)
        speech_secs = speech.shape[1] // 16000
        interval_secs = intervals[i]["interval"]
        new_dialogs.append(
            {
                "role": dialog["role"],
                "content": dialog["content"],
                "start": start_secs,
                "end": start_secs + speech_secs,
            }
        )
        start_secs += speech_secs + interval_secs

    audio_descs = model.generate(
        DIALOG_TO_AUDIO.format(
            dialogs=json.dumps(new_dialogs, indent=4, ensure_ascii=False)
        ),
        DIALOG_TO_AUDIO_SYSTEM,
    )
    return audio_descs


def gen_audio(audio_descs: List[Dict], output_dir: str, model: TTAModel):
    for i, audio in enumerate(tqdm(audio_descs)):
        desc = audio["audio_desc"]
        duration = min(audio["end"] - audio["start"], 3)
        output_path = os.path.join(output_dir, f"audio_{i}.wav")
        if os.path.exists(output_path):
            continue
        model.generate(
            desc,
            duration,
            output_path,
        )

    audio_files = []
    fade_durations = []
    end = 0
    last_duration = 100
    for j, audio in enumerate(audio_descs):
        if audio["start"] > end:
            silence_file = os.path.join(
                output_dir, f"silence_{audio['start'] - end}.wav"
            )
            create_silence(audio["start"] - end, silence_file)
            audio_files.append(silence_file)
            fade_durations.append(min(2, min(last_duration, audio["start"] - end)))
            last_duration = audio["start"] - end
        audio_file = os.path.join(output_dir, f"audio_{j}.wav")
        audio_files.append(audio_file)
        duration = min(3, audio["end"] - audio["start"])
        fade_durations.append(min(2, min(last_duration, duration)))
        last_duration = duration
        end = audio["start"] + duration
    concat(audio_files, os.path.join(output_dir, "audio.wav"), fade_durations)


DIALOG_TO_AUDIO_SYSTEM = """
你是一个擅长给文本配音的助手，你能根据文本描述的场景描述契合的音效
"""
DIALOG_TO_AUDIO = """
请你帮我为一段对话音频生成背景音效。

这段对话将被制作成广播剧。请你帮我根据对话内容，在需要的位置生成一些因氛围和环境需要特别突出的声音，从而增加对话的沉浸感。

### 音效类型

你生成的音效
- **不应该** 包含 **任何人物的声音**，如
    - 人物的笑声
    - 呼吸声
    - 叹息声
    - ...
- **应该** 包含能表现环境变化或者突出人物动作的音效，如
    - 人群嘈杂的声音
    - 人物的脚步声
    - 天气的声音
    - 物品碰撞的声音
    - ...


### 音效时长

你应该控制好生成的音量时间。输入中会给出每句对话的开始时间和结束时间，你需要根据内容在适当的时间插入音效。

你应该注意有些音效是持续的，比如环境声音，请设置少于`3s`

有些音效是瞬间的，比如动作的声音，请设置为`1s`。

请设置好每段音效的时长。

### 音效描述

你需要生成的并不是音频本身，而是一段描述这段音频的 **英文** 文本。

注意：你的描述是一个完整的陈述句，不需要用`Sound of`之类的表达，只需要描述音效相关的正在发生的事情。

注意：你的描述应该是独立于对话内容的，其中不应当包含任何具体事物的名称

注意：你的描述不应包含复杂的形容，不应是详细的刻画，仅仅是描述有什么声音即可

### 输入

每个对话包含说话人`role`和内容`content`，以及这段对话的开始时间`start`和结束时间`end`（以秒为单位）。

{dialogs}

### 输出

注意：你需要返回一个 `json` 数组，其中每个对象格式如下

{{
    "audio_desc": **英文** 的音频描述文本,
    "start": 开始时间（以秒为单位）,
    "end": 结束时间（以秒为单位）,
    "explaination": 解释说明你插入这段音效的意图
}}
"""
