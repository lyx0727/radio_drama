from typing import Dict, List
import os, copy
from tqdm import tqdm
import shutil
import subprocess

from ..utils.alloc import LRUAllocator
from ..utils.ffmpeg import concat, create_silence
from .tts import TTSModel


def gen_speech(
    dialogs: List[Dict],
    roles: List[Dict],
    intervals: List[Dict],
    model: TTSModel,
    output_dir: str,
    role_timbre_map: Dict = {},
    male_speech_map: Dict = {},
    female_speech_map: Dict = {},
):
    """
    生成人物对话配音
    Args:
        dialogs (List[Dict]): 对话列表
        roles (List[Dict]): 人物列表
        intervals (List[Dict]): 对话之间的时间间隔
        model (TTSModel): 语音合成模型
        role_timbre_map (Dict): 人物到音色的映射
        male_speech_map (Dict): 男音色到音频文件映射（wav格式）
        female_speech_map (Dict): 女音色到音频文件映射（wav格式）
    Returns:
        role_timbre_map (Dict): 更新的人物到音色的映射
    """
    role_dic = {}
    for role in roles:
        role_dic[role["name"]] = role
        for name in role["alias"]:
            role_dic[name] = role
    # 固定旁白
    role_dic["旁白"] = {"name": "旁白", "gender": "男", "personality": "冷静客观平淡"}

    for dialog in dialogs:
        if dialog["instruct"] is None:
            dialog["instruct"] = ""

        role_name = _get_role_name(dialog)

        if role_name not in role_dic:
            print(f"unknown role: {dialog['role']}")
            dialog["personality"] = None
            dialog["gender"] = "男"
            continue
        role = role_dic[role_name]
        dialog["personality"] = role["personality"]
        dialog["gender"] = role["gender"]

    # 注册音色对应的音源文件
    for timbre_key, speech_file in (male_speech_map | female_speech_map).items():
        model.register(timbre_key, speech_file)

    male_timbre_map = {}
    female_timbre_map = {}
    for k, v in role_timbre_map.items():
        if v in male_speech_map.keys():
            male_timbre_map[k] = v
        elif v in female_speech_map.keys():
            female_timbre_map[k] = v
        else:
            raise ValueError(f"Speech file {v} not found in male or female speech map")

    # 从给定的音色表基础上初始化 LRU 分配器
    male_timbre_allocator = LRUAllocator(
        candidates=list(male_speech_map.keys()),
        allocated=male_timbre_map,
    )
    female_timbre_allocator = LRUAllocator(
        candidates=list(female_speech_map.keys()),
        allocated=female_timbre_map,
    )

    os.makedirs(output_dir, exist_ok=True)

    for idx, dialog in enumerate(tqdm(dialogs)):
        tts_key = f"{dialog['role']}_{idx}"
        output_file = os.path.join(output_dir, f"{tts_key}.wav")
        if os.path.exists(output_file):
            continue
        tts_text = dialog["content"]
        instruct_text = _get_instruct_text(dialog)

        role_name = _get_role_name(dialog)

        # 分配音色
        if dialog["gender"] == "女":
            speech_key = female_timbre_allocator.get(role_name)
        else:
            speech_key = male_timbre_allocator.get(role_name)

        model.generate(
            tts_text,
            instruct_text,
            speech_key,
            output_file,
        )

        if dialog["role"].endswith("(os)"):
            _transform_os(output_file)

    # 合并
    audio_files = []
    for idx, dialog in enumerate(dialogs):
        tts_key = f"{dialog['role']}_{idx}"
        audio_file = os.path.join(output_dir, f"{tts_key}.wav")
        audio_files.append(audio_file)

        interval = intervals[idx]["interval"]
        if interval > 0:
            silence_file = os.path.join(output_dir, f"silence_{interval}.wav")
            create_silence(interval, silence_file)
            audio_files.append(silence_file)
    concat(audio_files, os.path.join(output_dir, f"speech.wav"))

    return male_timbre_map | female_timbre_map


def _get_role_name(dialog: Dict[str, str]):
    role_name = dialog["role"]
    if role_name.endswith("(os)"):
        role_name = role_name[:-4]
    return role_name


def _get_instruct_text(dialog: Dict[str, str]):
    personality, emo, speed, instruct = (
        dialog["personality"],
        dialog["emo"],
        dialog["speed"],
        dialog["instruct"],
    )
    # 旁白忽略设置的情感、语气，速度保持正常语速
    if dialog["role"] == "旁白":
        instruct = None
        emo = None
        speed = 3

    def speed2instruct(speed: int):
        speed = min(speed, 5)
        speed = max(speed, 1)
        return {
            1: "非常慢速",
            2: "慢速",
            3: "正常语速",
            4: "快速",
            5: "非常快速",
        }[speed]

    def cat(*texts: str, sep=","):
        return sep.join([text for text in texts if text])

    return cat(personality, emo, speed2instruct(speed), instruct)


def _transform_os(file: str):
    # todo
    # 目前仅加混响
    origin_file = file.replace(".wav", "_origin.wav")
    shutil.move(file, origin_file)
    subprocess.check_output(
        [
            "ffmpeg",
            "-i",
            origin_file,
            "-filter:a",
            "volume=0.5",
            "-af",
            "aecho=0.5:0.8:50:0.8",
            "-y",
            file,
        ]
    )
