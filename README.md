# Radio Drama

## Clone

```bash
git clone https://github.com/lyx0727/radio_drama.git
cd radio_drama
git submodule update --init --recursive
```

## 环境

安装 `conda` 环境

``` sh
conda create -n radio_drama python=3.10
conda activate radio_drama
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platform.
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt
```

请安装 `ffmpeg`

```bash
sudo apt-get install ffmpeg
# Or if you are using Anaconda or Miniconda
conda install "ffmpeg<5" -c conda-forge
```

## 运行

提供文本，将文本转换成广播剧

[`run.sh`](run.sh) 脚本提供一键运行，需要配置相关参数，如输入文本路径 `text_file`

```bash
source run.sh
```

## 代码说明

### 总览

代码树状图如下
```
src
├── utils
│   ├── alloc.py: 实现了一个 LRU 分配器供分配音色使用
│   ├── chat_model.py: 封装了 openai 对话接口
│   ├── ffmpeg.py: 封装了 ffmpeg 一些音频处理操作
├── dialog
│   ├── dialog.py: 对话处理逻辑
│   ├── role.py: 角色生成逻辑
├── audio
│   ├── lib: 开源音频模型源码仓库
│   ├── tts: 封装了 tts 模型生成接口
│   ├── tta: 封装了 tta 模型生成接口
│   ├── speech.py: 配音生成逻辑
│   └── audio.py: 音效生成逻辑
```
### 配音生成

下面是配音生成主要逻辑，节选自[src/audio/speech.py@gen_speech](src/audio/speech.py#L12)

```python
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
    role_dic["旁白"] = {
        "name": "旁白", 
        "gender": "男", 
        "personality": "冷静客观平淡"
    }

    for dialog in dialogs:
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

    # 从给定的音色表基础上初始化 LRU 分配器
    male_timbre_allocator = LRUAllocator(
        candidates=list(male_speech_map.keys()),
        allocated=male_timbre_map,
    )
    female_timbre_allocator = LRUAllocator(
        candidates=list(female_speech_map.keys()),
        allocated=female_timbre_map,
    )


    for idx, dialog in dialogs:
        tts_text = dialog["content"]
        instruct_text = _get_instruct_text(dialog)

        role_name = _get_role_name(dialog)

        # 分配音色
        if dialog["gender"] == "女":
            speech_key = female_timbre_allocator.get(role_name)
        else:
            speech_key = male_timbre_allocator.get(role_name)

        # 生成配音
        model.generate(
            tts_text,
            instruct_text,
            speech_key,
            output_file,
        )

        # 处理心理活动
        if dialog["role"].endswith("(os)"):
            _transform_os(output_file)

    # 合并
    ...

    return male_timbre_map | female_timbre_map

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
```
TTS 模型推理逻辑封装在[src/audio/tts/cosyvoice.py@CosyVoiceTTSModel](src/audio/tts/cosyvoice.py#L13)中

### 音效生成

下面是音效生成逻辑，节选自[src/audio/audio.py@gen_audio](src/audio/audio.py#L63)

```python
def gen_audio(audio_descs: List[Dict], output_dir: str, model: TTAModel):
    for i, audio in enumerate(audio_descs):
        desc = audio["audio_desc"]
        # 限制最长时间为 3 秒
        duration = min(audio["end"] - audio["start"], 3)
        # 生成
        model.generate(
            desc,
            duration,
            output_path,
        )

    audio_files = []
    # 音效拼接时设置淡入淡出时间，默认为 2 秒
    fade_durations = []
    end = 0
    last_duration = 100
    for j, audio in enumerate(audio_descs):
        if audio["start"] > end:
            silence_file = os.path.join(
                output_dir, f"silence_{audio['start'] - end}.wav"
            )
            # 通过静音文件对齐时间
            create_silence(audio["start"] - end, silence_file)
            audio_files.append(silence_file)
            # 如果音频本身短于 2s，淡入淡出时间取音频时间
            fade_durations.append(min(2, min(last_duration, audio["start"] - end)))
            last_duration = audio["start"] - end
        audio_files.append(audio_file)
        duration = min(3, audio["end"] - audio["start"])
        fade_durations.append(min(2, min(last_duration, duration)))
        last_duration = duration
        end = audio["start"] + duration
    concat(audio_files, output_file, fade_durations)
```
