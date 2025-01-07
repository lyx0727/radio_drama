from typing import List, Dict
import json

from ..utils.chat_model import ChatModel


def extract_dialog(text: str, model: ChatModel):
    """
    从一段文本中提取对话信息，包含以下信息：
    Args:
        text (str): 文本
        model (str): openai 模型
    Returns:
        [{
            "role": 角色名（"旁白"或人物名或"<人物>(os)"）,
            "content": "说话内容",
            "emo": 说话情感,
            "speed": 说话语速, 1-5（慢-快）,
            "instruct": 其他关于语气的描述
        }]
    """
    dialogs = model.generate(
        TEXT_TO_DIALOG.format(text=text),
        TEXT_TO_DIALOG_SYSTEM,
        max_completion_tokens=16384,
        return_type="json",
    )
    return dialogs


def gen_interval(dialogs: List[Dict], model: ChatModel):
    """
    生成对话之间的时间间隔
    Args:
        dialogs (List[Dict]): 对话列表
        model (str): openai 模型
    Returns:
        [{
            "role": 当前说话的角色，与输入一致,
            "content": 当前说话的内容，与输入一致,
            "interval": 与下一句间隔的秒数,
        }]
    """
    dialogs = [
        {
            "role": dialog["role"],
            "content": dialog["content"],
        }
        for dialog in dialogs
    ]

    intervals = model.generate(
        DIALOG_TO_INTERVAL.format(
            dialogs=json.dumps(dialogs, indent=4, ensure_ascii=False)
        ),
    )
    return intervals


TEXT_TO_DIALOG_SYSTEM = """
你是一个擅长阅读小说的帮手，能够梳理出小说中人物的对话信息，并提取出其中的各种信息
"""

TEXT_TO_DIALOG_old = """
帮我把这段文本转换成对话的形式。对于每一句对话，要求：

### 按人物分类

将对话内容按人物分类，非对话内容作为`旁白`。人物放在`role`字段，说话的内容放在`content`字段。

注意：人物的心理描写不要放到旁白里面，用`人物（内心）`作为对话的一部分。

### 判断语速

用一个`1-5`（慢-快）的数字给语速赋值，放在`speed`字段。

### 判断情感语气

从下面的词语中选一个词语判断主要的情感，放在`emo`字段（如果没有特别的情感也可以不填）：
高兴(Happy)
悲伤(Sad)
惊讶(Surprised)
愤怒(Angry)
恐惧(Fearful)
厌恶(Disgusted)
冷静(Calm)
严肃(Serious)

如果有特别的语气、说明需要修饰，用一句话概括，放在`instruct`字段。
如神秘(Mysterious), 凶猛(Fierce), 好奇(Curious), 优雅(Elegant), 孤独(Lonely), 机器
人(Robot), 小猪佩奇(Peppa)等

你也可以直接在`content`中插入下面的标签，以辅助语气的需要
[breath]
[noise],
[laughter]
[cough]
[clucking]
[accent]
[quick_breath]
[hissing]
[sigh]
[vocalized-noise]
[lipsmack]
[mn]
<strong>, </strong>
<laughter>, </laughter>

### 判断音效

人物对话过程中可能有一些环境音效，你需要根据文本的描写自行判断，并用一句 **英文** 描述，如果有放在`audio`字段。
同时，在`audio_type`字段标明这个音效是一次性的（`immediate`）还是持续的（`continuous`）
注意：这个动作描述不应包含任何特定人物相关信息，应该是独立于文本的。

### 返回格式及示例

注意：你需要返回一个json格式的数组，每个对象包括上述定义的字段。请返回全部内容，不要省略文本中的信息！！！请返回全部内容，不要省略文本中的信息！！！请返回全部内容，不要省略文本中的信息！！！
返回示例如下

```json
[
    {{ 
        "role": "旁白", 
        "content": "纳兰嫣然明眸紧紧的盯着不远处那身子略显单薄的青年，目光停留在那张清秀的脸庞之上，在那里，她能够依稀的辨认出当年少年的轮廓，只不过，三年岁月，磨去了少年的稚嫩与尖锐的菱角，现在面前的青年，再没有了当年萧家大厅中骤然爆发的那股锋芒锐气，取而代之的，是深邃的内敛。", 
        "speed": 3, 
        "emo": null, 
        "instruct": null,
        "audio": null,
        "audio_type": null
    }}, 
    {{ 
        "role": "纳兰嫣然(内心)",
        "content": "他...真的变了。",
        "speed": 2, 
        "emo": "惊讶",
        "instruct": null,
        "audio": null,
        "audio_type": null
    }},
    {{
        "role": "旁白",
        "content": "纳兰嫣然目光中略微有些复杂，她从来没有想到过，当年的那个废物，居然真的能够毫无惧色的来到云岚宗，并且在面对云岚宗近千弟子时，仍然淡如轻风，没有丝毫的紧张与变色。",
        "speed": 3,
        "emo": null,
        "instruct": null,
        "audio": null,
        "audio_type": null
    }},
    {{ 
        "role": "纳兰嫣然",
        "content": "纳兰家，纳兰嫣然...",
        "speed": 3, 
        "emo": "冷静",
        "instruct": null,
        "audio": null,
        "audio_type": null
    }},
    {{
        "role": "旁白",
        "content": "纳兰嫣然缓缓的站起身来，娇躯挺拔得犹如一朵傲骨雪莲，明眸盯着萧炎。",
        "speed": 3,
        "emo": null,
        "instruct": null,
        "audio": "Slowly stood up and walked over",
        "audio_type": "immediate"
    }},
    {{
        "role": "加刑天",
        "content": "那便是萧家的那个小家伙？不是说是个不能储存斗气的废物么？",
        "speed": 4,
        "emo": "惊讶",
        "instruct": null,
        "audio": null,
        "audio_type": null
    }},
    {{
        "role": "旁白",
        "content": "巨树之上，加刑天望着萧炎，眼中有着几缕诧异",
        "speed": 3,
        "emo": null,
        "instruct": null,
        "audio": null,
        "audio_type": null
    }},
    {{
        "role": "加刑天",
        "content": "<laughter>呵呵</laughter>，可看他现在这副气度，可不象是外强内干强行装出来的，而且，就算是装的，能够在云岚宗那些老家伙特意组合而成的整体气势中保持这般从容，那也不是普通人能干得出来的事啊。",
        "speed": 3,
        "emo": "惊讶",
        "instruct": "轻笑",
        "audio": null,
        "audio_type": null
    }}
    ...
]
```

### 文本

{text}
"""

TEXT_TO_DIALOG = """
帮我把这段文本转换成对话的形式。对于每一句对话，要求：

### 按人物分类

将对话内容按人物分类，非对话内容作为`旁白`。人物放在`role`字段，说话的内容放在`content`字段。

注意：你可以适当浓缩旁白的内容，从而突出人物对话内容。但是对话内容不应随意更改

注意：人物的心理描写需要单独提取出来，用`人物(os)`作为对话的一部分。

### 判断语速

用一个`1-5`（慢-快）的数字给语速赋值，放在`speed`字段。

### 判断情感语气

从下面的词语中选一个词语判断主要的情感，放在`emo`字段（如果没有特别的情感也可以不填）：
高兴(Happy)
悲伤(Sad)
惊讶(Surprised)
愤怒(Angry)
恐惧(Fearful)
厌恶(Disgusted)
冷静(Calm)
严肃(Serious)

如果有特别的语气、说明需要修饰，用一句话概括，放在`instruct`字段。

注意：这句话应当与人物无关，不应包含任何具体人物的信息，仅仅表示说话的语气，如神秘(Mysterious), 凶猛(Fierce), 好奇(Curious), 优雅(Elegant), 孤独(Lonely)等

你也可以直接在`content`中插入下面的标签，以辅助语气的需要
[breath]
[noise],
[laughter]
[cough]
[clucking]
[accent]
[quick_breath]
[hissing]
[sigh]
[vocalized-noise]
[lipsmack]
[mn]
<strong>, </strong>
<laughter>, </laughter>

### 返回格式及示例

注意：你需要返回一个json格式的数组，每个对象包括上述定义的字段。请返回全部内容，不要省略文本中的信息！！！请返回全部内容，不要省略文本中的信息！！！请返回全部内容，不要省略文本中的信息！！！
返回示例如下

```json
[
    {{ 
        "role": "旁白", 
        "content": "纳兰嫣然明眸紧紧的盯着不远处那身子略显单薄的青年，目光停留在那张清秀的脸庞之上，在那里，她能够依稀的辨认出当年少年的轮廓，只不过，三年岁月，磨去了少年的稚嫩与尖锐的菱角，现在面前的青年，再没有了当年萧家大厅中骤然爆发的那股锋芒锐气，取而代之的，是深邃的内敛。", 
        "speed": 3, 
        "emo": null, 
        "instruct": null,
    }}, 
    {{ 
        "role": "纳兰嫣然(os)",
        "content": "他...真的变了。",
        "speed": 2, 
        "emo": "惊讶",
        "instruct": null,
    }},
    {{
        "role": "旁白",
        "content": "纳兰嫣然目光中略微有些复杂，她从来没有想到过，当年的那个废物，居然真的能够毫无惧色的来到云岚宗，并且在面对云岚宗近千弟子时，仍然淡如轻风，没有丝毫的紧张与变色。",
        "speed": 3,
        "emo": null,
        "instruct": null,
    }},
    {{ 
        "role": "纳兰嫣然",
        "content": "纳兰家，纳兰嫣然...",
        "speed": 3, 
        "emo": "冷静",
        "instruct": null,
    }},
    {{
        "role": "旁白",
        "content": "纳兰嫣然缓缓的站起身来，娇躯挺拔得犹如一朵傲骨雪莲，明眸盯着萧炎。",
        "speed": 3,
        "emo": null,
        "instruct": null
    }},
    {{
        "role": "加刑天",
        "content": "那便是萧家的那个小家伙？不是说是个不能储存斗气的废物么？",
        "speed": 4,
        "emo": "惊讶",
        "instruct": null
    }},
    {{
        "role": "旁白",
        "content": "巨树之上，加刑天望着萧炎，眼中有着几缕诧异",
        "speed": 3,
        "emo": null,
        "instruct": null
    }},
    {{
        "role": "加刑天",
        "content": "<laughter>呵呵</laughter>，可看他现在这副气度，可不象是外强内干强行装出来的，而且，就算是装的，能够在云岚宗那些老家伙特意组合而成的整体气势中保持这般从容，那也不是普通人能干得出来的事啊。",
        "speed": 3,
        "emo": "惊讶",
        "instruct": "轻笑"
    }}
    ...
]
```

### 文本

{text}
"""

DIALOG_TO_INTERVAL = """
下面是从一段音频中获取的对话文本，每个对话包含说话人`role`和内容`content`。
{dialogs}

请你给出对话两两之前间隔的时间（单位为秒）

注意：你需要判断出对话之间的关系，如对话被打断时应当设置停顿为`0`。正常情况下，两句对话应相隔`1`。

注意：你只需要返回一个json格式的数组，不需要任何其他输出！！！

返回格式如下定义

[
    {{
        "role": 当前说话的角色，与原文一致,
        "content": 当前说话的内容，与原文一致,
        "interval": 与下一句间隔的秒数
    }},
    ...
]
"""
