TEXT_TO_DIALOG_SYSTEM = """
你是一个擅长阅读小说的帮手，能够梳理出小说中人物的对话信息
"""

TEXT_TO_DIALOG = """
帮我把这段文本转换成对话的形式。对于每一句对话，要求：

### 按人物分类

将对话内容按人物分类，非对话内容作为`旁白`。人物放在`role`字段，说话的内容放在`content`字段。

注意：人物的心理描写不要放到旁白里面，用`人物（内心）`作为对话的一部分。

### 判断语速

用一个`1-5`（慢-快）的数字给语速赋值，放在`speed`字段。

### 判断情感语气

用一个词语判断主要的情感，放在`emo`字段；如果有特别的语气、说明需要修饰，用一句话概括，放在`instruct`字段。

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

你需要返回一个json格式的数组，每个对象包括上述定义的字段。返回示例如下

```json
[
    {{ 
        "role": "旁白", 
        "content": "纳兰嫣然明眸紧紧的盯着不远处那身子略显单薄的青年，目光停留在那张清秀的脸庞之上，在那里，她能够依稀的辨认出当年少年的轮廓，只不过，三年岁月，磨去了少年的稚嫩与尖锐的菱角，现在面前的青年，再没有了当年萧家大厅中骤然爆发的那股锋芒锐气，取而代之的，是深邃的内敛。", 
        "speed": 3, 
        "emo": null, 
        "instruct": null
    }}, 
    {{ 
        "role": "纳兰嫣然(内心)",
        "content": "他...真的变了。",
        "speed": 2, 
        "emo": "诧异",
        "instruct": null
    }},
    {{
        "role": "旁白",
        "content": "纳兰嫣然目光中略微有些复杂，她从来没有想到过，当年的那个废物，居然真的能够毫无惧色的来到云岚宗，并且在面对云岚宗近千弟子时，仍然淡如轻风，没有丝毫的紧张与变色。",
        "speed": 3,
        "emo": null,
        "instruct": null
    }},
    {{ 
        "role": "纳兰嫣然",
        "content": "纳兰家，纳兰嫣然...",
        "speed": 3, 
        "emo": "平静",
        "instruct": null
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
        "emo": "诧异",
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
        "emo": "诧异",
        "instruct": "轻笑"
    }}
    ...
]
```

### 文本：

{text}
"""

TEXT_TO_ROLE_SYSTEM = """
你是一个擅长阅读小说的帮手，能够梳理出小说片段中人物信息
"""

TEXT_TO_ROLE = """
帮我提取出这段文本中所有的角色，包含以下信息：

`name`: 角色名
`gender`: 角色性别，男或女
`personality`: 角色性格，一句话概括

注意：你只需要返回一个json格式的数组，不需要任何其他输出！！！

每个对象包括上述定义的字段。返回示例如下

[
    {{
        "name": "纳兰嫣然",
        "gender": "女",
        "personality": "高傲倔强，内心复杂"
    }},
    ...
]

文本：

{text}
"""