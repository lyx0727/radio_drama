from ..utils.chat_model import ChatModel


def extract_role(text: str, model: ChatModel):
    """
        从一段文本中提取角色信息，包含以下信息：
        Args:
            text (str): 文本
            model (str): openai 模型
        Returns:
            [{
                "name": 角色名,
                "gender": "男"或"女",
                "personality": 角色性格，一句话概括,
                "alias": [角色名字未揭晓时旁白给出的一些称谓，类型为数组]
            }]
    """
    roles = model.generate(
        TEXT_TO_ROLE.format(text=text),
        TEXT_TO_ROLE_SYSTEM,
        return_type='json'
    )
    return roles

TEXT_TO_ROLE_SYSTEM = """
你是一个擅长阅读小说的帮手，能够梳理出小说片段中人物信息
"""

TEXT_TO_ROLE = """
帮我提取出这段文本中所有的角色，包含以下信息：

`name`: 角色名
`gender`: 角色性别，男或女
`personality`: 角色性格，一句话概括
`alias`: 角色名字未揭晓时旁白给出的一些称谓，类型为数组

注意：你只需要返回一个json格式的数组，不需要任何其他输出！！！

每个对象包括上述定义的字段。返回示例如下

[
    {{
        "name": "纳兰嫣然",
        "gender": "女",
        "personality": "高傲倔强",
        "alias": [],
    }},
    ...
]

文本：

{text}
"""
