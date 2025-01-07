import sys
sys.path.append("..")

from src.utils.chat_model import OpenAIChatModel

def test_openai_chat_model():
    model = OpenAIChatModel()

    print(model.generate("hello", return_type="text"))