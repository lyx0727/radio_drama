from abc import ABC, abstractmethod


class TTSModel(ABC):
    @abstractmethod
    def register(self, key: str, speech_file: str): ...

    @abstractmethod
    def generate(
        self,
        tts_text: str,
        instruct_text: str,
        speech_key: str,
        output_path: str,
        **kwargs,
    ): 
        """
        Args
        tts_text: 生成语音的文本
        instruct_text: 指令文本
        output_path: 输出路径
        speech_key: 音源对应的 key（注册音源时所指定的），为 None 时自动分配
        """
        ...
