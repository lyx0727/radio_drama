from abc import ABC, abstractmethod


class TTAModel(ABC):
    @abstractmethod
    def generate(
        self,
        desc: str,
        duration: int,
        output_path: str,
        **kwargs,
    ): 
        """
        Args:
        desc: 描述文本
        duration: 音频时长，单位秒
        output_path: 输出路径
        """
        ...
