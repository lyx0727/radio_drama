from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

from .tta_model import TTAModel


class AudioGenTTAModel(TTAModel):
    def generate(
        self,
        desc,
        duration,
        output_path,
        **kwargs,
    ):
        autogen = AudioGen.get_pretrained("./facebook/audiogen-medium")
        autogen.set_generation_params(duration=duration)

        audio = autogen.generate([desc])[0]
        audio_write(
            output_path.replace(".wav", ""),
            audio.cpu(),
            autogen.sample_rate,
            strategy="loudness",
            loudness_compressor=True,
        )
