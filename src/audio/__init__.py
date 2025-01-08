import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib", "CosyVoice"))
sys.path.append(os.path.join(os.path.dirname(__file__), "lib", "CosyVoice", "third_party", "Matcha-TTS"))

from .audio import gen_audio_desc, gen_audio, get_wav_secs
from .speech import gen_speech
