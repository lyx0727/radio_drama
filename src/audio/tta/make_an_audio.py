import subprocess
import os

from ...utils.ffmpeg import cut
from .tta_model import TTAModel


class MakeAnAudioTTAModel(TTAModel):
    def generate(
        self,
        desc,
        duration,
        output_path,
        **kwargs,
    ):
        output_path = os.path.abspath(output_path)
        output_name, ext = os.path.splitext(output_path)
        cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib", "Make-An-Audio"))
        subprocess.check_output(
            [
                "python",
                "gen_wav.py",
                "--prompt",
                desc,
                "--ddim_steps",
                "100",
                "--duration",
                "10",
                "--scale",
                "3",
                "--n_samples",
                "1",
                "--save_name",
                output_name,
            ],
            cwd=cwd,
        )
        # 生成 10s 效果更好，再裁剪到要求时长
        cut(
            f"{output_name}_0{ext}",
            output_path,
            0,
            duration,
        )
