import os
import argparse
from tqdm import tqdm

from src.utils.ffmpeg import scale_volume, merge, concat
from src.audio import get_wav_secs
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", type=str, required=True, nargs='+')
    parser.add_argument("--output_file", type=str, default="output.wav")

    args = parser.parse_args()

    output_files = []
    for text_file in tqdm(args.text_file):
        text_name = os.path.basename(text_file).split(".")[0]
        result_dir = os.path.join("results", f"speech_{text_name}")

        speech_file = os.path.join(result_dir, "speech.wav")
        if not os.path.exists(speech_file):
            raise FileNotFoundError(f"{speech_file} does not exist")
        audio_file = os.path.join(result_dir, "audio.wav")
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"{audio_file} does not exist")
        
        # 背景音效调低音量
        audio_tmp_file = audio_file.replace(".wav", "_tmp.wav")
        scale_volume(audio_file, audio_tmp_file, 0.5)

        output_file = os.path.join(result_dir, "output.wav")
        output_files.append(output_file)
        merge([speech_file, audio_tmp_file], output_file)
        print(f"Successfully merged {output_file}: {get_wav_secs(output_file)}s")
    concat(output_files, args.output_file)

if __name__ == "__main__":
    main()