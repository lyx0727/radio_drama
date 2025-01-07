import subprocess
from typing import List


def cut(input_path: str, output_path: str, start: int, end: int):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ss",
            str(start),
            "-to",
            str(end),
            "-c",
            "copy",
            output_path,
        ]
    )


def create_silence(secs: int, file_path: str = None, sample_rate: int = 24000):
    if not file_path:
        file_path = f"silence_{secs}.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl=mono",
            "-t",
            str(secs),
            "-c:a",
            "pcm_s16le",
            "-y",
            file_path,
        ],
        capture_output=True,
    )
    return file_path


def concat(audios: List[str], output_path: str, fade_durations: List[int] = None):

    input_args = []
    ffmpeg_args = []
    for audio in audios:
        input_args.extend(["-i", audio])
    filter_complex, out = _get_filter_complex(audios, fade_durations)
    ffmpeg_args.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            out,
        ]
    )
    ffmpeg_args.extend(
        [
            # "-loglevel",
            # "debug",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "24000",
            "-ac",
            "2",
            "-y",
            output_path,
        ]
    )
    args = (
        [
            "ffmpeg",
        ]
        + input_args
        + ffmpeg_args
    )
    subprocess.check_output(args)


def merge(audios: List[str], output_path: str):
    pass


def _get_filter_complex(files, fade_durations, curve_type="tri"):
    filter_complex = ""

    if fade_durations:
        last_output = "[0:a]"

        for i, (file, fade_duration) in enumerate(zip(files[1:], fade_durations)):
            filter_complex += f"{last_output}[{i}:a]acrossfade=d={fade_duration}:c1={curve_type}:c2={curve_type}[a{i}]; "
            last_output = f"[a{i}]"

        # 去掉最后一个分号和空格
        filter_complex = filter_complex.rstrip("; ")
    else:
        for i, file in enumerate(files):
            filter_complex += f"[{i}:a]"
        filter_complex += f"concat=n={len(files)}:v=0:a=1[out]"
        last_output = "[out]"
    return filter_complex, last_output