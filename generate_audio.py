import argparse
import logging
import os
import json

from src.dialog import split_text
from src.audio import gen_audio_desc, gen_audio, gen_speech, get_wav_secs
from src.utils.chat_model import OpenAIChatModel
from src.audio.tta import MakeAnAudioTTAModel, AudioGenTTAModel
from src.audio.tts import CosyVoiceTTSModel

def init_chat_model(model: str):
    if "gpt" in model.lower():
        return OpenAIChatModel(model_name=model.lower())
    else:
        raise NotImplementedError()

def init_tts_model(tts_model: str):
    if tts_model.lower() == 'cosyvoice':
        return CosyVoiceTTSModel(
            model_dir=os.path.join("src", "audio", "lib", "CosyVoice", "pretrained_models", "CosyVoice2-0.5B")
        )
    else:
        raise NotImplementedError()

def init_tta_model(tta_model: str):
    if tta_model.lower() == 'make-an-audio':
        return MakeAnAudioTTAModel()
    elif tta_model.lower() == 'audiogen':
        return AudioGenTTAModel()
    else:
        raise NotImplementedError()

def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", type=str, required=True)
    parser.add_argument("--chat_model", type=str, required=True, default="gpt-4o")
    parser.add_argument("--tts_model", type=str, required=True, default="CosyVoice")
    parser.add_argument("--tta_model", type=str, required=True, default="Make-An-Audio")
    parser.add_argument("--speech_source_dir", type=str, default=os.path.join("data", "speech"))
    parser.add_argument("--role_timbre_file", type=str, default=os.path.join("results", "role_timbre.json"))
    
    args = parser.parse_args()
    text_file_name = os.path.basename(args.text_file).split(".")[0]

    dialog_dir = os.path.join("results", "dialog", f"{text_file_name}")
    assert os.path.exists(dialog_dir), f"Run process_text.py first"

    # 载入音源
    male_speech_map = {}
    female_speech_map = {}
    
    for file in os.listdir(args.speech_source_dir):
        if file.endswith(".wav"):
            speech_key = file.split(".")[0]
            speech_file = os.path.join(args.speech_source_dir, file)
            if speech_key.startswith("male"):
                male_speech_map[speech_key] = speech_file
            elif speech_key.startswith("female"):
                female_speech_map[speech_key] = speech_file

    # 载入角色配音表（如果有的话）
    if os.path.exists(args.role_timbre_file):
        role_timbre_map = json.load(open(args.role_timbre_file))
    else:
        role_timbre_map = {}

    speech_output_dir = os.path.join("results", "speech", text_file_name)
    os.makedirs(os.path.join("results", "speech"), exist_ok=True)
    os.makedirs(speech_output_dir, exist_ok=True)

    for i in range(len(split_text(args.text_file))):
        dialog_file = os.path.join(dialog_dir, f"dialog_{i}.json")
        dialogs = json.load(open(dialog_file))
        
        interval_file = os.path.join(dialog_dir, f"interval_{i}.json")
        intervals = json.load(open(interval_file))

        role_file = os.path.join(dialog_dir, f"role_{i}.json")
        roles = json.load(open(role_file))

        logger.info(f"Load {len(dialogs)} dialogs, {len(roles)} roles from {text_file_name}_{i}")

        logger.info("Generating speech...")
        chunk_speech_output_dir = os.path.join(speech_output_dir, str(i))
        os.makedirs(chunk_speech_output_dir, exist_ok=True)
        role_timbre_map = gen_speech(
            dialogs=dialogs,
            roles=roles,
            intervals=intervals,
            model=init_tts_model(args.tts_model),
            output_dir=chunk_speech_output_dir,
            role_timbre_map=role_timbre_map,
            male_speech_map=male_speech_map,
            female_speech_map=female_speech_map,
        )
        speech_output_file = os.path.join(chunk_speech_output_dir, f"speech.wav")
        logger.info(f"Generating {get_wav_secs(speech_output_file)}s speech")

        with open(args.role_timbre_file, "w") as f:
            json.dump(role_timbre_map, f, indent=4, ensure_ascii=False)

        logger.info("Generating audio description...")
        
        chunk_audio_output_dir = chunk_speech_output_dir
        audio_desc_file = os.path.join(chunk_audio_output_dir, "audio_desc.json")
        if os.path.exists(audio_desc_file):
            logger.info(f"Already exists: {audio_desc_file}")
            with open(audio_desc_file, "r") as f:
                audio_descs = json.load(f)
        else:
            audio_descs = gen_audio_desc(
                dialogs=dialogs,
                intervals=intervals,
                speech_source_dir=chunk_speech_output_dir,
                model=init_chat_model(args.chat_model),
            )
            with open(audio_desc_file, "w") as f:
                json.dump(audio_descs, f, indent=4, ensure_ascii=False)
            logger.info(f"Generate {len(audio_descs)} audio description")
            logger.info(f"Save audio description to {audio_desc_file}")

        logger.info("Generating audio...")
        gen_audio(
            audio_descs=audio_descs,
            output_dir=chunk_audio_output_dir,
            model=init_tta_model(args.tta_model),
        )
        audio_output_file = os.path.join(chunk_audio_output_dir, "audio.wav")
        logger.info(f"Generating {get_wav_secs(audio_output_file)}s audio")


if __name__ == "__main__":
    main()