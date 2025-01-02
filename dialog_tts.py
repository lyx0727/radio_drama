# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
import torch
from torch.utils.data import DataLoader
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
import json

from CosyVoice.cosyvoice.cli.model import CosyVoice2Model
from CosyVoice.cosyvoice.cli.frontend import CosyVoiceFrontEnd
from CosyVoice.cosyvoice.utils.file_utils import load_wav


def get_args():
    parser = argparse.ArgumentParser(description='inference with your model')
    parser.add_argument('--dialog_file', required=True, help='dialog json file')
    parser.add_argument('--role_file', required=True, help='role json file')
    parser.add_argument('--model_dir', required=True, help='model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--llm_model', required=True, help='llm model file')
    parser.add_argument('--flow_model', required=True, help='flow model file')
    parser.add_argument('--hifigan_model', required=True, help='hifigan model file')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--spk2info_file', help='spk2info embedding file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--load_jit',
                        action='store_true',
                        help='load jit model')
    parser.add_argument('--load_onnx',
                        action='store_true',
                        help='load onnx model')
    parser.add_argument('--load_trt',
                        action='store_true',
                        help='load trt model')
    args = parser.parse_args()
    logging.info(args)
    return args

def get_model_input_from_speech(frontend: CosyVoiceFrontEnd, prompt_speech_16k, resample_rate):
    prompt_speech_resample = torchaudio.transforms.Resample(
        orig_freq=16000, new_freq=resample_rate
    )(prompt_speech_16k)
    speech_feat, speech_feat_len = frontend._extract_speech_feat(prompt_speech_resample)
    speech_token, speech_token_len = frontend._extract_speech_token(prompt_speech_16k)
    if resample_rate == 24000:
        # cosyvoice2, force speech_feat % speech_token = 2
        token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
        speech_feat, speech_feat_len[:] = (
            speech_feat[:, : 2 * token_len],
            2 * token_len,
        )
        speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
    embedding = frontend._extract_spk_embedding(prompt_speech_16k)
    return {
        # "llm_prompt_speech_token": speech_token,
        # "llm_prompt_speech_token_len": speech_token_len,
        "flow_prompt_speech_token": speech_token,
        "flow_prompt_speech_token_len": speech_token_len,
        "prompt_speech_feat": speech_feat,
        "prompt_speech_feat_len": speech_feat_len,
        "llm_embedding": embedding,
        "flow_embedding": embedding,
    }


def main():
    
    args = get_args()


    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Init cosyvoice models from configs
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={
            "qwen_pretrain_path": os.path.join(
                args.model_dir, "CosyVoice-BlankEN"
            )
        })
    logging.info(configs.keys())
    model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'])
    model.load(args.llm_model, args.flow_model, args.hifigan_model)

    frontend = CosyVoiceFrontEnd(
        configs["get_tokenizer"],
        configs["feat_extractor"],
        "{}/campplus.onnx".format(args.model_dir),
        "{}/speech_tokenizer_v2.onnx".format(args.model_dir),
        args.spk2info_file if args.spk2info_file else "{}/spk2info.pt".format(args.model_dir),
        configs["allowed_special"],
    )

    if args.load_jit:
        model.load_jit("{}/flow.encoder.fp32.zip".format(args.model_dir))
    if args.load_trt is True and args.load_onnx is True:
        args.load_onnx = False
        logging.warning(
            "can not set both load_trt and load_onnx to True, force set load_onnx to False"
        )
    if args.load_onnx:
        model.load_onnx(
            "{}/flow.decoder.estimator.fp32.onnx".format(args.model_dir)
        )
    if args.load_trt:
        model.load_trt(
            "{}/flow.decoder.estimator.fp16.Volta.plan".format(args.model_dir)
        )


    sample_rate = configs["sample_rate"]

    del configs

    with open(args.dialog_file, "r", encoding="utf-8") as f:
        dialogs = json.load(f)
    with open(args.role_file, "r", encoding="utf-8") as f:
        roles = json.load(f)

    logging.info(f"{len(dialogs)} dialogs loaded")
    logging.info(f"{len(roles)} roles loaded")

    role_dic = {role["name"]: role for role in roles}
    role_dic["旁白"] = {"name": "旁白", "gender": "男", "personality": "冷静客观平淡"}
    
    # preprocess
    for dialog in dialogs:
        if dialog['instruct'] is None:
            dialog["instruct"] = ''

        role_name = dialog["role"]
        if role_name.endswith("(内心)"):
            role_name = role_name[:-4]
        if role_name not in role_dic:
            logging.info(f"unknown role: {dialog['role']}")
            dialog['personality'] = ''
            dialog['gender'] = '男'
            continue
        role = role_dic[role_name]
        dialog['personality'] = role['personality']
        dialog['gender'] = role['gender']

    test_dataset = dialogs

    speech_input_dic = {}
    def register_wav(key: str, wav_file: str):
        speech_16k = load_wav(wav_file, 16000)
        speech_input_dic[key] = get_model_input_from_speech(frontend, speech_16k, sample_rate)
    register_wav('female1', '/home/lyx/radio_drama/female1.wav')
    register_wav('male1', '/home/lyx/radio_drama/male1.wav')

    def get_prompt_speech_input(item):
        if item['gender'] == '男':
            return speech_input_dic['male1']
        else:
            return speech_input_dic['female1']
    
    SPEED2INSTRUCT = {
        1: '非常慢速',
        2: '慢速',
        3: '正常语速',
        4: '快速',
        5: '非常快速',
    }

    def speed2instruct(speed: int):
        assert speed in range(1, 6), "speed must in range(1, 6)"
        return SPEED2INSTRUCT[speed]

    def get_instruct_text(item):
        personality, emo, speed, instruct = item['personality'], item['emo'], item['speed'], item['instruct']
        def cat(*texts: str, sep=','):
            return sep.join([text for text in texts if text])
        return cat(personality, emo, speed2instruct(speed), instruct)

    
    os.makedirs(args.result_dir, exist_ok=True)
    fn = os.path.join(args.result_dir, 'wav.scp')
    f = open(fn, 'w')
    with torch.no_grad():
        for idx, item in enumerate(tqdm(test_dataset)):
            tts_text = item["content"]
            prompt_speech_input = get_prompt_speech_input(item)
            instruct_text = get_instruct_text(item)
            

            for i in frontend.text_normalize(
                tts_text, split=True, text_frontend=True
            ):
                tts_text_token, tts_text_token_len = frontend._extract_text_token(tts_text)
                prompt_text_token, prompt_text_token_len = frontend._extract_text_token(instruct_text + "<|endofprompt|>")

                model_input = {
                    "text": tts_text_token,
                    "text_len": tts_text_token_len,
                    "prompt_text": prompt_text_token,
                    "prompt_text_len": prompt_text_token_len,
                    **prompt_speech_input
                }

                model_input = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in model_input.items()
                }
                logging.info(20 * ">>>")
                logging.info("synthesis text {}".format(i))
                logging.info("instruct: {}".format(instruct_text))
                tts_speeches = []
                for model_output in model.tts(
                    **model_input, stream=False, speed=1.0
                ):
                    speech_len = model_output["tts_speech"].shape[1] / sample_rate
                    logging.info(
                        "generate speech len {}".format(speech_len)
                    )
                    tts_speeches.append(model_output['tts_speech'])
            
            tts_speeches = torch.concat(tts_speeches, dim=1)
            tts_key = f"{item['role']}_{idx}"
            tts_fn = os.path.join(args.result_dir, '{}.wav'.format(tts_key))
            torchaudio.save(tts_fn, tts_speeches, sample_rate=sample_rate)
            f.write('{} {}\n'.format(tts_key, tts_fn))
            f.flush()
    f.close()
    logging.info('Result wav.scp saved in {}'.format(fn))

if __name__ == '__main__':
    main()
