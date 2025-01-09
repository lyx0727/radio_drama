import os
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

from ..lib.CosyVoice.cosyvoice.cli.model import CosyVoice2Model
from ..lib.CosyVoice.cosyvoice.cli.frontend import CosyVoiceFrontEnd
from ..lib.CosyVoice.cosyvoice.utils.file_utils import load_wav

from .tts_model import TTSModel


class CosyVoiceTTSModel(TTSModel):
    def __init__(
        self,
        model_dir: str,
        spk2info_file: str = None,
        load_jit=False,
        load_trt=False,
        load_onnx=False,
    ):
        with open(os.path.join(model_dir, "cosyvoice.yaml"), "r") as f:
            configs = load_hyperpyyaml(
                f,
                overrides={
                    "qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")
                },
            )
        self.model = CosyVoice2Model(configs["llm"], configs["flow"], configs["hift"], fp16=False)
        self.model.load(
            os.path.join(model_dir, "llm.pt"),
            os.path.join(model_dir, "flow.pt"),
            os.path.join(model_dir, "hift.pt"),
        )

        self.frontend = CosyVoiceFrontEnd(
            configs["get_tokenizer"],
            configs["feat_extractor"],
            os.path.join(model_dir, "campplus.onnx"),
            os.path.join(model_dir, "speech_tokenizer_v2.onnx"),
            spk2info_file if spk2info_file else os.path.join(model_dir, "spk2info.pt"),
            configs["allowed_special"],
        )

        if load_jit:
            self.model.load_jit(os.path.join(model_dir, "flow.encoder.fp32.zip"))
        if load_trt is True and load_onnx is True:
            load_onnx = False
            print(
                "can not set both load_trt and load_onnx to True, force set load_onnx to False"
            )
        if load_onnx:
            self.model.load_onnx(
                os.path.join(model_dir, "flow.decoder.estimator.fp32.onnx")
            )
        if load_trt:
            self.model.load_trt(
                os.path.join(model_dir, "flow.decoder.estimator.fp16.Volta.plan")
            )

        self.sample_rate = configs["sample_rate"]
        del configs

        self.speech_input_dic = {}

    def register(self, key, speech_file: str):
        speech_16k = load_wav(speech_file, 16000)
        self.speech_input_dic[key] = get_model_input_from_speech(
            self.frontend, speech_16k, self.sample_rate
        )

    @torch.no_grad()
    def generate(
        self,
        tts_text,
        instruct_text,
        speech_key,
        output_path,
        **kwargs,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert (
            speech_key in self.speech_input_dic
        ), f"speech_key not found: {speech_key}"
        prompt_speech_input = self.speech_input_dic[speech_key]

        for text in self.frontend.text_normalize(
            tts_text, split=True, text_frontend=True
        ):
            tts_text_token, tts_text_token_len = self.frontend._extract_text_token(text)
            prompt_text_token, prompt_text_token_len = (
                self.frontend._extract_text_token(instruct_text + "<|endofprompt|>")
            )

            model_input = {
                "text": tts_text_token,
                "text_len": tts_text_token_len,
                "prompt_text": prompt_text_token,
                "prompt_text_len": prompt_text_token_len,
                **prompt_speech_input,
            }

            model_input = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in model_input.items()
            }

            tts_speeches = []
            for model_output in self.model.tts(**model_input, stream=False, speed=1.0):
                speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
                print("generate speech len {}".format(speech_len))
                tts_speeches.append(model_output["tts_speech"])

        tts_speeches = torch.concat(tts_speeches, dim=1)
        torchaudio.save(output_path, tts_speeches, sample_rate=self.sample_rate)


def get_model_input_from_speech(
    frontend: CosyVoiceFrontEnd, prompt_speech_16k, resample_rate
):
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
