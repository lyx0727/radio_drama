pretrained_model_dir=CosyVoice/pretrained_models/CosyVoice2-0.5B

export PYTHONPATH=CosyVoice:CosyVoice/third_party/Matcha-TTS:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

nohup python CosyVoice/cosyvoice/bin/dialog_tts.py \
      --gpu 0 \
      --dialog_file results/dialog/dialog0_gpt-4.json \
      --role_file results/dialog/role_gpt-4o.json \
      --model_dir $pretrained_model_dir \
      --config $pretrained_model_dir/cosyvoice.yaml \
      --llm_model $pretrained_model_dir/llm.pt \
      --flow_model $pretrained_model_dir/flow.pt \
      --hifigan_model $pretrained_model_dir/hift.pt \
      --result_dir results/dialog_tts \
      --spk2info_file CosyVoice/pretrained_models/CosyVoice-300M-SFT/spk2info.pt \
      --load_jit \
> log/dialog_tts.log 2>&1 &