text_file='data/text/text7.txt'
output_file='output.wav'
chat_model='gpt-4o'
tts_model='CosyVoice'
tta_model='Make-An-Audio'

# 将自己的 OpenAI API Key 配置在环境变量中
export OPENAI_BASE_URL=<your_openai_base_url>
export OPENAI_API_KEY=<your_openai_api_key>

export CUDA_VISIBLE_DEVICES=0

echo "[Step 0] prepare model"
python prepare.py

echo "[Step 1] process text"

python process_text.py \
    --text_file $text_file \
    --model $chat_model \

echo "[Step 2] generate audio"
python generate_audio.py \
    --text_file $text_file \
    --chat_model $chat_model \
    --tts_model $tts_model \
    --tta_model $tta_model \
    --speech_source_dir data/speech \
    --role_timbre_file results/role_timbre.json \

echo "[Step 3] merge speech and audio"
python merge.py --text_file $text_file --output_file $output_file