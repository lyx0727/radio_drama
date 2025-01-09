import os

def main():
    # 下载 CosyVoice2 模型
    model_dir = os.path.join(
            "src", "audio", "lib", "CosyVoice", "pretrained_models", "CosyVoice2-0.5B",
        )
    if os.path.exists(model_dir):
        print("Model Already Exists, skip...")
    else:
        from modelscope import snapshot_download

        snapshot_download(
            "iic/CosyVoice2-0.5B",
            local_dir=model_dir,
        )


if __name__ == "__main__":
    main()
