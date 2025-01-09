# Radio Drama

## 环境

安装 `conda` 环境

``` sh
conda create -n radio_drama python=3.10
conda activate radio_drama
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platform.
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt
```

请安装 `ffmpeg`

```bash
sudo apt-get install ffmpeg
# Or if you are using Anaconda or Miniconda
conda install "ffmpeg<5" -c conda-forge
```

## 运行

提供文本，将文本转换成广播剧

[`run.sh`](run.sh) 脚本提供一键运行，需要配置相关参数，如输入文本路径 `text_file`

```bash
source run.sh
```