import os

from ..utils.chat_model import ChatModel


def split_text(
    text_file: str,
    model: ChatModel=None,
    max_chunk_lines=50,
):
    with open(text_file, "r") as f:
        text = f.read()

    text_dir = os.path.dirname(text_file)
    text_name = os.path.basename(text_file).split(".")[0]
    chunk_dir = os.path.join(text_dir, text_name)

    if os.path.exists(chunk_dir):
        print(f"Already exists {chunk_dir}, skipping")
        return [
            os.path.join(chunk_dir, f"{i}.txt") for i in range(len(os.listdir(chunk_dir)))
        ]


    assert model is not None, "Please provide a model"
    os.makedirs(chunk_dir, exist_ok=True)

    text_lines = text.splitlines()
    text_lines = [line for line in text_lines if line.strip()]

    start = 0
    i = 0
    chunks = []
    while start < len(text_lines):
        end = min(start + max_chunk_lines, len(text_lines))
        chunk = text_lines[start:end]
        first, second = 0, 0
        first, second = _gen_split(chunk, model)
        if first + second != len(chunk):
            print(
                f"Warning: split assert failed, first={first}, second={second}, len(chunk)={len(chunk)}"
            )
        chunks.append(chunk[:first])

        i += 1
        start += first

    # 在人为合并一下，防止太短
    merged_chunks = []
    last_chunk = []
    for chunk in chunks:
        if len(last_chunk) + len(chunk) > max_chunk_lines:
            merged_chunks.append(last_chunk)
            last_chunk = chunk
        else:
            last_chunk += chunk
    chunk_files = []
    for i, chunk in enumerate(merged_chunks):
        chunk_file = os.path.join(chunk_dir, f"{i}.txt")
        chunk_files.append(chunk_file)
        with open(chunk_file, "w") as f:
            f.write("\n".join(chunk))
        print(f"Chunk[{i}]: {len(chunk)} lines in {chunk_file}")
    return chunk_files


def _gen_split(lines: list[str], model: ChatModel):
    res = model.generate(
        TEXT_TO_CHUNKS.format(n=len(lines), text="\n".join(lines)),
        return_type="json",
    )
    print(20 * ">>>")
    print(res)
    divide_pos = -1
    for i, line in enumerate(lines):
        if res["divide_line"] and res["divide_line"] in line:
            divide_pos = i
            break
    if divide_pos < 0:
        return res["first"], res["second"]
    return divide_pos, len(lines) - divide_pos


TEXT_TO_CHUNKS = """
请把下面这段文本按照内容分割成两部分，满足
- 第一部分在语义上是比较连贯的，是相对完整的
- 第二部分不一定是完整的（甚至可以是空的），但与第一部分有一定的语义差别

注意：你需要在确保第一段足够完整的基础上，保证第一部分尽可能多。你甚至可以将整段文本作为第一部分。你最少需要保证第一部分的行数超过第二部分！！！

注意：我将给你`n`行文本，你需要返回一个`json`格式的对象，包含两部分的行数。返回格式如下

注意：你需要确保返回的两部分行数相加等于`n`！！！

注意：你只需要返回一个`json 格式的对象，不要返回其他任何内容！！！

{{
    "first": 第一部分行数,
    "divide_line": 第二部分的第一行内容（与原文一模一样！！！）,
    "second": 第二部分行数
}}

下面是文本，总行数为`{n}`。每行之间用换行符连接。如果有空行请忽略：

{text}
"""
