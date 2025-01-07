import argparse
import os
import json

from src.dialog import extract_dialog, extract_role, gen_interval

from src.utils.chat_model import OpenAIChatModel
import logging

def init_chat_model(model: str):
    if "gpt" in model.lower():
        return OpenAIChatModel(model_name=model.lower())
    else:
        raise NotImplementedError()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    model = init_chat_model(args.model)

    with open(args.text_file, "r") as f:
        text = f.read()
    text_file_name = os.path.basename(args.text_file).split(".")[0]

    os.makedirs("results", exist_ok=True)
    # extrac dialog
    dialog_output_file = os.path.join("results", "dialog", f"{text_file_name}.json")
    if os.path.exists(dialog_output_file):
        logger.info(f"Already exists: {dialog_output_file}")
    else:
        dialogs = extract_dialog(text, model)
        os.makedirs(os.path.join("results", "dialog"), exist_ok=True)
        with open(dialog_output_file, "w") as f:
            json.dump(dialogs, f, indent=4, ensure_ascii=False)
        logger.info(f"Extract {len(dialogs)} dialogs from {args.text_file}")
        logger.info(f"Save dialogs to {dialog_output_file}")

    # extract roles
    role_output_file = os.path.join("results", "dialog", f"role_{text_file_name}.json")
    if os.path.exists(role_output_file):
        logger.info(f"Already exists: {role_output_file}")
    else:
        roles = extract_role(text, model)
        with open(role_output_file, "w") as f:
            json.dump(roles, f, indent=4, ensure_ascii=False)
        logger.info(f"Extract {len(roles)} roles from {args.text_file}")
        logger.info(f"Save roles to {role_output_file}")

    # generate intervals
    interval_output_file = os.path.join(
        "results", "dialog", f"interval_{text_file_name}.json"
    )
    if os.path.exists(interval_output_file):
        logger.info(f"Already exists: {interval_output_file}")
    else:
        intervals = gen_interval(dialogs, model)
        with open(interval_output_file, "w") as f:
            json.dump(intervals, f, indent=4, ensure_ascii=False)
        logger.info(f"Generate {len(intervals)} intervals from {args.text_file}")
        logger.info(f"Save intervals to {interval_output_file}")


if __name__ == "__main__":
    main()
