import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

from card_vqa.config import load_config, resolve_image_path
from card_vqa.data import SYSTEM_PROMPT
from card_vqa.metrics import calculate_metrics, print_metrics
from card_vqa.model import load_adapter_for_eval


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned VQA adapter.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--visualize-errors", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    adapter_path = args.adapter_path or os.path.join(
        cfg["output_dir"], "best_lora_adapter"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_adapter_for_eval(
        cfg["model_id"], adapter_path, device
    )

    with open(cfg["test_json"], "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    print(f"Evaluating {len(test_data)} test examples...")
    with torch.no_grad():
        for item in tqdm(test_data):
            image_path = resolve_image_path(item["image_id"], cfg["image_dir"])
            if not os.path.exists(image_path):
                print(f"Skipping missing image: {image_path}")
                continue

            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": item["question"]},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[prompt],
                images=[Image.open(image_path).convert("RGB")],
                return_tensors="pt",
            ).to(device)
            output = model.generate(
                **inputs, max_new_tokens=int(cfg["max_new_tokens"]), do_sample=False
            )
            prediction = processor.batch_decode(
                [output[0][inputs.input_ids.shape[1]:]],
                skip_special_tokens=True,
            )[0].strip()
            result = dict(item)
            result["prediction"] = prediction
            results.append(result)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    result_path = os.path.join(cfg["output_dir"], "test_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    stats = calculate_metrics(results)
    print_metrics(stats)
    print(f"Saved predictions to: {result_path}")

    if args.visualize_errors:
        from card_vqa.visualize import visualize_wrong_samples
        visualize_wrong_samples(results, cfg["image_dir"])


if __name__ == "__main__":
    main()
