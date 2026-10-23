import json
import os

from PIL import Image
from torch.utils.data import Dataset


SYSTEM_PROMPT = (
    "Answer the question directly and concisely with only the exact requested answer. "
    "Do not explain."
)


class VQADataset(Dataset):
    def __init__(self, json_path, image_dir, processor):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def _image_path(self, image_id):
        candidate = os.path.join(self.image_dir, str(image_id))
        if os.path.exists(candidate):
            return candidate
        if os.path.exists(str(image_id)):
            return str(image_id)
        raise FileNotFoundError(
            f"Image not found: image_id={image_id!r}; checked "
            f"{candidate!r} and {str(image_id)!r}"
        )

    def __getitem__(self, index):
        item = self.data[index]
        image_path = self._image_path(item["image_id"])
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": item["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(item["answer"]).strip()}],
            },
        ]
        return {
            "image": Image.open(image_path).convert("RGB"),
            "full_prompt": self.processor.apply_chat_template(
                messages, tokenize=False
            ),
        }


class VQACollator:
    def __init__(self, processor):
        self.processor = processor
        self.assistant_prefix_ids = processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    def __call__(self, batch):
        inputs = self.processor(
            text=[item["full_prompt"] for item in batch],
            images=[item["image"] for item in batch],
            padding=True,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        prefix_len = len(self.assistant_prefix_ids)

        for i in range(len(batch)):
            token_ids = inputs["input_ids"][i].tolist()
            start = next(
                (
                    j + prefix_len
                    for j in range(len(token_ids) - prefix_len + 1)
                    if token_ids[j:j + prefix_len] == self.assistant_prefix_ids
                ),
                None,
            )
            if start is None:
                labels[i, :] = -100
            else:
                labels[i, :start] = -100

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs
