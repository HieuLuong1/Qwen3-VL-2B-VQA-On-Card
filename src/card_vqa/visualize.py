import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from card_vqa.config import resolve_image_path
from card_vqa.metrics import get_box, get_iou, is_correct


def visualize_wrong_samples(results, image_dir, max_samples=None):
    wrong = [record for record in results if not is_correct(record)]
    print(f"Incorrect samples: {len(wrong)}/{len(results)}")
    if max_samples is not None:
        wrong = wrong[:max_samples]

    for index, record in enumerate(wrong, start=1):
        image_path = resolve_image_path(record.get("image_id", ""), image_dir)
        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
        ax.imshow(image)

        gt_box = get_box(record.get("answer"))
        pred_box = get_box(record.get("prediction"))
        if gt_box:
            ax.add_patch(
                patches.Rectangle(
                    (gt_box[0] * width / 1000, gt_box[1] * height / 1000),
                    (gt_box[2] - gt_box[0]) * width / 1000,
                    (gt_box[3] - gt_box[1]) * height / 1000,
                    linewidth=2, edgecolor="green", facecolor="none",
                )
            )
        if pred_box:
            ax.add_patch(
                patches.Rectangle(
                    (pred_box[0] * width / 1000, pred_box[1] * height / 1000),
                    (pred_box[2] - pred_box[0]) * width / 1000,
                    (pred_box[3] - pred_box[1]) * height / 1000,
                    linewidth=2, edgecolor="red", linestyle="--", facecolor="none",
                )
            )

        ax.axis("off")
        ax.set_title(
            f"[WRONG #{index}] Group {record.get('group')}\\n"
            f"Q: {record.get('question')}\\nGT: {record.get('answer')} | "
            f"Pred: {record.get('prediction')}",
            fontsize=8,
        )
        plt.tight_layout()
        plt.show()
        plt.close(fig)
