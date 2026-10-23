import re

import Levenshtein
import numpy as np


def compute_anls(gt, pred):
    distance = Levenshtein.distance(gt, pred)
    denominator = max(len(gt), len(pred))
    normalized_distance = distance / denominator if denominator else 0.0
    return 1.0 - normalized_distance if normalized_distance < 0.5 else 0.0


def get_box(text):
    match = re.search(
        r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", str(text)
    )
    return [int(match.group(i)) for i in range(1, 5)] if match else None


def get_iou(box1, box2):
    if not box1 or not box2:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def is_correct(record):
    group = record["group"]
    gt = str(record["answer"]).strip()
    pred = str(record["prediction"]).strip()

    if group == "A":
        return gt.lower() == pred.lower()
    if group == "B":
        return ("yes" in gt.lower()) == ("yes" in pred.lower())
    if group == "C":
        return pred[:1].upper() == gt[:1].upper()
    if group == "E":
        gt_box = get_box(gt) or record.get("bbox", [])
        pred_box = get_box(pred)
        return get_iou(gt_box, pred_box) >= 0.5
    if group == "H":
        gt_lower, pred_lower = gt.lower(), pred.lower()
        if gt_lower in ("yes", "no"):
            return ("yes" in pred_lower) == ("yes" in gt_lower)
        match = re.search(r"\b\d+\b", pred)
        return gt == (match.group(0) if match else "")
    return False


def calculate_metrics(results):
    stats = {
        "A": {"anls": [], "em": []},
        "B": {"gt": [], "pred": []},
        "C": {"correct": 0, "total": 0},
        "E": {"iou": [], "ap50": []},
        "H_yesno": {"correct": 0, "total": 0},
        "H_numeric": {"mae": [], "em": []},
    }

    for record in results:
        group = record["group"]
        gt = str(record["answer"]).strip()
        pred = str(record["prediction"]).strip()
        gt_lower, pred_lower = gt.lower(), pred.lower()

        if group == "A":
            stats["A"]["anls"].append(compute_anls(gt_lower, pred_lower))
            stats["A"]["em"].append(float(gt_lower == pred_lower))
        elif group == "B":
            stats["B"]["gt"].append(1 if "yes" in gt_lower else 0)
            stats["B"]["pred"].append(
                1 if "yes" in pred_lower else (0 if "no" in pred_lower else -1)
            )
        elif group == "C":
            match = re.search(r"\b([A-D])\b", pred.upper())
            predicted_choice = match.group(1) if match else pred[:1].upper()
            stats["C"]["total"] += 1
            stats["C"]["correct"] += int(predicted_choice == gt.upper())
        elif group == "E":
            gt_box = get_box(gt) or record.get("bbox", [])
            iou = get_iou(gt_box, get_box(pred))
            stats["E"]["iou"].append(iou)
            stats["E"]["ap50"].append(float(iou >= 0.5))
        elif group == "H":
            if gt_lower in ("yes", "no"):
                stats["H_yesno"]["total"] += 1
                predicted_yesno = (
                    "yes" if "yes" in pred_lower
                    else ("no" if "no" in pred_lower else "")
                )
                stats["H_yesno"]["correct"] += int(predicted_yesno == gt_lower)
            else:
                match = re.search(r"\b\d+\b", pred)
                predicted_number = int(match.group(0)) if match else None
                if gt.isdigit():
                    stats["H_numeric"]["mae"].append(
                        abs(int(gt) - predicted_number)
                        if predicted_number is not None else 10.0
                    )
                    stats["H_numeric"]["em"].append(
                        float(predicted_number == int(gt))
                    )
    return stats


def _mean(values):
    return float(np.mean(values)) if values else float("nan")


def print_metrics(stats):
    verification = (
        float(np.mean(np.array(stats["B"]["gt"]) == np.array(stats["B"]["pred"])))
        if stats["B"]["gt"] else float("nan")
    )
    choice = (
        stats["C"]["correct"] / stats["C"]["total"]
        if stats["C"]["total"] else float("nan")
    )
    h_yesno = (
        stats["H_yesno"]["correct"] / stats["H_yesno"]["total"]
        if stats["H_yesno"]["total"] else float("nan")
    )

    print("=" * 78)
    print(f"{'Task Group':<30} | {'Metric':<16} | {'Score'}")
    print("-" * 78)
    print(f"{'Group A: Direct Extraction':<30} | {'ANLS / EM':<16} | {_mean(stats['A']['anls'])*100:.2f}% / {_mean(stats['A']['em'])*100:.2f}%")
    print(f"{'Group E: Visual Grounding':<30} | {'mIoU / AP50':<16} | {_mean(stats['E']['iou']):.4f} / {_mean(stats['E']['ap50'])*100:.2f}%")
    print(f"{'Group B: Verification':<30} | {'Accuracy':<16} | {verification*100:.2f}%")
    print(f"{'Group C: Multiple Choice':<30} | {'Accuracy':<16} | {choice*100:.2f}%")
    print(f"{'Group H: Reasoning yes/no':<30} | {'Accuracy':<16} | {h_yesno*100:.2f}%")
    print(f"{'Group H: Reasoning numeric':<30} | {'MAE / EM':<16} | {_mean(stats['H_numeric']['mae']):.2f} / {_mean(stats['H_numeric']['em'])*100:.2f}%")
    print("=" * 78)
