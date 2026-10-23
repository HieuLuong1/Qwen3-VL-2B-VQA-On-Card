import argparse
import json
import os
from pathlib import Path


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def add_config_argument(parser):
    parser.add_argument("--config", default="configs/default.json")
    return parser


def resolve_image_path(image_id, image_dir):
    candidate = os.path.join(image_dir, str(image_id))
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(str(image_id)):
        return str(image_id)
    return candidate


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
