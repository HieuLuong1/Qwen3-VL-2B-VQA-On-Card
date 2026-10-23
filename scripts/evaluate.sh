#!/usr/bin/env bash
set -e
python -m card_vqa.evaluate --config configs/default.json --visualize-errors
