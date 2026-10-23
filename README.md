# Card VQA with Qwen3-VL and LoRA

A modular implementation of a card/document Visual Question Answering (VQA) pipeline using
`Qwen/Qwen3-VL-2B-Instruct` with LoRA fine-tuning. The project evaluates five task groups:

- **A — Direct extraction:** ANLS and exact match (EM)
- **B — Verification:** yes/no accuracy
- **C — Multiple choice:** accuracy
- **E — Visual grounding:** mean IoU and AP@0.5
- **H — Reasoning:** yes/no accuracy, numeric MAE and exact match

This repository contains code only. Dataset files, trained adapters, and generated predictions are not included.

## Repository structure

```text
card-vqa-qwen3-vl/
├── configs/
│   └── default.json
├── src/
│   └── card_vqa/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── model.py
│       ├── metrics.py
│       ├── train.py
│       ├── evaluate.py
│       └── visualize.py
├── scripts/
│   ├── train.sh
│   └── evaluate.sh
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Environment

Use a GPU runtime with a PyTorch build compatible with your CUDA driver. Install PyTorch separately if needed, then:

```bash
pip install -r requirements.txt
pip install -e .
```

The model is downloaded from Hugging Face. Set `HF_TOKEN` if your environment needs authenticated Hugging Face access.

## 2. Dataset format

The JSON files are expected to contain a list of records. Each record should include:

```json
{
  "image_id": "relative/or/absolute/path/to/image.jpg",
  "question": "Example question?",
  "answer": "Ground-truth answer",
  "group": "A"
}
```

Records for visual grounding may additionally contain `bbox`. The current box parser expects answer/prediction coordinates in the form `[x1, y1, x2, y2]`, normalized to a 0–1000 coordinate scale, matching the original notebook's evaluation logic.

Keep the dataset private or follow its license and data-sharing requirements. Do not commit private images, annotations, or credentials.

## 3. Configure paths

Edit `configs/default.json` or provide CLI overrides. The default config uses placeholder paths, not the original Kaggle mount paths.

## 4. Train

```bash
python -m card_vqa.train --config configs/default.json
```

The best adapter and processor are saved under `output_dir/best_lora_adapter`.

## 5. Evaluate

```bash
python -m card_vqa.evaluate --config configs/default.json \
  --adapter-path /path/to/best_lora_adapter
```

Evaluation writes predictions to JSON and prints task-group metrics. Optional qualitative error visualizations can be enabled with `--visualize-errors`.

## Notes and limitations

- The original notebook used Qwen3-VL-2B-Instruct, LoRA rank 16, and a small test set (117 evaluated examples in the logged run). The reported results are from that run and should not be interpreted as a guarantee of reproducibility.
- The notebook's original metrics use simple string rules for yes/no and multiple-choice answers. Inspect these rules against your annotation conventions before presenting results.
- The grounding metric labelled `AP50` in the notebook is the fraction of examples with IoU ≥ 0.5, not a full object-detection average precision calculation.
- The notebook monkey-patched `peft.import_utils.is_torchao_available` to work around an environment compatibility issue. This workaround is retained as an opt-in compatibility flag in the config; remove it if your environment does not need it.
- Do not upload adapter weights unless their size, model license, dataset license, and company/internship confidentiality rules permit redistribution.
