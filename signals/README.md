# Signals

The `signals` folder contains scripts and resources for implementing rule-based and model-based filters for dataset curation, as described in the [Big-Math paper](https://alon-albalak.github.io/images/Big_MATH.pdf). These filtering techniques are critical for ensuring the quality, diversity, and relevance of the dataset for reinforcement learning.

---

## 📂 Contents

### Rule-based filters
- `add_empty_boxed_signal.py`
- `add_hyperlink_signal.py`
- `add_language_signal.py`
- `add_multipartq_signal.py`
- `add_multiple_choice_signal.py`
- `add_proof_signal.py`
- `add_true_false_signal.py`
- `add_yes_no_signal.py`

### Deduplication
- `add_semdedup_signal.py`

### Model-based signals
- `model_based_signals.py`

### Solve rate
- `rollouts_based_signals/example_solve_rate_script.sh`


## 🚀 Getting Started

### Prerequisites
- Python3.10+
- Install dependencies:
```bash
pip install -r requirements.txt
```

## 🛠 Usage

### Rule-based signals
Rule-based signals are written in separate files, but share the same use pattern. Each signal requires only a single input variable, `dataset_path`. The script will automatically update the source dataset with a new column.

This is a simple example that will detect whether the `problem` or `solution` columns have a hyperlink, adding 2 new columns, `has_hyperlink_problem` and `has_hyperlink_solution`:
```bash
DATASET_PATH=<your HF dataset path here>
python3 add_hyperlink_signal.py --dataset_path ${DATASET_PATH}
```

### Deduplication
We use SemDeDup to mark duplicated problems. Just like the rule-based signals, this requires only a single input variable, `dataset_path`. The script, by default, uses a similarity threshold of 0.5, but that can easily be adjusted by adding/removing values from the `EPSILONS` variable.
```bash
DATASET_PATH=<your HF dataset path here>
python3 add_semdedup_signal.py --dataset_path ${DATASET_PATH}
```

### Model-based signals
The model-based signals are all included in a single file for efficiency when running multiple filters. Our scripts make use of [SGLang](https://github.com/sgl-project/sglang) for fast inference.

Model-based signals require a bit more specification than rule-based signals. Here, we need to specify:
- `model_name`: The name of a local or HF model
- `output_model_name`: The name of model you want to use in the signal's column name. For example, the below code uses `--output_model_name "llama3-70b"` and will add the multiple choice signal in a new column: "multiple_choice_llama3-70b"
- `dataset_name`: A path to HF or local dataset
- `dataset_name_outputs`: Optional - The path to save the updated dataset to. Will save to `dataset_name` when not specified
- `save_folder`: Local directory to save intermediate outputs to
- `save_name`: The name of the file to save intermediate outputs to
- `tp`: tensor parallelism
- Signal options: `--multiple_choice`, `--proof`, `--yes_no`, `--true_false`, `--multiple_part`

The below example runs all model-based signals using Llama-3.1-70B:
```bash
OUTPUT_DATASET_NAME=<your dataset name here>

python3 model_based_filters.py \
    --model_name "meta-llama/Meta-Llama-3.1-70B-Instruct"\
    --output_model_name "llama3-70b"\
    --dataset_name "SynthLabsAI/Big-Math-RL-Verified"\
    --dataset_name_outputs "${OUTPUT_DATASET_NAME}"\
    --save_folder "signal_outputs"\
    --save_name "model_based_signals"\
    --tp 8\
    --multiple_choice\
    --proof\
    --yes_no\
    --true_false\
    --multiple_part\
    > model_based_signal_outputs.txt 2>&1
```

### Solve rate
See `rollouts_based_signals/example_solve_rate_script.sh` for example usage.
