# NLP → Required Phrase Marking Pipeline (Proof of Concept)

> GSoC 2026 proof-of-concept for [Project: ML-Based Required Phrase Marking](https://github.com/dikshaa2909/GSoC-2026-ScanCode-Proposal)

## Results

### Scenario 1: Unmarked Custom License

| Original ScanCode Output | ML Suggested Marker |
|---|---|
| ![Original Placeholder](screenshots/scenario1_original.png) | ![Fixed Placeholder](screenshots/scenario1_fixed.png) |

### Scenario 2: False Positive (URL Match)

| Original ScanCode Output | ML Suggested Marker |
|---|---|
| ![Original Placeholder](screenshots/scenario2_original.png) | ![Fixed Placeholder](screenshots/scenario2_fixed.png) |

### Scenario 3: Complex Multi-License Rule

| Original ScanCode Output | ML Suggested Marker |
|---|---|
| ![Original Placeholder](screenshots/scenario3_original.png) | ![Fixed Placeholder](screenshots/scenario3_fixed.png) |

## How to run

### Prerequisites

- Python 3.9+
- [ScanCode Toolkit](https://github.com/aboutcode-org/scancode-toolkit) (local clone required for integration)
  ```bash
  git clone https://github.com/aboutcode-org/scancode-toolkit.git
  cd scancode-toolkit
  pip install -e ".[dev]"
  ```

> **For reviewers:** The core pipeline orchestrator is in [`ml_required_phrases/run_pipeline.py`](ml_required_phrases/run_pipeline.py). To see the results without running the training or prediction jobs yourself, view the pre-computed outputs in `demo_results/` or simply launch the review UI using the included JSON data.

### Steps

**Step 1: Setup PoC in your ScanCode clone**
```bash
# Point the setup script to your local scancode-toolkit directory
./setup_local.sh ../scancode-toolkit
```

**Step 2: Build dataset and train the model**
```bash
# Navigate to the ST root with the newly copied PoC modules
cd ../scancode-toolkit

# Note: We use ST's virtual environment python to run the pipeline
./venv/bin/python src/licensedcode/ml_required_phrases/run_pipeline.py build-dataset --rules-dir demo_rules/
./venv/bin/python src/licensedcode/ml_required_phrases/run_pipeline.py train --mode sklearn
```

**Step 3: Run prediction and safety gates**
```bash
./venv/bin/python src/licensedcode/ml_required_phrases/run_pipeline.py predict --rules-dir demo_rules/
```
Predictions bounded by the 5-Gate Safety System are saved to `demo_results/suggestions.json`.

**Step 4 (optional): Open in Web Review UI**
```bash
./venv/bin/python src/licensedcode/ml_required_phrases/run_pipeline.py review-ui --port 8089
```

## Architecture

```
ScanCode Corpus             ML Inference Pipeline     Review System
┌──────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│ License Rule │     │                     │     │ Suggestion UI    │
│  ├─ Text     │────▶│  DeBERTa-v3 Model   │────▶│  ├─ Rule Context │
│  ├─ Flags    │     │  (BIO labeling,     │     │  ├─ AI Proposal  │
│  │  (intro,  │     │   confidence score, │     │  ├─ Approve/Deny │
│  │   fp,     │     │   safety gating,    │     │  │  Buttons      │
│  │   ...)    │     │   ignorable URLs)   │     │  └─ Output JSON  │
└──────────────┘     └─────────────────────┘     └──────────────────┘
```

## Known limitations (POC scope)

- The fast `sklearn` estimator is primarily geared for local demo speeds; production requires `deberta` mode running on GPU instances.
- Pre-trained DeBERTa inference requires substantial compute (falling back to simple vectors for standard desktop POC testing).
- Heuristic fallback logic isn't yet fully synchronized natively inside `licensedcode.index`.
- Strictly requires running from within the ScanCode toolkit fork environment (via `setup_local.sh`) to align with existing indexing utilities.

## Author

**Diksha Deware** — GSoC 2026 applicant
[GitHub](https://github.com/dikshaa2909) | Applying for GSoC 2026 with ScanCode Toolkit
