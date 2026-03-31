# ScanCode Core Integration Preview

This directory contains the files and visual evidence showing how the ML-Based Required Phrase Marking pipeline integrates directly into the ScanCode Toolkit core.

## Integration Files
- **cli/required_phrases.py**: Modified core module with new `review-required-phrases-ml` and `apply-approved-required-phrases-ml` commands.
- **config/setup.cfg**: Configuration for entry points and dependencies.
- **cli/__init__.py**: Module initialization for the new ML package.

## Execution Workflow (Visual Evidence)
Located in the [screenshots/](./screenshots/) directory:

### 1. CLI Pipeline Execution
- **[Phase A: Building Dataset](./screenshots/phase_a.jpeg)**: Scanning rules to extract training pairs.
- **[Phase B: Training Logic](./screenshots/phase_b.jpeg)**: Rapid baseline training with performance metrics.
- **[Phase C-E: Prediction Loop](./screenshots/phase_ce.jpeg)**: Running inference with safety gating.
- **[Pipeline Complete](./screenshots/pipeline_complete.jpeg)**: Final artifact generation summary.

### 2. Human-in-the-Loop Review UI
- **[Main Review Dashboard](./screenshots/review_ui_main.jpeg)**: High-level overview of ML-suggested phrases.
- **[Detailed Phrase Validation](./screenshots/review_ui_detail.jpeg)**: Side-by-side view for rule verification.
- **[Rejected Suggestions](./screenshots/rejected_ui.jpeg)**: View of phrases filtered by safety criteria or human review.

These modifications allow maintainers to run the ML pipeline natively:
```bash
python -m licensedcode.ml_required_phrases.run_pipeline run-all --max-rules 1000
```
