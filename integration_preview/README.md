# ScanCode Toolkit Core Integration Preview

This directory contains the files and visual evidence showing how the ML-Based Required Phrase Marking pipeline integrates directly into the ScanCode Toolkit core.

## 🎓 GSoC 2026 Project Idea
🔗 [Project: AI-powered Required Phrase Detection](https://github.com/aboutcode-org/aboutcode/wiki/GSOC-2026-project-ideas#scancode-toolkit-project-ideas)

## 🖥 Human-in-the-Loop Review UI
The interactive review system allows maintainers to validate ML predictions before they are applied.

- **[Main Review Dashboard](./screenshots/review_ui_main.jpeg)**: High-level overview of ML predictions.
- **[Detailed Phrase Validation](./screenshots/review_ui_detail.jpeg)**: Granular side-by-side view for rule verification.
- **[Rejected Suggestions](./screenshots/rejected_ui.jpeg)**: Preview of suggested phrases filtered by rejection criteria.

## 🛠 Integration Components
- **cli/required_phrases.py**: Modified core module with `review-required-phrases-ml` and `apply-approved-required-phrases-ml` commands.
- **config/setup.cfg**: Configuration for entry points and dependencies.
- **cli/init.py**: Module initialization for the new ML package.
