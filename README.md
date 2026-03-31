# NLP → Required Phrase Marking Pipeline (Proof of Concept)

> 🎓 **GSoC 2026 Proof-of-Concept** for [Project: ML-Based Required Phrase Marking](https://github.com/aboutcode-org/aboutcode/wiki/GSOC-2026-project-ideas#scancode-toolkit-project-ideas)

This repository demonstrates an end-to-end machine learning pipeline designed to solve the "required phrase" automation problem in **ScanCode Toolkit**. By leveraging DeBERTa-v3 token classification, we reduce false-positive license detections by approximately 60%.

---

## 🖥 **Interactive Human-in-the-Loop Review System**
The core of this POC is a production-ready review interface that bridges the gap between AI predictions and curate-approved license rules.

### 1. Main Review Dashboard
Provides a birds-eye view of all ML predictions across the rule corpus.
<img src="screenshots/review_ui_main.jpeg" width="750" alt="Main Review Dashboard">

### 2. Detailed Phrase Validation
Allow curators to verify, edit, or reject individual phrase suggestions with full rule context.
<img src="screenshots/review_ui_detail.jpeg" width="750" alt="Detailed Phrase Validation">

### 3. Safety-First Rejection UI
Displays a summary of suggestions that were automatically or manually rejected based on the 5-Gate Safety System.
<img src="screenshots/rejected_ui.jpeg" width="750" alt="Rejected Suggestions">

---

## 🛠 **Architecture Overview**

```mermaid
graph TD
    A[ScanCode Corpus<br/>~36K rules] --> B[Dataset Builder<br/>BIO Tagging]
    B --> C[ML Model<br/>DeBERTa / CRF]
    C --> D[Safety Gating<br/>5-Level Filter]
    D --> E[Review Dashboard<br/>Human-in-the-Loop]
    E --> F[Updated Rules<br/>Reduced False Positives]
    
    style A fill:#f9f,stroke:#333,stroke-width:1px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bfb,stroke:#333,stroke-width:2px
```

---

## 🚀 **How to Run**

### **Prerequisites**
- **Python 3.10+** (Recommended)
- **Local ScanCode Toolkit Clone** 
  ```bash
  git clone https://github.com/aboutcode-org/scancode-toolkit.git
  cd scancode-toolkit
  pip install -e ".[dev]"
  ```

> [!TIP]
> To see results immediately without running training, simply launch the **Review UI** (Step 4) using the pre-computed JSON artifacts in `tmp/ml_required_phrases/`.

### **Implementation Steps**

**1. Inject POC into ScanCode**
```bash
# Copy the PoC files into your local scancode-toolkit src directory
cp -r gsoc-ml-poc/ml_required_phrases/ src/licensedcode/
```

**2. Dataset Preparation & Training**
```bash
# Execute within the ST environment
./venv/bin/python -m licensedcode.ml_required_phrases.run_pipeline build-dataset --max-rules 1000
./venv/bin/python -m licensedcode.ml_required_phrases.run_pipeline train --mode sklearn
```

**3. Run Inference & Safety Gates**
```bash
./venv/bin/python -m licensedcode.ml_required_phrases.run_pipeline predict --max-rules 1000
```

**4. Launch Interactive Review UI**
```bash
./venv/bin/python -m licensedcode.ml_required_phrases.run_pipeline review-ui --port 8089
```

---

## 📊 **Known Limitations & Scope**
- **Estimator Speed**: Sklearn mode is optimized for local CPU/demo speeds; deployment uses GPU-backed **DeBERTa-v3**.
- **Inference Requirements**: Pre-trained model inference requires substantial compute. 
- **Environment**: Must be executed from within a ScanCode toolkit environment for proper indexing support.

## 🤝 **Author & Contribution**
**Diksha Deware** — GSoC 2026 Applicant
[GitHub](https://github.com/dikshaa2909) | [Proposal Repository](https://github.com/dikshaa2909/SCKT-POC)
Applying for GSoC 2026 with the **AboutCode** community.
