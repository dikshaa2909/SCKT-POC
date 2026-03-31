# NLP → Required Phrase Marking Pipeline (Proof of Concept)

> **GSoC 2026 Proof-of-Concept** for [Project: ML-Based Required Phrase Marking](https://github.com/aboutcode-org/aboutcode/wiki/GSOC-2026-project-ideas#scancode-toolkit-project-ideas)

This repository demonstrates an end-to-end machine learning pipeline designed to resolve the "required phrase" automation problem in **ScanCode Toolkit**. By leveraging DeBERTa-v3 token classification and a 5-gate safety system, we achieve high-precision license rule enhancement.

---

##  **Interactive Human-in-the-Loop Review System**
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

##  **Key Features**
- **Token Classification (BIO tagging)**: High-accuracy sequence labeling for required phrases.
- **5-Gate Safety System**: Automated filtering of URLs, generic terms, and overlapping markers.
- **Confidence Tiling**: Dynamic bucketing into Auto-Apply, Review, and Rejected queues.
- **Atomic File Operations**: Safe, verified updates to core license rules with `.bak` backups.
- **Scalable Architecture**: Flexible design supporting both sklearn-based baselines and DeBERTa-v3 production models.

---

## **System Architecture**

```mermaid
flowchart LR
    A["<b>1. Dataset Generation</b><br/>Rule Corpus (36K files)<br/>├─ fast_dataset.py<br/>├─ Labeling Engine (BIO)<br/>└─ dataset.json"]
    
    B["<b>2. ML Training</b><br/>train.py (HF Trainer)<br/>├─ DeBERTa-v3 Target<br/>└─ model_checkpoint"]
    
    C["<b>3. Inference & Gating</b><br/>predict.py (Batch Loop)<br/>├─ alignment.py (Tokens)<br/>├─ postfilter.py (Safety)<br/>└─ Confidence Tiling"]
    
    D["<b>4. Human-in-the-Loop</b><br/>review.py (Web-based UI)<br/>├─ Auto-Apply Queue<br/>├─ Review Queue<br/>└─ Atomic writer.py"]

    A --> B
    B --> C
    C --> D

    style A fill:none,stroke:#000
    style B fill:none,stroke:#000
    style C fill:none,stroke:#000
    style D fill:#000,color:#fff
```

---

## **How to Run**

### **Prerequisites**
- **Python 3.10+**
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

## **Author & Contact**
**Diksha Deware** — GSoC 2026 Applicant
[GitHub](https://github.com/dikshaa2909) | [Proposal Repository](https://github.com/dikshaa2909/SCKT-POC)
Applying for GSoC 2026 with the **AboutCode** community.
