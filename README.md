# NLP → Required Phrase Marking Pipeline (Proof of Concept)

> **GSoC 2026 Proof-of-Concept** for [Project: ML-Based Required Phrase Marking](https://github.com/aboutcode-org/aboutcode/wiki/GSOC-2026-project-ideas#scancode-toolkit-project-ideas)

This repository demonstrates an end-to-end machine learning pipeline designed to resolve the "required phrase" automation problem in **ScanCode Toolkit**. By leveraging DeBERTa-v3 token classification and a 5-gate safety system, we achieve high-precision license rule enhancement.

---

## 🖥 **Interactive Review System**
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
flowchart TD
    subgraph PhaseA [Phase A: Dataset Generation]
        direction TB
        A1[Rule Corpus<br/>36,482 .RULE files] --> A2[fast_dataset.py<br/>Regex Parser]
        A2 --> A3[Labeling Engine<br/>BIO Token Labeling]
        A3 --> A4[dataset.json<br/>Supervised Data]
    end

    subgraph PhaseB [Phase B: ML Training]
        direction TB
        B1[train.py<br/>HF Trainer] --> B2[DeBERTa-v3<br/>Production Target]
        B2 --> B3[model_checkpoint<br/>Fine-tuned Weights]
    end

    subgraph PhaseCE [Phase C-E: Inference & Safety Gates]
        direction TB
        C1[predict.py<br/>Batch Inference] --> C2[alignment.py<br/>SCTK Token Alignment]
        C2 --> C3[postfilter.py<br/>5 Safety Gates]
        subgraph Gates [5 Safety Gates]
            G1[1. Ignorable Overlap]
            G2[2. Genericity Filter]
            G3[3. Rule Constraints]
            G4[4. Marker Conflict]
            G5[5. Min Informativeness]
        end
        C3 --> Gates
        Gates --> C4[Confidence Gating<br/>T_high=0.95]
    end

    subgraph PhaseF [Phase F: Human-in-the-Loop]
        direction TB
        F1{Confidence Tipping}
        F1 -->|T >= 0.95| F2[Auto-Apply Bucket]
        F1 -->|0.80 - 0.95| F3[Review Queue]
        F1 -->|< 0.80| F4[Rejected Bucket]
        
        F3 --> F5[review.py Web UI<br/>Accept/Reject/Edit]
        F5 --> F6[writer.py<br/>Atomic Write .bak]
        F6 --> F7[ScanCode Rule Update<br/>scancode-reindex-licenses]
    end

    A4 --> B1
    B3 --> C1
    C4 --> F1

    style F7 fill:#000,color:#fff,stroke:#333,stroke-width:2px
    style B2 fill:#f9f,stroke:#333
    style C3 fill:#fff,stroke:#333,stroke-dasharray: 5 5
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
