#
# Copyright (c) nexB Inc. and others. All rights reserved.
# ScanCode is a trademark of nexB Inc.
# SPDX-License-Identifier: Apache-2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/nexB/scancode-toolkit for support or download.
# See https://aboutcode.org for more information about nexB OSS projects.
#

"""
ML-assisted required phrase prediction for ScanCode license rules.

This package provides a machine learning pipeline to automatically suggest
required phrases for license rules that currently lack them. It extends
the existing heuristic-based required phrase tools in
``licensedcode.required_phrases`` with an ML-assisted prediction stage.

The pipeline:
1. dataset.py  - Extract BIO-labeled training data from existing annotated rules
2. train.py    - Train a token classification model (CRF / DeBERTa-v3)
3. predict.py  - Run predictions on unannotated rules
4. alignment.py - Align model predictions to ScanCode token positions
5. postfilter.py - Apply safety post-filters (ignorable overlap, genericity, etc.)
6. review.py   - CLI + minimal web UI for bulk review of suggestions
"""
