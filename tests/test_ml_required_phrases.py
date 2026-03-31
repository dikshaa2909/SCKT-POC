#
# Copyright (c) nexB Inc. and others. All rights reserved.
# ScanCode is a trademark of nexB Inc.
# SPDX-License-Identifier: Apache-2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/nexB/scancode-toolkit for support or download.
# See https://aboutcode.org for more information about nexB OSS projects.
#

"""
Tests for ML-assisted required phrase prediction pipeline (DeBERTa-v3).

Tests cover:
1. Dataset extraction and BIO labeling correctness
2. Token alignment and span conversion correctness
3. Safety filter correctness (URL/path/ignorable protections)
"""

import json
import os
import tempfile
import pytest

from licensedcode.spans import Span


class TestBIOLabeling:
    """Test BIO label extraction from rule texts with {{...}} markers."""

    def test_simple_required_phrase(self):
        from licensedcode.ml_required_phrases.dataset import text_to_bio_labels
        text = 'This is {{MIT License}} text'
        pairs = text_to_bio_labels(text)
        tokens = [t for t, _ in pairs]
        labels = [l for _, l in pairs]

        assert len(tokens) > 0
        assert 'B-REQ' in labels
        # 'mit' should be B-REQ, 'license' should be I-REQ
        for i, (t, l) in enumerate(pairs):
            if t == 'mit':
                assert l == 'B-REQ', f"Expected B-REQ for 'mit', got {l}"
            elif t == 'license' and i > 0 and pairs[i-1][1] in ('B-REQ', 'I-REQ'):
                assert l == 'I-REQ', f"Expected I-REQ for 'license', got {l}"

    def test_multiple_required_phrases(self):
        from licensedcode.ml_required_phrases.dataset import text_to_bio_labels
        text = '{{GNU General}} Public {{License v3}}'
        pairs = text_to_bio_labels(text)
        labels = [l for _, l in pairs]

        b_count = labels.count('B-REQ')
        assert b_count == 2, f"Expected 2 B-REQ labels, got {b_count}"

    def test_no_markers(self):
        from licensedcode.ml_required_phrases.dataset import text_to_bio_labels
        text = 'This is a simple text without markers'
        pairs = text_to_bio_labels(text)
        labels = [l for _, l in pairs]

        assert all(l == 'O' for l in labels), "All labels should be O"


class TestBIODecode:
    """Test BIO label decoding to spans."""

    def test_single_span(self):
        from licensedcode.ml_required_phrases.alignment import bio_decode
        labels = ['O', 'B-REQ', 'I-REQ', 'I-REQ', 'O']
        spans = bio_decode(labels)

        assert len(spans) == 1
        assert spans[0]['start'] == 1
        assert spans[0]['end'] == 3

    def test_multiple_spans(self):
        from licensedcode.ml_required_phrases.alignment import bio_decode
        labels = ['B-REQ', 'I-REQ', 'O', 'B-REQ', 'O']
        spans = bio_decode(labels)

        assert len(spans) == 2
        assert spans[0]['start'] == 0
        assert spans[0]['end'] == 1
        assert spans[1]['start'] == 3
        assert spans[1]['end'] == 3


class TestSpanScoring:
    """Test confidence scoring of spans."""

    def test_score_span(self):
        from licensedcode.ml_required_phrases.alignment import score_span
        span = {
            'start': 0, 'end': 2,
            'token_probs': [0.95, 0.88, 0.92],
        }
        scored = score_span(span)

        assert scored['confidence'] == 0.88  # min
        assert abs(scored['mean_confidence'] - 0.9167) < 0.01
        assert scored['span_length'] == 3


class TestAlignPredictions:
    """Test full alignment pipeline."""

    def test_align_simple(self):
        from licensedcode.ml_required_phrases.alignment import align_predictions_to_spans
        tokens = ['gnu', 'general', 'public', 'license', 'v3']
        labels = ['B-REQ', 'I-REQ', 'I-REQ', 'I-REQ', 'O']
        probs = [
            {'B-REQ': 0.92, 'I-REQ': 0.05, 'O': 0.03, 'max_prob': 0.92},
            {'B-REQ': 0.05, 'I-REQ': 0.90, 'O': 0.05, 'max_prob': 0.90},
            {'B-REQ': 0.03, 'I-REQ': 0.88, 'O': 0.09, 'max_prob': 0.88},
            {'B-REQ': 0.02, 'I-REQ': 0.85, 'O': 0.13, 'max_prob': 0.85},
            {'B-REQ': 0.05, 'I-REQ': 0.10, 'O': 0.85, 'max_prob': 0.85},
        ]

        aligned = align_predictions_to_spans(tokens, labels, probs)

        assert len(aligned) == 1
        assert aligned[0]['text'] == 'gnu general public license'
        assert aligned[0]['span'] == Span(0, 3)
        assert aligned[0]['confidence'] == 0.85  # min of token probs


class TestSafetyFilters:
    """Test safety post-filters."""

    def test_filter_ignorable_overlap_url(self):
        from licensedcode.ml_required_phrases.postfilter import filter_ignorable_overlap

        class MockRule:
            ignorable_urls = ['http://www.apache.org/licenses/LICENSE-2.0']
            referenced_filenames = []

        span_info = {
            'tokens': ['http', 'www', 'apache', 'org', 'licenses'],
            'start': 0, 'end': 4,
        }

        result = filter_ignorable_overlap(span_info, MockRule())
        assert not result.passed, "Should reject span overlapping with URL"

    def test_filter_genericity_stopwords(self):
        from licensedcode.ml_required_phrases.postfilter import filter_genericity

        # Span with mostly stopwords
        span_info = {'tokens': ['the', 'a', 'is', 'of'], 'start': 0, 'end': 3}
        result = filter_genericity(span_info, min_non_stopword_tokens=2)
        assert not result.passed, "Should reject stopword-heavy spans"


class TestClassifySuggestion:
    """Test confidence-based suggestion bucket classification."""

    def test_high_confidence_auto_apply(self):
        from licensedcode.ml_required_phrases.postfilter import classify_suggestion

        class MockRule:
            is_false_positive = False
            is_required_phrase = False
            skip_for_required_phrase_generation = False
            text = 'Some license text without markers'
            ignorable_urls = []
            referenced_filenames = []

        span_info = {
            'tokens': ['apache', 'license', 'version'],
            'start': 0, 'end': 2,
            'confidence': 0.95,
        }

        result = classify_suggestion(span_info, MockRule())
        assert result['bucket'] == 'auto_apply'


class TestHuggingfaceAlignment:
    """Test conversion of word boundaries to subwords alignment logic."""

    def test_align_labels_with_tokens(self):
        try:
            from licensedcode.ml_required_phrases.train import align_labels_with_tokens
        except ImportError:
            pytest.skip("PyTorch/Transformers not installed yet")
            
        labels = ["O", "B-REQ", "I-REQ", "O"]
        word_ids = [None, 0, 1, 1, 2, 3, None] 
        # label 0: 'O' -> index 0 (O)
        # label 1: 'B-REQ' -> index 1 (B-REQ). Next subword gets -100
        # label 2: 'I-REQ' -> index 2 (I-REQ)
        # label 3: 'O' -> index 0 (O)
        
        aligned = align_labels_with_tokens(labels, word_ids)
        assert aligned[0] == -100  # padding/special
        assert aligned[1] == 0     # O
        assert aligned[2] == 1     # B-REQ
        assert aligned[3] == -100  # subsequent subword ignore
        assert aligned[4] == 2     # I-REQ
        assert aligned[5] == 0     # O
        assert aligned[6] == -100  # padding/special
