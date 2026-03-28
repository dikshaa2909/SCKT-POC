#
# Copyright (c) nexB Inc. and others. All rights reserved.
# ScanCode is a trademark of nexB Inc.
# SPDX-License-Identifier: Apache-2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/nexB/scancode-toolkit for support or download.
# See https://aboutcode.org for more information about nexB OSS projects.
#

"""
Alignment of model-predicted BIO spans to ScanCode token positions.

Converts raw model output (BIO labels with confidence scores) into
deterministic ScanCode token-position spans, aligned with the original
rule text. This module ensures that subword or model token positions
map correctly back to the ScanCode tokenizer's token positions.

Usage::

    from licensedcode.ml_required_phrases.alignment import align_predictions_to_spans
    spans = align_predictions_to_spans(tokens, pred_labels, pred_probs)
"""

from licensedcode.spans import Span


def bio_decode(pred_labels, pred_probs=None):
    """
    Decode BIO-labeled predictions into contiguous candidate spans.

    Args:
        pred_labels: list of predicted BIO label strings
        pred_probs: optional list of prediction probability dicts
                    keyed by label name, e.g. [{'B-REQ': 0.9, 'I-REQ': 0.05, 'O': 0.05}, ...]

    Returns a list of dicts, each with:
      - ``start``: start token index
      - ``end``: end token index (inclusive)
      - ``token_probs``: list of per-token probabilities for the predicted label
    """
    spans = []
    current_start = None
    current_probs = []

    for i, label in enumerate(pred_labels):
        prob = 1.0
        if pred_probs and i < len(pred_probs):
            prob = pred_probs[i].get(label, pred_probs[i].get('max_prob', 1.0))

        if label == 'B-REQ':
            # Close previous span if open
            if current_start is not None:
                spans.append({
                    'start': current_start,
                    'end': i - 1,
                    'token_probs': current_probs,
                })
            current_start = i
            current_probs = [prob]

        elif label == 'I-REQ':
            if current_start is not None:
                current_probs.append(prob)
            else:
                # Orphan I-REQ: treat as B-REQ
                current_start = i
                current_probs = [prob]

        else:  # O
            if current_start is not None:
                spans.append({
                    'start': current_start,
                    'end': i - 1,
                    'token_probs': current_probs,
                })
                current_start = None
                current_probs = []

    # Close final span
    if current_start is not None:
        spans.append({
            'start': current_start,
            'end': len(pred_labels) - 1,
            'token_probs': current_probs,
        })

    return spans


def score_span(span_dict):
    """
    Compute confidence score for a candidate span.

    Uses the minimum probability across all tokens in the span as the
    conservative confidence estimate. Also computes mean confidence.

    Args:
        span_dict: dict with 'token_probs' list

    Returns updated span_dict with added keys:
      - ``confidence``: minimum token probability (conservative)
      - ``mean_confidence``: average token probability
      - ``span_length``: number of tokens in span
    """
    probs = span_dict.get('token_probs', [1.0])
    span_dict['confidence'] = min(probs) if probs else 0.0
    span_dict['mean_confidence'] = sum(probs) / len(probs) if probs else 0.0
    span_dict['span_length'] = span_dict['end'] - span_dict['start'] + 1
    return span_dict


def align_predictions_to_spans(tokens, pred_labels, pred_probs=None):
    """
    Full alignment pipeline: BIO decode → score → produce ScanCode Spans.

    Args:
        tokens: list of token strings (ScanCode-normalized)
        pred_labels: list of predicted BIO label strings
        pred_probs: optional list of per-token probability dicts

    Returns a list of dicts, each with:
      - ``span``: licensedcode.spans.Span object
      - ``tokens``: list of token strings in the span
      - ``text``: joined text of the span tokens
      - ``confidence``: conservative confidence score (min prob)
      - ``mean_confidence``: average confidence score
      - ``span_length``: number of tokens
    """
    if len(tokens) != len(pred_labels):
        raise ValueError(
            f"Token count ({len(tokens)}) != label count ({len(pred_labels)})"
        )

    # Step 1: BIO decode to raw spans
    raw_spans = bio_decode(pred_labels, pred_probs)

    # Step 2: Score each span
    scored_spans = [score_span(s) for s in raw_spans]

    # Step 3: Convert to ScanCode Span objects with metadata
    aligned = []
    for s in scored_spans:
        start = s['start']
        end = s['end']

        span_tokens = tokens[start:end + 1]

        aligned.append({
            'span': Span(start, end),
            'tokens': span_tokens,
            'text': ' '.join(span_tokens),
            'confidence': s['confidence'],
            'mean_confidence': s['mean_confidence'],
            'span_length': s['span_length'],
            'start': start,
            'end': end,
        })

    return aligned


def validate_span_alignment(span_info, rule_text_tokens):
    """
    Validate that a predicted span aligns correctly with the rule text tokens.

    Returns True if the span tokens match the original rule text tokens
    at the predicted positions.
    """
    start = span_info['start']
    end = span_info['end']

    if end >= len(rule_text_tokens):
        return False

    expected_tokens = rule_text_tokens[start:end + 1]
    return expected_tokens == span_info['tokens']
