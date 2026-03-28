#
# Copyright (c) nexB Inc. and others. All rights reserved.
# ScanCode is a trademark of nexB Inc.
# SPDX-License-Identifier: Apache-2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/nexB/scancode-toolkit for support or download.
# See https://aboutcode.org for more information about nexB OSS projects.
#

"""
Dataset extraction and BIO labeling for ML-assisted required phrase prediction.

Extracts (rule_text, BIO_labels) pairs from existing ScanCode rules that have
``{{...}}`` required phrase markers. These labeled examples form the supervised
training dataset for the DeBERTa-v3 token classification model.

Usage::

    from licensedcode.ml_required_phrases.dataset import build_dataset
    dataset = build_dataset()
"""

import json
import os
from collections import Counter
from collections import defaultdict
from pathlib import Path

from licensedcode.models import load_rules
from licensedcode.models import rules_data_dir
from licensedcode.tokenize import REQUIRED_PHRASE_CLOSE
from licensedcode.tokenize import REQUIRED_PHRASE_OPEN
from licensedcode.tokenize import required_phrase_tokenizer
from licensedcode.tokenize import get_existing_required_phrase_spans


def get_normalized_tokens_for_ml(text, preserve_case=False):
    """
    Return a list of normalized token strings from ``text``, excluding
    required phrase marker tokens ({{ and }}).
    """
    markers = {REQUIRED_PHRASE_OPEN, REQUIRED_PHRASE_CLOSE}
    tokens = list(required_phrase_tokenizer(text=text, preserve_case=preserve_case))
    return [t for t in tokens if t not in markers]


def text_to_bio_labels(text):
    """
    Convert a rule ``text`` containing ``{{...}}`` required phrase markers into
    a list of (token, BIO_label) tuples.

    BIO labels:
      - ``B-REQ``: first token of a required phrase span
      - ``I-REQ``: continuation token of a required phrase span
      - ``O``: outside any required phrase span

    Returns a list of (token_str, label_str) tuples, or an empty list
    if the text has no valid tokens.
    """
    if not text or not text.strip():
        return []

    # Get required phrase spans (these are token-position-based Spans)
    try:
        req_spans = get_existing_required_phrase_spans(text)
    except Exception:
        return []

    # Build a set of token positions that are inside required phrase spans
    required_positions = set()
    for span in req_spans:
        for pos in range(span.start, span.end + 1):
            required_positions.add(pos)

    # Get the tokens (excluding markers)
    tokens = get_normalized_tokens_for_ml(text)
    if not tokens:
        return []

    # Assign BIO labels
    result = []
    for i, token in enumerate(tokens):
        if i in required_positions:
            # Check if this is the start of a span
            if i - 1 not in required_positions:
                result.append((token, 'B-REQ'))
            else:
                result.append((token, 'I-REQ'))
        else:
            result.append((token, 'O'))

    return result


def build_dataset(
    rules_data_dir=rules_data_dir,
    max_rules=None,
    verbose=False,
):
    """
    Build the ML training dataset from existing ScanCode rules for
    fine-tuning a HuggingFace Transformer model.

    Returns a dict with:
      - ``examples``: list of dicts (rule_id, license_expression, tokens, labels, text)
      - ``stats``: dataset stats
      - ``label_counts``: token label frequency
    """
    examples = []
    stats = {
        'total_rules_scanned': 0,
        'rules_with_markers': 0,
        'rules_without_markers': 0,
        'rules_skipped_errors': 0,
        'rules_skipped_flags': 0,
        'total_tokens': 0,
        'total_req_tokens': 0,
    }
    label_counts = Counter()

    rules = load_rules(rules_data_dir=rules_data_dir, with_checks=False)

    for rule in rules:
        stats['total_rules_scanned'] += 1

        if max_rules and stats['total_rules_scanned'] > max_rules:
            break

        # Skip rules that aren't suitable for training
        if getattr(rule, 'is_false_positive', False):
            stats['rules_skipped_flags'] += 1
            continue

        if getattr(rule, 'is_required_phrase', False):
            stats['rules_skipped_flags'] += 1
            continue

        if not rule.text or not rule.text.strip():
            stats['rules_skipped_flags'] += 1
            continue

        # Check if rule has required phrase markers
        has_markers = '{{' in rule.text and '}}' in rule.text

        if not has_markers:
            stats['rules_without_markers'] += 1
            continue

        # Extract BIO labels
        try:
            bio_pairs = text_to_bio_labels(rule.text)
        except Exception as e:
            stats['rules_skipped_errors'] += 1
            if verbose:
                print(f"Error processing {rule.identifier}: {e}")
            continue

        if not bio_pairs:
            stats['rules_skipped_errors'] += 1
            continue

        tokens = [t for t, _ in bio_pairs]
        labels = [l for _, l in bio_pairs]

        # Verify we actually have some REQ labels
        if not any(l != 'O' for l in labels):
            stats['rules_skipped_errors'] += 1
            continue

        stats['rules_with_markers'] += 1

        # Update stats
        stats['total_tokens'] += len(tokens)
        for label in labels:
            label_counts[label] += 1
            if label in ('B-REQ', 'I-REQ'):
                stats['total_req_tokens'] += 1

        example = {
            'rule_id': rule.identifier,
            'license_expression': getattr(rule, 'license_expression', ''),
            'text': rule.text,
            'tokens': tokens,
            'labels': labels,
        }
        examples.append(example)

        if verbose and stats['rules_with_markers'] % 100 == 0:
            print(f"  Processed {stats['rules_with_markers']} rules with markers...")

    if verbose:
        print(f"\nDataset build complete:")
        print(f"  Total rules scanned: {stats['total_rules_scanned']}")
        print(f"  Rules with markers (training data): {stats['rules_with_markers']}")
        print(f"  Rules without markers: {stats['rules_without_markers']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Required phrase tokens: {stats['total_req_tokens']}")

    return {
        'examples': examples,
        'stats': stats,
        'label_counts': dict(label_counts),
    }


def split_dataset(dataset, test_ratio=0.2, seed=42):
    """
    Split dataset into train and test sets, grouping by license expression
    to avoid data leakage between related rules.
    """
    import random
    rng = random.Random(seed)

    examples = dataset['examples']
    by_expression = defaultdict(list)
    for ex in examples:
        by_expression[ex['license_expression']].append(ex)

    expressions = list(by_expression.keys())
    rng.shuffle(expressions)

    total = len(examples)
    test_target = int(total * test_ratio)

    test_examples = []
    train_examples = []
    test_count = 0

    for expr in expressions:
        group = by_expression[expr]
        if test_count < test_target:
            test_examples.extend(group)
            test_count += len(group)
        else:
            train_examples.extend(group)

    return train_examples, test_examples


def save_dataset(dataset, output_path):
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2, default=str)


def load_dataset(input_path):
    with open(input_path) as f:
        return json.load(f)

