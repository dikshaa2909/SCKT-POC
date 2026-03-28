#
# Copyright (c) nexB Inc. and others. All rights reserved.
# ScanCode is a trademark of nexB Inc.
# SPDX-License-Identifier: Apache-2.0
#

"""
Fast dataset builder that reads RULE files directly without full ScanCode
index initialization. This is much faster for prototyping/demo.
"""

import json
import os
import re
import yaml
from collections import Counter, defaultdict
from pathlib import Path


# Minimal tokenizer that matches ScanCode's behavior
def simple_tokenize(text):
    """Tokenize text into word tokens (lowercased, alphanumeric)."""
    return [t.lower() for t in re.findall(r'[a-zA-Z0-9]+', text) if len(t) > 0]


def parse_rule_file(filepath):
    """
    Parse a .RULE file: extract YAML frontmatter and text body.
    Returns (metadata_dict, text_body) or (None, None) on failure.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception:
        return None, None

    if not content.startswith('---'):
        return {}, content.strip()

    parts = content.split('---', 2)
    if len(parts) < 3:
        return {}, content.strip()

    yaml_part = parts[1].strip()
    text_part = parts[2].strip()

    metadata = {}
    for line in yaml_part.split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip()
            if key == 'is_false_positive': metadata[key] = (val.lower() == 'yes')
            elif key == 'is_required_phrase': metadata[key] = (val.lower() == 'yes')
            elif key == 'is_license_intro': metadata[key] = (val.lower() == 'yes')
            elif key == 'license_expression': metadata[key] = val
            elif key == 'skip_for_required_phrase_generation': metadata[key] = (val.lower() == 'yes')

    return metadata, text_part


def text_to_bio_labels_fast(text):
    """
    Convert text with {{...}} markers into (tokens, labels) lists.
    Fast version that does direct string parsing.
    """
    if not text or '{{' not in text or '}}' not in text:
        return [], []

    # Split text by {{ and }} markers
    tokens = []
    labels = []
    in_required = False

    # Process text: remove markers, track which tokens are inside markers
    cleaned_parts = []
    i = 0
    marker_ranges = []  # (start_char, end_char) of text inside markers

    while i < len(text):
        if text[i:i+2] == '{{':
            marker_start = len(''.join(cleaned_parts))
            i += 2
            in_required = True
        elif text[i:i+2] == '}}':
            marker_end = len(''.join(cleaned_parts))
            marker_ranges.append((marker_start, marker_end))
            i += 2
            in_required = False
        else:
            cleaned_parts.append(text[i])
            i += 1

    cleaned_text = ''.join(cleaned_parts)

    # Tokenize the cleaned text
    all_tokens = []
    for match in re.finditer(r'[a-zA-Z0-9]+', cleaned_text):
        token_text = match.group().lower()
        token_start = match.start()
        token_end = match.end()
        all_tokens.append((token_text, token_start, token_end))

    if not all_tokens:
        return [], []

    # Label each token based on whether it falls inside a marker range
    tokens = []
    labels_list = []

    for token_text, t_start, t_end in all_tokens:
        is_in_required = False
        for m_start, m_end in marker_ranges:
            if t_start >= m_start and t_end <= m_end:
                is_in_required = True
                break

        tokens.append(token_text)

        if is_in_required:
            # Check if previous token was also in required (I-REQ vs B-REQ)
            if labels_list and labels_list[-1] in ('B-REQ', 'I-REQ'):
                # Check continuity: was previous token in the same marker range?
                prev_start = all_tokens[len(labels_list) - 1][1]
                same_range = False
                for m_start, m_end in marker_ranges:
                    if prev_start >= m_start and t_end <= m_end:
                        same_range = True
                        break
                if same_range:
                    labels_list.append('I-REQ')
                else:
                    labels_list.append('B-REQ')
            else:
                labels_list.append('B-REQ')
        else:
            labels_list.append('O')

    return tokens, labels_list


def build_fast_dataset(rules_dir=None, max_rules=None, verbose=True):
    """
    Build training dataset by directly reading RULE files.
    Much faster than using load_rules() since we skip the full index.
    """
    if rules_dir is None:
        rules_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'rules'
        )

    rules_dir = Path(rules_dir)
    if not rules_dir.exists():
        raise FileNotFoundError(f"Rules directory not found: {rules_dir}")

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

    rule_files = sorted(rules_dir.glob('*.RULE'))
    if verbose:
        print(f"  Found {len(rule_files)} .RULE files")

    for rule_file in rule_files:
        stats['total_rules_scanned'] += 1

        if max_rules and stats['total_rules_scanned'] > max_rules:
            break

        metadata, text = parse_rule_file(rule_file)
        if metadata is None or not text:
            stats['rules_skipped_errors'] += 1
            continue

        # Skip rules with certain flags
        if metadata.get('is_false_positive'):
            stats['rules_skipped_flags'] += 1
            continue

        if metadata.get('is_required_phrase'):
            stats['rules_skipped_flags'] += 1
            continue

        has_markers = '{{' in text and '}}' in text

        if not has_markers:
            stats['rules_without_markers'] += 1
            continue

        # Extract BIO labels
        try:
            tokens, labels = text_to_bio_labels_fast(text)
        except Exception as e:
            stats['rules_skipped_errors'] += 1
            continue

        if not tokens or not labels:
            stats['rules_skipped_errors'] += 1
            continue

        # Verify we have some REQ labels
        if not any(l != 'O' for l in labels):
            stats['rules_skipped_errors'] += 1
            continue

        stats['rules_with_markers'] += 1
        stats['total_tokens'] += len(tokens)

        for label in labels:
            label_counts[label] += 1
            if label in ('B-REQ', 'I-REQ'):
                stats['total_req_tokens'] += 1

        example = {
            'rule_id': rule_file.name,
            'license_expression': metadata.get('license_expression', ''),
            'text': text,
            'tokens': tokens,
            'labels': labels,
        }
        examples.append(example)

        if verbose and stats['total_rules_scanned'] % 500 == 0:
            print(f"  Scanned {stats['total_rules_scanned']} rules (found {stats['rules_with_markers']} with markers)...")

    if verbose:
        print(f"\n  Dataset build complete:")
        print(f"    Total rules scanned: {stats['total_rules_scanned']}")
        print(f"    Rules with markers (training data): {stats['rules_with_markers']}")
        print(f"    Rules without markers: {stats['rules_without_markers']}")
        print(f"    Skipped (flags): {stats['rules_skipped_flags']}")
        print(f"    Skipped (errors): {stats['rules_skipped_errors']}")
        print(f"    Total tokens: {stats['total_tokens']}")
        print(f"    Required phrase tokens: {stats['total_req_tokens']}")

    return {
        'examples': examples,
        'stats': stats,
        'label_counts': dict(label_counts),
    }


class SimpleRule:
    """Minimal Rule-like object for prediction phase."""

    def __init__(self, filepath, metadata, text):
        self.identifier = filepath.name if isinstance(filepath, Path) else os.path.basename(filepath)
        self.text = text
        self.license_expression = metadata.get('license_expression', '')
        self.is_false_positive = metadata.get('is_false_positive', False)
        self.is_required_phrase = metadata.get('is_required_phrase', False)
        self.is_license_intro = metadata.get('is_license_intro', False)
        self.skip_for_required_phrase_generation = metadata.get(
            'skip_for_required_phrase_generation', False
        )
        self.ignorable_urls = metadata.get('ignorable_urls', []) or []
        self.referenced_filenames = metadata.get('referenced_filenames', []) or []


def load_rules_fast(rules_dir=None, max_rules=None):
    """Load rules as SimpleRule objects, much faster than load_rules()."""
    if rules_dir is None:
        rules_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'rules'
        )

    rules_dir = Path(rules_dir)
    rule_files = sorted(rules_dir.glob('*.RULE'))

    rules = []
    for i, rule_file in enumerate(rule_files):
        if max_rules and i >= max_rules:
            break
        metadata, text = parse_rule_file(rule_file)
        if metadata is not None and text:
            rules.append(SimpleRule(rule_file, metadata, text))

    return rules
