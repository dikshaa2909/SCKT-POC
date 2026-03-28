#
# Copyright (c) nexB Inc. and others. All rights reserved.
# ScanCode is a trademark of nexB Inc.
# SPDX-License-Identifier: Apache-2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/nexB/scancode-toolkit for support or download.
# See https://aboutcode.org for more information about nexB OSS projects.
#

"""
Prediction pipeline for ML-assisted required phrase suggestion.

Runs a trained model on license rules that currently lack required phrase
markers, generates candidate spans, applies safety post-filters, and
classifies suggestions into auto-apply / review / reject buckets.

Supports both sklearn (fast prototype) and DeBERTa (production) modes.

Usage::

    from licensedcode.ml_required_phrases.predict import suggest_required_phrases
    suggestions = suggest_required_phrases(model_bundle, rules_data_dir)
"""

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from licensedcode.models import load_rules
from licensedcode.models import rules_data_dir
from licensedcode.stopwords import STOPWORDS
from licensedcode.tokenize import REQUIRED_PHRASE_CLOSE
from licensedcode.tokenize import REQUIRED_PHRASE_OPEN
from licensedcode.tokenize import required_phrase_tokenizer

from licensedcode.ml_required_phrases.dataset import get_normalized_tokens_for_ml
from licensedcode.ml_required_phrases.alignment import align_predictions_to_spans
from licensedcode.ml_required_phrases.postfilter import classify_suggestion
from licensedcode.ml_required_phrases.postfilter import DEFAULT_T_HIGH
from licensedcode.ml_required_phrases.postfilter import DEFAULT_T_LOW

# We need the same label list used in train
from licensedcode.ml_required_phrases.train import LABEL_LIST, ID_TO_LABEL


def predict_bio_labels_sklearn(tokens, model_bundle):
    """
    Predict BIO labels using sklearn classifier.

    Args:
        tokens: list of token strings
        model_bundle: dict with 'classifier' and 'vocab'

    Returns:
        (pred_labels, pred_probs)
    """
    from licensedcode.ml_required_phrases.train import featurize_example
    from licensedcode.ml_required_phrases.train import features_to_vector

    clf = model_bundle['classifier']
    vocab = model_bundle['vocab']

    token_features = featurize_example(tokens)
    X = np.array([features_to_vector(f, vocab) for f in token_features])

    pred_ids = clf.predict(X)
    pred_proba = clf.predict_proba(X)

    pred_labels = []
    pred_probs = []

    classes = clf.classes_

    for i, pred_id in enumerate(pred_ids):
        label = ID_TO_LABEL[pred_id]
        pred_labels.append(label)

        prob_dict = {}
        for j, cls_id in enumerate(classes):
            prob_dict[ID_TO_LABEL[cls_id]] = float(pred_proba[i][j])
        prob_dict['max_prob'] = float(np.max(pred_proba[i]))
        pred_probs.append(prob_dict)

    return pred_labels, pred_probs


def predict_bio_labels_deberta(tokens, model_bundle):
    """
    Predict BIO labels using DeBERTa model.

    Args:
        tokens: list of token strings
        model_bundle: dict with 'model' and 'tokenizer'

    Returns:
        (pred_labels, pred_probs)
    """
    import torch
    import torch.nn.functional as F

    model = model_bundle['model']
    tokenizer = model_bundle['tokenizer']

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    word_ids = encoded.word_ids()
    inputs = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    preds = np.argmax(probs, axis=-1)

    pred_labels = []
    pred_probs = []
    current_word = None

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != current_word:
            pred_id = preds[idx]
            subword_probs = probs[idx]
            pred_labels.append(ID_TO_LABEL[pred_id])
            prob_dict = {
                ID_TO_LABEL[i]: float(p) for i, p in enumerate(subword_probs)
            }
            prob_dict['max_prob'] = float(np.max(subword_probs))
            pred_probs.append(prob_dict)
            current_word = word_id

    while len(pred_labels) < len(tokens):
        pred_labels.append("O")
        prob_dict = {l: 0.0 for l in LABEL_LIST}
        prob_dict["O"] = 1.0
        prob_dict['max_prob'] = 1.0
        pred_probs.append(prob_dict)

    return pred_labels, pred_probs


def predict_bio_labels(tokens, model_bundle):
    """
    Predict BIO labels using the appropriate model mode.
    Auto-dispatches based on model_bundle['mode'].
    """
    mode = model_bundle.get('mode', 'sklearn')
    if mode == 'deberta':
        return predict_bio_labels_deberta(tokens, model_bundle)
    else:
        return predict_bio_labels_sklearn(tokens, model_bundle)


def suggest_for_rule(rule, model_bundle, config=None):
    """
    Generate required phrase suggestions for a single rule.

    Args:
        rule: a Rule object without required phrase markers
        model_bundle: trained model bundle
        config: optional configuration dict with threshold overrides

    Returns a list of suggestion dicts.
    """
    config = config or {}

    text = rule.text
    if not text or not text.strip():
        return []

    tokens = get_normalized_tokens_for_ml(text)
    if not tokens or len(tokens) < 3:
        return []

    # Run model prediction
    pred_labels, pred_probs = predict_bio_labels(tokens, model_bundle)

    # Align predictions to spans
    aligned_spans = align_predictions_to_spans(tokens, pred_labels, pred_probs)

    if not aligned_spans:
        return []

    # Apply safety post-filters and classify each span
    suggestions = []
    for span_info in aligned_spans:
        classification = classify_suggestion(span_info, rule, config)

        suggestion = {
            'rule_id': rule.identifier,
            'license_expression': getattr(rule, 'license_expression', ''),
            'span': span_info['span'],
            'start': span_info['start'],
            'end': span_info['end'],
            'tokens': span_info['tokens'],
            'text': span_info['text'],
            'confidence': span_info['confidence'],
            'mean_confidence': span_info['mean_confidence'],
            'bucket': classification['bucket'],
            'rejection_reasons': classification['rejection_reasons'],
            'filter_results': [
                {'name': fr.filter_name, 'passed': fr.passed, 'reason': fr.reason}
                for fr in classification['filter_results']
            ],
            'original_text': text,
        }
        suggestions.append(suggestion)

    return suggestions


def suggest_required_phrases(
    model_bundle,
    rules_data_dir=rules_data_dir,
    config=None,
    max_rules=None,
    verbose=True,
    dry_run=True,
):
    """
    Run the full ML suggestion pipeline on all eligible rules.

    Scans rules without required phrase markers, runs predictions,
    applies safety filters, and classifies suggestions.

    Returns a dict with auto_apply, review, rejected lists and stats.
    """
    config = config or {}

    results = {
        'auto_apply': [],
        'review': [],
        'rejected': [],
        'stats': {
            'total_rules': 0,
            'eligible_rules': 0,
            'rules_with_suggestions': 0,
            'total_suggestions': 0,
            'auto_apply_count': 0,
            'review_count': 0,
            'rejected_count': 0,
        },
    }

    from licensedcode.ml_required_phrases.fast_dataset import load_rules_fast
    rules = load_rules_fast(rules_dir=rules_data_dir, max_rules=max_rules)
    processed = 0

    for rule in rules:
        results['stats']['total_rules'] += 1

        if max_rules and processed >= max_rules:
            break

        # Skip rules that already have markers or are ineligible
        text = getattr(rule, 'text', '') or ''
        if '{{' in text and '}}' in text:
            continue

        if getattr(rule, 'is_false_positive', False):
            continue

        if getattr(rule, 'is_required_phrase', False):
            continue

        if not text.strip():
            continue

        if len(text) > 4000:
            continue

        if getattr(rule, 'skip_for_required_phrase_generation', False):
            continue

        tokens = get_normalized_tokens_for_ml(text)
        if len(tokens) < 3:
            continue

        results['stats']['eligible_rules'] += 1
        processed += 1

        # Generate suggestions
        try:
            suggestions = suggest_for_rule(rule, model_bundle, config)
        except Exception as e:
            if verbose:
                print(f"  Error processing {rule.identifier}: {e}")
            continue

        if suggestions:
            results['stats']['rules_with_suggestions'] += 1

        for s in suggestions:
            results['stats']['total_suggestions'] += 1
            bucket = s['bucket']

            if bucket == 'auto_apply':
                results['auto_apply'].append(s)
                results['stats']['auto_apply_count'] += 1
            elif bucket == 'review':
                results['review'].append(s)
                results['stats']['review_count'] += 1
            else:
                results['rejected'].append(s)
                results['stats']['rejected_count'] += 1

        if verbose and processed % 500 == 0:
            print(f"  Processed {processed} eligible rules...")

    if verbose:
        stats = results['stats']
        print(f"\n{'='*60}")
        print(f"ML SUGGESTION PIPELINE RESULTS")
        print(f"{'='*60}")
        print(f"Total rules scanned: {stats['total_rules']}")
        print(f"Eligible rules (no markers): {stats['eligible_rules']}")
        print(f"Rules with suggestions: {stats['rules_with_suggestions']}")
        print(f"Total suggestions: {stats['total_suggestions']}")
        print(f"  Auto-apply: {stats['auto_apply_count']}")
        print(f"  Review queue: {stats['review_count']}")
        print(f"  Rejected: {stats['rejected_count']}")

    return results


def generate_suggested_text(original_text, tokens, start, end):
    """
    Generate rule text with {{...}} markers inserted around the suggested span.
    """
    from licensedcode.tokenize import matched_query_text_tokenizer

    token_tuples = list(matched_query_text_tokenizer(original_text))
    result_parts = []
    token_idx = 0
    marker_open_inserted = False
    marker_close_inserted = False

    for is_word, token_text in token_tuples:
        if is_word and token_text.lower() not in STOPWORDS:
            if token_idx == start and not marker_open_inserted:
                result_parts.append('{{')
                marker_open_inserted = True

            result_parts.append(token_text)

            if token_idx == end and not marker_close_inserted:
                result_parts.append('}}')
                marker_close_inserted = True

            token_idx += 1
        else:
            result_parts.append(token_text)

    return ''.join(result_parts)


def save_suggestions(results, output_path):
    """Save suggestion results to a JSON file for review."""
    serializable = {
        'stats': results['stats'],
        'auto_apply': [],
        'review': [],
        'rejected': [],
        'rejected_count': len(results['rejected']),
    }

    for bucket in ('auto_apply', 'review', 'rejected'):
        for s in results[bucket]:
            item = {
                'rule_id': s['rule_id'],
                'license_expression': s['license_expression'],
                'text': s['text'],
                'tokens': s['tokens'],
                'start': s['start'],
                'end': s['end'],
                'confidence': s['confidence'],
                'mean_confidence': s['mean_confidence'],
                'bucket': s['bucket'],
                'original_text': s['original_text'],
                'filter_results': s.get('filter_results', []),
            }
            serializable[bucket].append(item)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    return output_path
