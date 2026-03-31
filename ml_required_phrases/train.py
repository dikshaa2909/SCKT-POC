#
# Copyright (c) nexB Inc. and others. All rights reserved.
# ScanCode is a trademark of nexB Inc.
# SPDX-License-Identifier: Apache-2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/nexB/scancode-toolkit for support or download.
# See https://aboutcode.org for more information about nexB OSS projects.
#

"""
Model training for ML-assisted required phrase prediction.

Supports two modes:
  1. **sklearn** (default, fast): Uses a LogisticRegression token classifier
     with hand-crafted features. Best for prototyping and demo runs.
  2. **deberta** (production): Fine-tunes microsoft/deberta-v3-small via
     HuggingFace Trainer. Better accuracy but requires torch+transformers.

Usage::

    from licensedcode.ml_required_phrases.train import train_model
    model_bundle, metrics = train_model(dataset, mode='sklearn')
"""

import json
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

# Label mapping (shared across modes)
LABEL_LIST = ["O", "B-REQ", "I-REQ"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


# ============================================================================
# sklearn mode: fast, lightweight token classifier
# ============================================================================

# Common license-related keywords that tend to be in required phrases
LICENSE_KEYWORDS = {
    'mit', 'gpl', 'lgpl', 'agpl', 'bsd', 'apache', 'mozilla', 'mpl',
    'cddl', 'epl', 'artistic', 'zlib', 'boost', 'unlicense', 'wtfpl',
    'isc', 'postgresql', 'openssl', 'license', 'licence', 'copyright',
    'permission', 'granted', 'warranty', 'liability', 'redistribution',
    'modification', 'sublicense', 'commercial', 'attribution', 'notice',
    'patent', 'trademark', 'disclaimer', 'endorsement', 'contributor',
    'derived', 'binary', 'source', 'distribute', 'conditions',
}

# Well-known stopwords that should never be required phrases
GENERIC_TOKENS = {
    'the', 'a', 'an', 'and', 'or', 'of', 'in', 'to', 'for', 'is',
    'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'shall', 'can', 'that', 'this', 'these', 'those',
    'it', 'its', 'with', 'from', 'by', 'on', 'at', 'as', 'not', 'no',
    'if', 'but', 'so', 'than', 'more', 'most', 'other', 'some', 'any',
    'all', 'each', 'every', 'such', 'only', 'own', 'same', 'also',
    'above', 'below', 'up', 'out', 'about', 'into', 'through',
    'you', 'your', 'we', 'our', 'they', 'their', 'he', 'she',
}


def extract_token_features(tokens, position):
    """
    Extract features for a single token at ``position`` in the ``tokens`` list.

    Features include:
      - The token itself (lowered)
      - Character-level features (length, has digits, all caps, etc.)
      - Context window features (previous and next tokens)
      - License keyword membership
      - Position features (relative position in text)
    """
    token = tokens[position].lower()
    n = len(tokens)

    features = {
        'bias': 1.0,
        'token': token,
        'token_len': len(token),
        'is_license_keyword': float(token in LICENSE_KEYWORDS),
        'is_generic': float(token in GENERIC_TOKENS),
        'has_digit': float(any(c.isdigit() for c in token)),
        'is_upper': float(tokens[position].isupper()),
        'is_title': float(tokens[position].istitle()),
        'relative_pos': position / max(n - 1, 1),
        'is_first': float(position == 0),
        'is_last': float(position == n - 1),
    }

    # Context: previous token
    if position > 0:
        prev = tokens[position - 1].lower()
        features['prev_token'] = prev
        features['prev_is_license_kw'] = float(prev in LICENSE_KEYWORDS)
    else:
        features['prev_token'] = '__BOS__'
        features['prev_is_license_kw'] = 0.0

    # Context: next token
    if position < n - 1:
        nxt = tokens[position + 1].lower()
        features['next_token'] = nxt
        features['next_is_license_kw'] = float(nxt in LICENSE_KEYWORDS)
    else:
        features['next_token'] = '__EOS__'
        features['next_is_license_kw'] = 0.0

    # Context: 2-token window
    if position > 1:
        features['prev2_token'] = tokens[position - 2].lower()
    else:
        features['prev2_token'] = '__BOS2__'

    if position < n - 2:
        features['next2_token'] = tokens[position + 2].lower()
    else:
        features['next2_token'] = '__EOS2__'

    return features


def featurize_example(tokens):
    """Extract features for all tokens in an example."""
    return [extract_token_features(tokens, i) for i in range(len(tokens))]


def features_to_vector(feature_dict, vocab):
    """Convert a feature dict to a sparse numerical vector."""
    vec = np.zeros(len(vocab), dtype=np.float32)
    for key, val in feature_dict.items():
        if isinstance(val, str):
            feat_key = f"{key}={val}"
        else:
            feat_key = key

        if feat_key in vocab:
            if isinstance(val, (int, float)):
                vec[vocab[feat_key]] = val
            else:
                vec[vocab[feat_key]] = 1.0
    return vec


def build_vocab(all_feature_dicts, min_count=2):
    """Build feature vocabulary from training data."""
    feat_counter = Counter()
    for feat_dict in all_feature_dicts:
        for key, val in feat_dict.items():
            if isinstance(val, str):
                feat_counter[f"{key}={val}"] += 1
            else:
                feat_counter[key] += 1

    vocab = {}
    for feat, count in feat_counter.items():
        if count >= min_count:
            vocab[feat] = len(vocab)

    return vocab


class NumpyLogisticRegression:
    def __init__(self, max_iter=100, learning_rate=0.1):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.classes_ = np.array([0, 1, 2])
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Need 3 classes
        self.weights = np.zeros((3, n_features))
        
        # very simple one-hot encoding
        y_encoded = np.zeros((n_samples, 3))
        y_encoded[np.arange(n_samples), y] = 1
        
        # Softmax gradient descent
        for _ in range(self.max_iter):
            scores = np.dot(X, self.weights.T)
            # stable softmax
            scores = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            error = probs - y_encoded
            gradient = np.dot(error.T, X) / n_samples
            self.weights -= self.learning_rate * gradient
            
    def predict_proba(self, X):
        scores = np.dot(X, self.weights.T)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
    def predict(self, X):
        return np.argmax(np.dot(X, self.weights.T), axis=1)

def train_sklearn_model(dataset, test_ratio=0.2, seed=42, verbose=True, C=1.0):
    """
    Train a fast token classifier without requiring torch or sklearn.
    """
    from collections import defaultdict
    import random
    rng = random.Random(seed)
    examples = dataset['examples']
    by_expression = defaultdict(list)
    for ex in examples:
        by_expression[ex.get('license_expression', '')].append(ex)
    expressions = list(by_expression.keys())
    rng.shuffle(expressions)
    total = len(examples)
    test_target = int(total * test_ratio)
    test_examples, train_examples, test_count = [], [], 0
    for expr in expressions:
        group = by_expression[expr]
        if test_count < test_target:
            test_examples.extend(group)
            test_count += len(group)
        else:
            train_examples.extend(group)

    if not examples:
        raise ValueError("No training examples found in dataset")
    if verbose:
        print(f"Training set: {len(train_examples)} examples")
        print(f"Test set: {len(test_examples)} examples")

    # Extract features for all training tokens
    if verbose:
        print("Extracting features...")

    all_train_features = []
    all_train_labels = []

    for ex in train_examples:
        token_features = featurize_example(ex['tokens'])
        for i, feat_dict in enumerate(token_features):
            all_train_features.append(feat_dict)
            all_train_labels.append(LABEL_TO_ID[ex['labels'][i]])

    # Build vocabulary
    vocab = build_vocab(all_train_features, min_count=2)
    if verbose:
        print(f"Feature vocabulary size: {len(vocab)}")

    # Vectorize
    X_train = np.array([features_to_vector(f, vocab) for f in all_train_features])
    y_train = np.array(all_train_labels)

    if verbose:
        print(f"Training matrix: {X_train.shape}")
        print("Training LogisticRegression...")

    clf = NumpyLogisticRegression(max_iter=300, learning_rate=0.5)
    clf.fit(X_train, y_train)

    # Evaluate on test set
    if verbose:
        print("Evaluating...")

    all_test_features = []
    all_test_labels = []
    for ex in test_examples:
        token_features = featurize_example(ex['tokens'])
        for i, feat_dict in enumerate(token_features):
            all_test_features.append(feat_dict)
            all_test_labels.append(LABEL_TO_ID[ex['labels'][i]])

    X_test = np.array([features_to_vector(f, vocab) for f in all_test_features])
    y_test = np.array(all_test_labels)

    y_pred = clf.predict(X_test)

    # Calculate span-level metrics
    span_metrics = evaluate_spans(test_examples, clf, vocab)

    # Calculate basic accuracy
    accuracy = float(np.mean(y_pred == y_test))

    metrics = {
        'accuracy': accuracy,
        'span_precision': span_metrics['precision'],
        'span_recall': span_metrics['recall'],
        'span_f1': span_metrics['f1'],
    }

    model_bundle = {
        'mode': 'sklearn',
        'classifier': clf,
        'vocab': vocab,
        'metrics': metrics,
        'model_path': None,
    }

    return model_bundle, metrics


def evaluate_spans(test_examples, clf, vocab):
    """Evaluate span-level precision/recall/F1 on test examples."""

    def get_spans_from_labels(labels):
        spans = set()
        start = None
        for i, label in enumerate(labels):
            if label == 'B-REQ':
                if start is not None:
                    spans.add((start, i - 1))
                start = i
            elif label == 'I-REQ':
                if start is None:
                    start = i
            else:
                if start is not None:
                    spans.add((start, i - 1))
                    start = None
        if start is not None:
            spans.add((start, len(labels) - 1))
        return spans

    total_true = 0
    total_pred = 0
    total_correct = 0

    for ex in test_examples:
        tokens = ex['tokens']
        true_labels = ex['labels']

        # Predict
        token_features = featurize_example(tokens)
        X = np.array([features_to_vector(f, vocab) for f in token_features])
        pred_ids = clf.predict(X)
        pred_labels = [ID_TO_LABEL[p] for p in pred_ids]

        true_spans = get_spans_from_labels(true_labels)
        pred_spans = get_spans_from_labels(pred_labels)

        total_true += len(true_spans)
        total_pred += len(pred_spans)
        total_correct += len(true_spans & pred_spans)

    precision = total_correct / max(total_pred, 1)
    recall = total_correct / max(total_true, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correct': total_correct,
        'true_spans': total_true,
        'pred_spans': total_pred,
    }


# ============================================================================
# DeBERTa mode (production, requires torch+transformers)
# ============================================================================

def train_deberta_model(dataset, test_ratio=0.2, seed=42, verbose=True, C=1.0):
    """
    Train a DeBERTa-v3 token classification model for required phrase prediction.
    Requires: torch, transformers, datasets (HuggingFace)
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "DeBERTa mode requires: pip install torch transformers datasets\n"
            "For quick prototype demo, use mode='sklearn' instead."
        )

    from collections import defaultdict
    import random as _random
    _rng = _random.Random(seed)
    _examples = dataset['examples']
    _by_expression = defaultdict(list)
    for _ex in _examples:
        _by_expression[_ex.get('license_expression', '')].append(_ex)
    _expressions = list(_by_expression.keys())
    _rng.shuffle(_expressions)
    _total = len(_examples)
    _test_target = int(_total * test_ratio)
    test_examples, train_examples, _test_count = [], [], 0
    for _expr in _expressions:
        _group = _by_expression[_expr]
        if _test_count < _test_target:
            test_examples.extend(_group)
            _test_count += len(_group)
        else:
            train_examples.extend(_group)
    import tempfile

    if not _examples:
        raise ValueError("No training examples found in dataset")

    if verbose:
        print(f"  Training set: {len(train_examples)} examples")
        print(f"  Test set: {len(test_examples)} examples")

    model_name = "microsoft/deberta-v3-small"

    if verbose:
        print(f"  Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id is None:
                new_labels.append(-100)
            elif word_id != current_word:
                new_labels.append(LABEL_TO_ID[labels[word_id]])
                current_word = word_id
            else:
                new_labels.append(-100)
        return new_labels

    def prepare_hf_dataset(examples_list):
        tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for ex in examples_list:
            if not ex["tokens"]:
                continue
            tokenized = tokenizer(
                ex["tokens"], truncation=True, max_length=512, is_split_into_words=True
            )
            word_ids = tokenized.word_ids()
            aligned_labels = align_labels_with_tokens(ex["labels"], word_ids)
            tokenized_inputs["input_ids"].append(tokenized["input_ids"])
            tokenized_inputs["attention_mask"].append(tokenized["attention_mask"])
            tokenized_inputs["labels"].append(aligned_labels)
        return Dataset.from_dict(tokenized_inputs)

    train_hf = prepare_hf_dataset(train_examples)
    eval_hf = prepare_hf_dataset(test_examples)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    tmpdir = tempfile.mkdtemp()
    num_train_epochs = 20 if len(train_examples) < 100 else 4

    training_args = TrainingArguments(
        output_dir=os.path.join(tmpdir, "results"),
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=os.path.join(tmpdir, "logs"),
        logging_steps=10,
        no_cuda=not torch.cuda.is_available(),
    )

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        true_preds = [
            [LABEL_LIST[p] for (p, l) in zip(pred, lab) if l != -100]
            for pred, lab in zip(predictions, labels)
        ]
        true_labels = [
            [LABEL_LIST[l] for (p, l) in zip(pred, lab) if l != -100]
            for pred, lab in zip(predictions, labels)
        ]
        total_true = total_pred = total_correct = 0
        def get_spans(bio_seq):
            spans, start = set(), None
            for i, tag in enumerate(bio_seq):
                if tag == 'B-REQ':
                    if start is not None: spans.add((start, i-1))
                    start = i
                elif tag == 'I-REQ':
                    if start is None: start = i
                else:
                    if start is not None: spans.add((start, i-1)); start = None
            if start is not None: spans.add((start, len(bio_seq)-1))
            return spans
        for t_seq, p_seq in zip(true_labels, true_preds):
            t_sp, p_sp = get_spans(t_seq), get_spans(p_seq)
            total_true += len(t_sp); total_pred += len(p_sp)
            total_correct += len(t_sp & p_sp)
        prec = total_correct / max(total_pred, 1)
        rec = total_correct / max(total_true, 1)
        f1 = 2*prec*rec / max(prec+rec, 1e-8)
        return {"precision": prec, "recall": rec, "f1": f1}

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_hf, eval_dataset=eval_hf,
        tokenizer=tokenizer, data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if verbose:
        print("  Training DeBERTa model...")
    trainer.train()

    eval_metrics = trainer.evaluate()
    if verbose:
        print(f"\n  Eval Metrics: {eval_metrics}")

    model_bundle = {
        'mode': 'deberta',
        'model': model,
        'tokenizer': tokenizer,
        'metrics': eval_metrics,
        'model_path': None,
    }
    return model_bundle, eval_metrics


# ============================================================================
# Unified interface
# ============================================================================

def train_model(dataset, test_ratio=0.2, seed=42, verbose=True, C=1.0, mode='sklearn'):
    """
    Train a token classification model for required phrase prediction.

    Args:
        dataset: dataset dict from build_dataset()
        test_ratio: fraction of data for test set
        seed: random seed
        verbose: print progress
        C: regularization parameter
        mode: 'sklearn' (fast prototype) or 'deberta' (production)

    Returns:
        (model_bundle, metrics)
    """
    if mode == 'deberta':
        return train_deberta_model(dataset, test_ratio, seed, verbose, C)
    else:
        return train_sklearn_model(dataset, test_ratio, seed, verbose, C)


def save_model(model_bundle, output_path):
    """Save trained model bundle to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = model_bundle.get('mode', 'sklearn')

    if mode == 'deberta':
        output_dir = output_path.parent / "deberta_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_bundle['model'].save_pretrained(output_dir)
        model_bundle['tokenizer'].save_pretrained(output_dir)
        model_bundle['model_path'] = str(output_dir)
        bundle_info = {'mode': 'deberta', 'model_path': str(output_dir)}
    else:
        # sklearn mode: pickle the classifier and vocab
        model_data = {
            'mode': 'sklearn',
            'classifier': model_bundle['classifier'],
            'vocab': model_bundle['vocab'],
        }
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        model_bundle['model_path'] = str(output_path)
        bundle_info = {'mode': 'sklearn', 'model_path': str(output_path)}

    # Save metrics
    metrics_path = output_path.with_suffix('.metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(model_bundle.get('metrics', {}), f, indent=2)

    # Save info pointer
    info_path = output_path.with_suffix('.info.json')
    with open(info_path, 'w') as f:
        json.dump(bundle_info, f, indent=2)

    if True:  # verbose
        print(f"Model saved to: {output_path}")
        print(f"Metrics saved to: {metrics_path}")


def load_model(model_path):
    """Load trained model bundle from disk."""
    model_path = Path(model_path)

    # Try loading sklearn pickle first
    if model_path.suffix == '.pkl':
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        if isinstance(model_data, dict) and model_data.get('mode') == 'sklearn':
            return {
                'mode': 'sklearn',
                'classifier': model_data['classifier'],
                'vocab': model_data['vocab'],
                'model_path': str(model_path),
            }

    # Try info file
    info_path = model_path.with_suffix('.info.json')
    if info_path.exists():
        with open(info_path) as f:
            bundle_info = json.load(f)

        mode = bundle_info.get('mode', 'sklearn')

        if mode == 'deberta':
            import torch
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            model_dir = bundle_info['model_path']
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
            model = AutoModelForTokenClassification.from_pretrained(model_dir)
            return {
                'mode': 'deberta',
                'model': model,
                'tokenizer': tokenizer,
                'model_path': model_dir,
            }

    # Legacy: try JSON pointer
    if model_path.suffix == '.json' or (model_path.exists() and model_path.stat().st_size < 1000):
        try:
            with open(model_path) as f:
                bundle_info = json.load(f)
            model_dir = bundle_info.get('model_path', '')
            if model_dir and Path(model_dir).exists():
                from transformers import AutoTokenizer, AutoModelForTokenClassification
                tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
                model = AutoModelForTokenClassification.from_pretrained(model_dir)
                return {
                    'mode': 'deberta',
                    'model': model,
                    'tokenizer': tokenizer,
                    'model_path': model_dir,
                }
        except Exception:
            pass

    raise FileNotFoundError(f"Could not load model from {model_path}")
