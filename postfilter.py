#
# Copyright (c) nexB Inc. and others. All rights reserved.
# ScanCode is a trademark of nexB Inc.
# SPDX-License-Identifier: Apache-2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/nexB/scancode-toolkit for support or download.
# See https://aboutcode.org for more information about nexB OSS projects.
#

"""
Safety post-filters for ML-predicted required phrase spans.

Every candidate required phrase span predicted by the ML model must pass
all safety filters before being auto-applied or sent for review. These
filters implement the hard gates described in the proposal:

1. Ignorable overlap filter (URLs, paths, referenced filenames)
2. Genericity filter (too generic / stopword-heavy)
3. Rule constraint filter (incompatible with rule flags/validation)
4. Marker conflict filter (preserve existing markers)
5. Minimum informativeness filter (too short / too weak)

Confidence policy:
- Auto-apply: confidence >= T_high AND all filters pass
- Review queue: T_low <= confidence < T_high
- Reject: confidence < T_low OR any filter fails
"""

import re
from collections import Counter

from licensedcode.stopwords import STOPWORDS
from licensedcode.tokenize import get_existing_required_phrase_spans


# Default confidence thresholds (conservative starting values)
DEFAULT_T_HIGH = 0.90  # Auto-apply threshold
DEFAULT_T_LOW = 0.70   # Minimum for review queue
DEFAULT_MAX_AUTO_SPANS = 3  # Max auto-applied spans per rule
DEFAULT_MIN_SPAN_TOKENS = 2  # Minimum non-stopword tokens in span


class FilterResult:
    """Result of applying a post-filter to a candidate span."""

    def __init__(self, passed, filter_name, reason=None):
        self.passed = passed
        self.filter_name = filter_name
        self.reason = reason

    def __repr__(self):
        status = 'PASS' if self.passed else 'FAIL'
        return f"FilterResult({status}, {self.filter_name}: {self.reason})"


def filter_ignorable_overlap(span_info, rule):
    """
    Reject spans that overlap with ignorable URLs, paths, or referenced filenames.

    These tokens are part of the rule text but should not be marked as required
    phrases because they are typically URLs or paths that would make the rule
    too restrictive.
    """
    start = span_info['start']
    end = span_info['end']
    span_tokens = set(t.lower() for t in span_info['tokens'])

    # Check against ignorable URLs
    ignorable_urls = getattr(rule, 'ignorable_urls', []) or []
    for url in ignorable_urls:
        url_lower = url.lower()
        # Check if any span token is a substring of a URL
        url_parts = set(re.split(r'[/:.\-_?=&#+]', url_lower))
        url_parts = {p for p in url_parts if p and len(p) > 2}
        overlap = span_tokens & url_parts
        if len(overlap) > len(span_tokens) * 0.5:
            return FilterResult(
                False, 'ignorable_overlap',
                f"Span overlaps with ignorable URL: {url}"
            )

    # Check against referenced filenames
    ref_filenames = getattr(rule, 'referenced_filenames', []) or []
    for fname in ref_filenames:
        fname_lower = fname.lower()
        fname_parts = set(re.split(r'[/.\-_]', fname_lower))
        fname_parts = {p for p in fname_parts if p and len(p) > 2}
        overlap = span_tokens & fname_parts
        if len(overlap) > len(span_tokens) * 0.5:
            return FilterResult(
                False, 'ignorable_overlap',
                f"Span overlaps with referenced filename: {fname}"
            )

    return FilterResult(True, 'ignorable_overlap', 'No ignorable overlap detected')


def filter_genericity(span_info, min_non_stopword_tokens=DEFAULT_MIN_SPAN_TOKENS):
    """
    Reject spans that are too generic or stopword-heavy.

    A required phrase should contain distinctive, informative tokens.
    Spans composed mostly of stopwords or very common words are rejected.
    """
    tokens = span_info['tokens']
    non_stopword_tokens = [t for t in tokens if t.lower() not in STOPWORDS]

    if len(non_stopword_tokens) < min_non_stopword_tokens:
        return FilterResult(
            False, 'genericity',
            f"Only {len(non_stopword_tokens)} non-stopword tokens "
            f"(minimum: {min_non_stopword_tokens})"
        )

    # Check stopword ratio
    stopword_ratio = 1 - (len(non_stopword_tokens) / max(len(tokens), 1))
    if stopword_ratio > 0.7:
        return FilterResult(
            False, 'genericity',
            f"Stopword ratio too high: {stopword_ratio:.2f}"
        )

    # Check for overly generic single-word spans
    OVERLY_GENERIC = {
        'software', 'code', 'file', 'program', 'source',
        'free', 'open', 'use', 'used', 'using',
        'work', 'works', 'copy', 'copies',
    }
    if len(non_stopword_tokens) == 1 and non_stopword_tokens[0].lower() in OVERLY_GENERIC:
        return FilterResult(
            False, 'genericity',
            f"Single generic token: {non_stopword_tokens[0]}"
        )

    return FilterResult(True, 'genericity', 'Span is sufficiently specific')


def filter_rule_constraints(span_info, rule):
    """
    Reject spans that are incompatible with rule flags/validation semantics.

    For example, false positive rules should never receive required phrases.
    """
    if getattr(rule, 'is_false_positive', False):
        return FilterResult(
            False, 'rule_constraints',
            "Cannot add required phrases to false positive rules"
        )

    if getattr(rule, 'is_required_phrase', False):
        return FilterResult(
            False, 'rule_constraints',
            "Cannot add required phrases to is_required_phrase rules"
        )

    if getattr(rule, 'skip_for_required_phrase_generation', False):
        return FilterResult(
            False, 'rule_constraints',
            "Rule is flagged skip_for_required_phrase_generation"
        )

    # Very long texts should be handled carefully
    text = getattr(rule, 'text', '') or ''
    if len(text) > 4000:
        return FilterResult(
            False, 'rule_constraints',
            "Rule text too long (>4000 chars) for automatic phrase marking"
        )

    return FilterResult(True, 'rule_constraints', 'Rule constraints satisfied')


def filter_marker_conflict(span_info, rule):
    """
    Reject spans that conflict with existing required phrase markers.

    Preserve existing valid markers; never destructively overwrite or
    overlap with them.
    """
    text = getattr(rule, 'text', '') or ''
    if '{{' not in text:
        return FilterResult(True, 'marker_conflict', 'No existing markers')

    try:
        existing_spans = get_existing_required_phrase_spans(text)
    except Exception:
        return FilterResult(
            False, 'marker_conflict',
            "Could not parse existing markers in rule text"
        )

    start = span_info['start']
    end = span_info['end']

    for existing in existing_spans:
        # Check overlap
        if not (end < existing.start or start > existing.end):
            return FilterResult(
                False, 'marker_conflict',
                f"Overlaps with existing marker at positions {existing.start}-{existing.end}"
            )

    return FilterResult(True, 'marker_conflict', 'No marker conflicts')


def filter_minimum_informativeness(span_info, rule=None):
    """
    Reject spans that are too short or uninformative to be useful
    required phrases.

    Short spans are allowed only if they contain high-confidence,
    expression-discriminative tokens (like license names).
    """
    tokens = span_info['tokens']
    confidence = span_info.get('confidence', 0)

    # Single-token spans need very high confidence and must be informative
    if len(tokens) == 1:
        token = tokens[0].lower()

        # Allow single-token if it's a well-known license identifier
        KNOWN_LICENSE_IDS = {
            'mit', 'gpl', 'lgpl', 'agpl', 'bsd', 'apache', 'mozilla',
            'mpl', 'cddl', 'epl', 'artistic', 'zlib', 'boost',
            'unlicense', 'wtfpl', 'isc', 'postgresql', 'openssl',
        }
        if token in KNOWN_LICENSE_IDS and confidence >= 0.85:
            return FilterResult(
                True, 'min_informativeness',
                f"Known license identifier: {token}"
            )

        return FilterResult(
            False, 'min_informativeness',
            f"Single-token span '{token}' not informative enough"
        )

    return FilterResult(True, 'min_informativeness', 'Span meets minimum informativeness')


def apply_all_filters(span_info, rule, config=None):
    """
    Apply all safety post-filters to a candidate span.

    Args:
        span_info: dict with span prediction data
        rule: the Rule object being annotated
        config: optional dict with threshold overrides

    Returns a tuple of (passed_all, filter_results) where:
      - passed_all: True if all filters pass
      - filter_results: list of FilterResult objects
    """
    config = config or {}
    min_span_tokens = config.get('min_span_tokens', DEFAULT_MIN_SPAN_TOKENS)

    filters = [
        filter_ignorable_overlap(span_info, rule),
        filter_genericity(span_info, min_span_tokens),
        filter_rule_constraints(span_info, rule),
        filter_marker_conflict(span_info, rule),
        filter_minimum_informativeness(span_info, rule),
    ]

    passed_all = all(f.passed for f in filters)
    return passed_all, filters


def classify_suggestion(span_info, rule, config=None):
    """
    Classify a candidate span into one of three buckets:
      - ``auto_apply``: high confidence + all filters pass
      - ``review``: medium confidence + all filters pass
      - ``reject``: low confidence or filter failure

    Args:
        span_info: dict with span prediction data (must include 'confidence')
        rule: the Rule object
        config: optional dict with threshold overrides

    Returns a dict with:
      - ``bucket``: 'auto_apply', 'review', or 'reject'
      - ``confidence``: the span confidence
      - ``filter_results``: list of FilterResult objects
      - ``rejection_reasons``: list of failure reasons (if rejected)
    """
    config = config or {}
    t_high = config.get('t_high', DEFAULT_T_HIGH)
    t_low = config.get('t_low', DEFAULT_T_LOW)

    confidence = span_info.get('confidence', 0)

    # Apply all safety filters
    passed_all, filter_results = apply_all_filters(span_info, rule, config)

    rejection_reasons = [f.reason for f in filter_results if not f.passed]

    if not passed_all:
        bucket = 'reject'
    elif confidence >= t_high:
        bucket = 'auto_apply'
    elif confidence >= t_low:
        bucket = 'review'
    else:
        bucket = 'reject'
        rejection_reasons.append(f'Confidence {confidence:.3f} below threshold {t_low}')

    return {
        'bucket': bucket,
        'confidence': confidence,
        'filter_results': filter_results,
        'rejection_reasons': rejection_reasons,
    }
