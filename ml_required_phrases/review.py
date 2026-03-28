#
# Copyright (c) nexB Inc. and others. All rights reserved.
# ScanCode is a trademark of nexB Inc.
# SPDX-License-Identifier: Apache-2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/nexB/scancode-toolkit for support or download.
# See https://aboutcode.org for more information about nexB OSS projects.
#

"""
Review CLI and minimal web UI for ML-predicted required phrase suggestions.

Provides:
1. CLI commands for reviewing and approving/rejecting suggestions
2. A minimal browser-based review interface for bulk review
3. Export of approved suggestions in a patch-friendly format

Usage::

    # CLI review
    from licensedcode.ml_required_phrases.review import review_suggestions_cli
    review_suggestions_cli(suggestions_file)

    # Web UI review
    from licensedcode.ml_required_phrases.review import start_review_server
    start_review_server(suggestions_file, port=8089)
"""

import json
import os
import sys
from http.server import HTTPServer
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs
from urllib.parse import urlparse

from licensedcode.ml_required_phrases.postfilter import DEFAULT_T_HIGH
from licensedcode.ml_required_phrases.postfilter import DEFAULT_T_LOW


def load_suggestions(filepath):
    """Load suggestions from a JSON file."""
    with open(filepath) as f:
        return json.load(f)


def save_review_results(results, filepath):
    """Save review decisions to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def format_suggestion_for_cli(suggestion, index):
    """Format a single suggestion for CLI display."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  Suggestion #{index + 1}")
    lines.append(f"{'='*70}")
    lines.append(f"  Rule ID:    {suggestion['rule_id']}")
    lines.append(f"  License:    {suggestion['license_expression']}")
    lines.append(f"  Confidence: {suggestion['confidence']:.4f} "
                 f"(mean: {suggestion['mean_confidence']:.4f})")
    lines.append(f"  Bucket:     {suggestion['bucket']}")
    lines.append(f"  Suggested:  {{{{ {suggestion['text']} }}}}")
    lines.append(f"")
    lines.append(f"  Original text (excerpt):")

    original = suggestion.get('original_text', '')
    # Show a truncated version
    if len(original) > 300:
        lines.append(f"    {original[:300]}...")
    else:
        lines.append(f"    {original}")

    lines.append(f"")

    # Show filter results
    filters = suggestion.get('filter_results', [])
    if filters:
        lines.append(f"  Filters:")
        for fr in filters:
            status = '✓' if fr['passed'] else '✗'
            lines.append(f"    {status} {fr['name']}: {fr['reason']}")

    return '\n'.join(lines)


def review_suggestions_cli(suggestions_file, output_file=None):
    """
    Interactive CLI review of suggestions.

    Presents each suggestion and asks for accept/reject/edit/skip decision.
    Saves approved suggestions to output_file.
    """
    data = load_suggestions(suggestions_file)

    review_items = data.get('review', [])
    auto_items = data.get('auto_apply', [])

    print(f"\n{'='*70}")
    print(f"  ML REQUIRED PHRASE SUGGESTION REVIEW")
    print(f"{'='*70}")
    print(f"  Auto-apply suggestions: {len(auto_items)}")
    print(f"  Review suggestions: {len(review_items)}")
    print(f"  Stats: {json.dumps(data.get('stats', {}), indent=4)}")

    approved = []
    rejected = []

    # Auto-apply items are pre-approved
    for item in auto_items:
        item['decision'] = 'approved'
        approved.append(item)

    print(f"\n  {len(auto_items)} auto-apply suggestions pre-approved.")
    print(f"  Starting interactive review of {len(review_items)} suggestions...")
    print(f"  Commands: [a]ccept, [r]eject, [s]kip, [q]uit\n")

    for i, suggestion in enumerate(review_items):
        print(format_suggestion_for_cli(suggestion, i))

        while True:
            try:
                choice = input(f"\n  Decision [a/r/s/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = 'q'

            if choice in ('a', 'accept'):
                suggestion['decision'] = 'approved'
                approved.append(suggestion)
                print(f"  ✓ Approved")
                break
            elif choice in ('r', 'reject'):
                suggestion['decision'] = 'rejected'
                rejected.append(suggestion)
                print(f"  ✗ Rejected")
                break
            elif choice in ('s', 'skip'):
                print(f"  ⟳ Skipped")
                break
            elif choice in ('q', 'quit'):
                print(f"\n  Quitting review. Saving progress...")
                break
            else:
                print(f"  Invalid choice. Use [a]ccept, [r]eject, [s]kip, [q]uit")

        if choice in ('q', 'quit'):
            break

    # Save results
    if not output_file:
        output_file = str(Path(suggestions_file).with_suffix('.approved.json'))

    results = {
        'approved': approved,
        'rejected': rejected,
        'total_reviewed': len(approved) + len(rejected),
    }
    save_review_results(results, output_file)

    print(f"\n  Review complete:")
    print(f"    Approved: {len(approved)}")
    print(f"    Rejected: {len(rejected)}")
    print(f"    Saved to: {output_file}")

    return results


# ============================================================================
# Minimal Web UI for bulk review
# ============================================================================

REVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ScanCode ML Required Phrase Review</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-primary: #ffffff;
    --bg-secondary: #f3f4f6;
    --bg-card: #ffffff;
    --bg-hover: #e5e7eb;
    --text-primary: #1f2937;
    --text-secondary: #4b5563;
    --text-muted: #9ca3af;
    --accent-blue: #2563eb;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-purple: #8b5cf6;
    --border: #e5e7eb;
    --shadow: 0 1px 3px rgba(0,0,0,0.1);
    --radius: 4px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    line-height: 1.6;
  }

  /* Header */
  .header {
    background: #ffffff;
    border-bottom: 2px solid var(--accent-blue);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .header h1 {
    font-size: 20px;
    font-weight: 600;
    color: var(--accent-blue);
  }
  .header-stats {
    display: flex;
    gap: 24px;
  }
  .stat-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: var(--bg-card);
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    border: 1px solid var(--border);
  }
  .stat-badge .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
  }
  .dot-green { background: var(--accent-green); }
  .dot-amber { background: var(--accent-amber); }
  .dot-red { background: var(--accent-red); }
  .dot-blue { background: var(--accent-blue); }

  /* Controls */
  .controls {
    padding: 16px 32px;
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    align-items: center;
    border-bottom: 1px solid var(--border);
    background: var(--bg-secondary);
  }
  .controls select, .controls input {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 13px;
    font-family: inherit;
    outline: none;
    transition: border-color 0.2s;
  }
  .controls select:focus, .controls input:focus {
    border-color: var(--accent-blue);
  }
  .btn {
    padding: 8px 18px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border: 1px solid var(--accent-blue);
    background: #ffffff;
    color: var(--accent-blue);
    font-family: inherit;
    transition: all 0.2s;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .btn:hover { background: #f0f4ff; }
  .btn-export:hover { background: rgba(79, 140, 247, 0.25); }

  /* Main content */
  .main { padding: 24px 32px; }

  /* Suggestion cards */
  .suggestion-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 16px;
    overflow: hidden;
    transition: all 0.2s;
  }
  .suggestion-card:hover {
    border-color: var(--accent-blue);
    box-shadow: var(--shadow);
  }
  .suggestion-card.accepted { border-left: 3px solid var(--accent-blue); }
  .suggestion-card.rejected { border-left: 3px solid #9ca3af; opacity: 0.6; }

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
    gap: 12px;
  }
  .card-meta {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  .rule-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    color: var(--accent-blue);
  }
  .license-badge {
    padding: 3px 10px;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    border-radius: 4px;
    font-size: 12px;
  }
  .confidence-badge {
    padding: 3px 10px;
    border-radius: 4px;
    border: 1px solid var(--border);
    font-size: 12px;
    font-family: monospace;
    color: var(--text-secondary);
  }
  .conf-high { font-weight: bold; }
  .conf-mid { }
  .conf-low { color: var(--text-muted); }

  .card-actions {
    display: flex;
    gap: 8px;
  }
  .card-body { padding: 16px 20px; }

  .phrase-preview {
    background: var(--bg-secondary);
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.8;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .phrase-preview .highlight {
    background: rgba(79, 140, 247, 0.2);
    border: 1px solid rgba(79, 140, 247, 0.4);
    padding: 2px 6px;
    border-radius: 4px;
    color: var(--accent-blue);
    font-weight: 500;
  }
  .original-text {
    background: var(--bg-primary);
    padding: 12px 16px;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text-secondary);
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .filters-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 12px;
  }
  .filter-tag {
    padding: 2px 8px;
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 11px;
    color: var(--text-secondary);
  }
  .filter-pass { border-color: var(--accent-blue); }
  .filter-fail { opacity: 0.7; }

  /* Responsive */
  @media (max-width: 768px) {
    .header, .controls, .main { padding-left: 16px; padding-right: 16px; }
    .header { flex-direction: column; gap: 12px; }
  }

  /* Toast notification */
  .toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    padding: 12px 20px;
    background: var(--bg-card);
    border: 1px solid var(--accent-green);
    border-radius: 8px;
    color: var(--accent-green);
    font-size: 13px;
    opacity: 0;
    transform: translateY(10px);
    transition: all 0.3s;
    z-index: 1000;
  }
  .toast.show { opacity: 1; transform: translateY(0); }
</style>
</head>
<body>

<div class="header">
  <h1>ScanCode ML Required Phrase Review</h1>
  <div class="header-stats" style="font-size: 13px; font-weight: 500;">
    <span><span id="stat-auto">0</span> Auto-Apply</span> |
    <span><span id="stat-review">0</span> Review</span> |
    <span><span id="stat-rejected-filter">0</span> Rejected by Filter</span> |
    <span><span id="stat-accepted">0</span> Accepted</span> |
    <span><span id="stat-rejected">0</span> Ignored</span>
  </div>
</div>

<div class="controls">
  <select id="filter-bucket">
    <option value="all">All Buckets</option>
    <option value="auto_apply">Auto-Apply</option>
    <option value="review" selected>Review Queue</option>
    <option value="rejected">Rejected by Filters</option>
  </select>
  <select id="filter-sort">
    <option value="confidence-desc">Confidence ↓</option>
    <option value="confidence-asc">Confidence ↑</option>
    <option value="rule-id">Rule ID</option>
  </select>
  <input type="text" id="filter-search" placeholder="Search rule ID or license..." style="width: 240px;">
  <button class="btn" onclick="bulkAcceptAll()">Accept All Visible</button>
  <button class="btn" onclick="bulkRejectAll()">Reject All Visible</button>
  <button class="btn" onclick="exportResults()" style="margin-left: auto;">Export Approved</button>
</div>

<div class="main" id="suggestions-container">
  <p style="color: var(--text-secondary); text-align: center; padding: 60px;">Loading suggestions...</p>
</div>

<div class="toast" id="toast"></div>

<script>
let suggestions = [];
let decisions = {};  // rule_id+start -> 'accepted'|'rejected'

async function init() {
  try {
    const res = await fetch('/api/suggestions');
    const data = await res.json();

    suggestions = [];
    (data.auto_apply || []).forEach(s => { s._bucket = 'auto_apply'; suggestions.push(s); });
    (data.review || []).forEach(s => { s._bucket = 'review'; suggestions.push(s); });
    (data.rejected || []).forEach(s => { s._bucket = 'rejected'; suggestions.push(s); });

    document.getElementById('stat-auto').textContent = (data.auto_apply || []).length;
    document.getElementById('stat-review').textContent = (data.review || []).length;
    document.getElementById('stat-rejected-filter').textContent = (data.rejected || []).length;

    // Pre-accept auto_apply items
    (data.auto_apply || []).forEach(s => {
      decisions[s.rule_id + '_' + s.start] = 'accepted';
    });

    renderSuggestions();
    updateStats();
  } catch(e) {
    document.getElementById('suggestions-container').innerHTML =
      '<p style="color: var(--accent-red); text-align: center; padding: 60px;">Error loading suggestions: ' + e.message + '</p>';
  }
}

function getFilteredSuggestions() {
  const bucket = document.getElementById('filter-bucket').value;
  const sort = document.getElementById('filter-sort').value;
  const search = document.getElementById('filter-search').value.toLowerCase();

  let filtered = suggestions.filter(s => {
    if (bucket !== 'all' && s._bucket !== bucket) return false;
    if (search) {
      const searchable = (s.rule_id + ' ' + s.license_expression + ' ' + s.text).toLowerCase();
      if (!searchable.includes(search)) return false;
    }
    return true;
  });

  if (sort === 'confidence-desc') filtered.sort((a,b) => b.confidence - a.confidence);
  else if (sort === 'confidence-asc') filtered.sort((a,b) => a.confidence - b.confidence);
  else if (sort === 'rule-id') filtered.sort((a,b) => a.rule_id.localeCompare(b.rule_id));

  return filtered;
}

function renderSuggestions() {
  const filtered = getFilteredSuggestions();
  const container = document.getElementById('suggestions-container');

  if (filtered.length === 0) {
    container.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 60px;">No suggestions match filters.</p>';
    return;
  }

  container.innerHTML = filtered.map((s, i) => renderCard(s, i)).join('');
}

function renderCard(s, i) {
  const key = s.rule_id + '_' + s.start;
  const decision = decisions[key] || '';
  const cardClass = decision === 'accepted' ? 'accepted' : decision === 'rejected' ? 'rejected' : '';

  const confClass = s.confidence >= 0.9 ? 'conf-high' : s.confidence >= 0.7 ? 'conf-mid' : 'conf-low';

  // Build highlighted text preview
  const tokens = s.tokens || [];
  const phraseText = '{{' + tokens.join(' ') + '}}';

  // Build original text with highlight
  let originalExcerpt = (s.original_text || '').substring(0, 500);
  if ((s.original_text || '').length > 500) originalExcerpt += '...';

  // Filter tags
  const filterTags = (s.filter_results || []).map(fr => {
    const cls = fr.passed ? 'filter-pass' : 'filter-fail';
    const icon = fr.passed ? '[Pass]' : '[Fail]';
    return '<span class="filter-tag ' + cls + '">' + icon + ' ' + fr.name + '</span>';
  }).join('');

  return '<div class="suggestion-card ' + cardClass + '" id="card-' + key.replace(/[^a-zA-Z0-9]/g,'_') + '">' +
    '<div class="card-header">' +
      '<div class="card-meta">' +
        '<span class="rule-id">' + s.rule_id + '</span>' +
        '<span class="license-badge">' + s.license_expression + '</span>' +
        '<span class="confidence-badge ' + confClass + '">' + (s.confidence * 100).toFixed(1) + '%</span>' +
        '<span style="font-size:11px; color: var(--text-muted);">' + s._bucket + '</span>' +
      '</div>' +
      '<div class="card-actions">' +
        '<button class="btn" onclick="decide(\\\'' + key.replace(/'/g, "\\\\'") + '\\\',\\\'accepted\\\')">Accept</button>' +
        '<button class="btn" onclick="decide(\\\'' + key.replace(/'/g, "\\\\'") + '\\\',\\\'rejected\\\')">Reject</button>' +
      '</div>' +
    '</div>' +
    '<div class="card-body">' +
      '<div class="phrase-preview">Suggested phrase: <span class="highlight">' + escapeHtml(phraseText) + '</span></div>' +
      '<details><summary style="cursor:pointer; color: var(--text-secondary); font-size: 12px; margin-bottom: 8px;">Show original rule text</summary>' +
        '<div class="original-text">' + escapeHtml(originalExcerpt) + '</div>' +
      '</details>' +
      (filterTags ? '<div class="filters-row">' + filterTags + '</div>' : '') +
    '</div>' +
  '</div>';
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function decide(key, decision) {
  decisions[key] = decision;
  renderSuggestions();
  updateStats();
  showToast(decision === 'accepted' ? 'Accepted' : 'Rejected');
}

function bulkAcceptAll() {
  getFilteredSuggestions().forEach(s => {
    decisions[s.rule_id + '_' + s.start] = 'accepted';
  });
  renderSuggestions();
  updateStats();
  showToast('✓ All visible accepted');
}

function bulkRejectAll() {
  getFilteredSuggestions().forEach(s => {
    decisions[s.rule_id + '_' + s.start] = 'rejected';
  });
  renderSuggestions();
  updateStats();
  showToast('✗ All visible rejected');
}

function updateStats() {
  const accepted = Object.values(decisions).filter(d => d === 'accepted').length;
  const ignored = Object.values(decisions).filter(d => d === 'rejected').length;
  document.getElementById('stat-accepted').textContent = accepted;
  document.getElementById('stat-rejected').textContent = ignored;
}

async function exportResults() {
  const approved = suggestions.filter(s => decisions[s.rule_id + '_' + s.start] === 'accepted');
  const rejected = suggestions.filter(s => decisions[s.rule_id + '_' + s.start] === 'rejected');

  const result = {
    approved: approved.map(s => ({ rule_id: s.rule_id, license_expression: s.license_expression,
      text: s.text, tokens: s.tokens, start: s.start, end: s.end, confidence: s.confidence })),
    rejected: rejected.map(s => ({ rule_id: s.rule_id, text: s.text })),
    total_reviewed: approved.length + rejected.length,
  };

  try {
    const res = await fetch('/api/export', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(result),
    });
    const data = await res.json();
    showToast('✓ Exported ' + approved.length + ' approved suggestions to ' + data.path);
  } catch(e) {
    showToast('Error exporting: ' + e.message);
  }
}

function showToast(msg) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2500);
}

// Event listeners for filters
document.getElementById('filter-bucket').addEventListener('change', renderSuggestions);
document.getElementById('filter-sort').addEventListener('change', renderSuggestions);
document.getElementById('filter-search').addEventListener('input', renderSuggestions);

init();
</script>
</body>
</html>"""


class ReviewHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for the review web UI."""

    suggestions_data = None
    export_dir = None

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/' or parsed.path == '':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(REVIEW_HTML.encode('utf-8'))

        elif parsed.path == '/api/suggestions':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.suggestions_data).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == '/api/export':
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            data = json.loads(body)

            export_path = os.path.join(
                self.export_dir,
                'ml_required_phrases_approved.json'
            )
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'path': export_path}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def start_review_server(suggestions_file, port=8089, export_dir=None):
    """
    Start a minimal web server for reviewing ML suggestions.

    Args:
        suggestions_file: path to suggestions JSON file
        port: HTTP port (default 8089)
        export_dir: directory for export output (defaults to same dir as suggestions_file)
    """
    data = load_suggestions(suggestions_file)

    if not export_dir:
        export_dir = str(Path(suggestions_file).parent)

    ReviewHandler.suggestions_data = data
    ReviewHandler.export_dir = export_dir

    server = HTTPServer(('localhost', port), ReviewHandler)

    print(f"\n{'='*60}")
    print(f"  ScanCode ML Required Phrase Review Server")
    print(f"{'='*60}")
    print(f"  URL: http://localhost:{port}")
    print(f"  Suggestions: {len(data.get('auto_apply', []))} auto-apply, "
          f"{len(data.get('review', []))} review")
    print(f"  Export dir: {export_dir}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()
