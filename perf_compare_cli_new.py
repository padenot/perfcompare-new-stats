#!/usr/bin/env python3
"""
Performance Compare CLI Tool with ECharts interactive visualization
Fetches performance data from Mozilla's perf.compare API and analyzes it using statistical pipeline.
"""

import sys
import json
import argparse
import numpy as np
from urllib.parse import urlparse, parse_qs, unquote
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
from datetime import datetime
import hashlib
import os
from pathlib import Path

import requests
from retry import retry
from scipy.stats import mannwhitneyu, ks_2samp, shapiro
from scipy.stats import bootstrap
from scipy.signal import argrelmax
from scipy.optimize import linear_sum_assignment
from KDEpy import FFTKDE

try:
    from IPython.display import display, Markdown, HTML
except ImportError:

    def display(obj):
        if hasattr(obj, "_repr_html_"):
            print(obj._repr_html_())
        elif hasattr(obj, "__str__"):
            print(str(obj))
        else:
            print(obj)

    class Markdown:
        def __init__(self, text):
            self.text = text

        def __str__(self):
            return self.text

    class HTML:
        def __init__(self, text):
            self.text = text

        def __str__(self):
            return self.text


PVALUE_THRESHOLD = 0.05
CACHE_DIR = Path.home() / ".cache" / "perf_compare_cli"

@retry(tries=3)
def get_data(url):
    """Fetch data from URL with retry logic."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to fetch data. HTTP Status Code: {response.status_code}"
        )


def get_cache_key(url):
    """Generate cache key from URL."""
    return hashlib.md5(url.encode()).hexdigest()


def load_cached_data(url):
    """Load cached data if available."""
    cache_key = get_cache_key(url)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


def save_to_cache(url, data):
    """Save data to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = get_cache_key(url)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data cached to {cache_file}")


def fuzzy_match(text, search_terms):
    """Simple fuzzy matching using substring search."""
    if not search_terms:
        return True

    # Normalize text and search terms - replace underscores with spaces
    text_lower = text.lower().replace('_', ' ')
    search_lower = search_terms.lower().replace('_', ' ')

    # Split search terms by space and check if all terms are present
    terms = search_lower.split()
    return all(term in text_lower for term in terms)


def filter_results_by_search(data, search_term):
    """Filter results based on search term using fuzzy matching."""
    if not search_term:
        return data

    print(f"Filtering results for search term: '{search_term}'")

    filtered_data = []
    for item in data:
        # Combine suite, test, and platform for searching
        searchable_text = f"{item.get('suite', '')} {item.get('test', '')} {item.get('platform', '')} {item.get('header_name', '')}"

        if fuzzy_match(searchable_text, search_term):
            filtered_data.append(item)

    print(f"Found {len(filtered_data)} matching results out of {len(data)} total")
    return filtered_data


def summarize_data(series):
    """Summarize statistical data."""
    import pandas as pd

    summary = {
        "Sample count": len(series),
        "Mean": np.mean(series),
        "Median": np.median(series),
        "Variance": np.var(series, ddof=1),
        "Standard Deviation": np.std(series, ddof=1),
        "Min": np.min(series),
        "Max": np.max(series),
    }
    return pd.DataFrame(summary, index=[0])


def fit_kde_modes(data, valley_threshold=0.5, min_peak_fraction=0.05, min_data_fraction=0.05):
    """Detect modes via KDE with ISJ bandwidth and a relative valley-depth criterion.

    Two peaks are distinct modes only when the KDE valley between them drops below
    valley_threshold * min(peak1_density, peak2_density), AND each mode must
    contain at least min_data_fraction of the data points.

    Returns (n_modes, peak_locations, x_grid, y_kde, split_boundaries).
    """
    fallback = (1, np.array([np.median(data)]), None, None, [])
    if len(data) < 4:
        return fallback

    p1, p99 = np.percentile(data, [1, 99])
    data_fit = data[(data >= p1) & (data <= p99)]
    if len(data_fit) < 4:
        data_fit = data

    try:
        x, y = FFTKDE(kernel="gaussian", bw="ISJ").fit(data_fit).evaluate()
    except Exception:
        return fallback

    peak_idxs = argrelmax(y, order=3)[0]
    peak_idxs = peak_idxs[y[peak_idxs] >= min_peak_fraction * y.max()]

    if len(peak_idxs) == 0:
        return 1, np.array([x[np.argmax(y)]]), x, y, []

    good = [peak_idxs[0]]
    for nxt in peak_idxs[1:]:
        prev = good[-1]
        valley = y[prev:nxt + 1].min()
        if valley < valley_threshold * min(y[prev], y[nxt]):
            good.append(nxt)
        elif y[nxt] > y[good[-1]]:
            good[-1] = nxt

    boundaries = []
    for i in range(len(good) - 1):
        seg = y[good[i]:good[i + 1] + 1]
        boundaries.append(x[good[i] + np.argmin(seg)])

    # Drop modes that contain too few data points
    assignments = np.searchsorted(boundaries, data)
    keep = [i for i in range(len(good)) if np.mean(assignments == i) >= min_data_fraction]
    if len(keep) < 2:
        return 1, np.array([x[good[np.argmax(y[good])]]]), x, y, []

    good = [good[i] for i in keep]
    boundaries = []
    for i in range(len(good) - 1):
        seg = y[good[i]:good[i + 1] + 1]
        boundaries.append(x[good[i] + np.argmin(seg)])

    return len(good), x[np.array(good)], x, y, boundaries


def split_per_mode(data, boundaries):
    """Assign each data point to a mode index using valley boundaries."""
    return np.searchsorted(boundaries, data)


def match_modes(base_locs, base_fracs, new_locs, new_fracs):
    """Match base modes to new modes using combined location+weight cost (Hungarian algorithm).

    Both distance and weight difference are normalized to [0,1] and averaged equally,
    so a large weight mismatch can override a small location difference.

    Returns (pairs, unmatched_base_indices, unmatched_new_indices).
    """
    if len(base_locs) == 0 or len(new_locs) == 0:
        return [], list(range(len(base_locs))), list(range(len(new_locs)))

    all_locs = np.concatenate([base_locs, new_locs])
    loc_range = max(all_locs.max() - all_locs.min(), 1.0)
    dist_norm = np.abs(base_locs[:, None] - new_locs[None, :]) / loc_range
    weight_diff = np.abs(np.array(base_fracs)[:, None] - np.array(new_fracs)[None, :])
    cost = 0.5 * dist_norm + 0.5 * weight_diff

    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = list(zip(row_ind.tolist(), col_ind.tolist()))
    matched_base = set(row_ind.tolist())
    matched_new = set(col_ind.tolist())
    unmatched_base = [i for i in range(len(base_locs)) if i not in matched_base]
    unmatched_new = [j for j in range(len(new_locs)) if j not in matched_new]
    return pairs, unmatched_base, unmatched_new


def bootstrap_median_diff_ci(a, b, n_iter=1000, alpha=0.05):
    """Bootstrap confidence interval for median difference.

    Uses vectorized operations for efficiency with numpy.
    """
    def statistic(x, y, axis=-1):
        return np.median(y, axis=axis) - np.median(x, axis=axis)

    data = (a, b)
    res = bootstrap(
        data,
        statistic=statistic,
        n_resamples=n_iter,
        confidence_level=1 - alpha,
        method="percentile",
        vectorized=True,
        paired=False,
        random_state=42,
    )
    return np.median(b) - np.median(a), res.confidence_interval


def generate_explanation(
    base_peak_locs, base_fracs,
    new_peak_locs, new_fracs,
    pairs, unmatched_base, unmatched_new,
    pair_shifts,
    lower_is_better,
    unit="ms",
    base_letter=None,
    new_letter=None,
):
    """Produce a plain-English summary of the performance change for non-statisticians."""

    if base_letter is None:
        base_letter = {}
    if new_letter is None:
        new_letter = {}

    def frac_words(f):
        if f >= 0.92: return "virtually all"
        if f >= 0.75: return "the large majority"
        if f >= 0.55: return "most"
        if f >= 0.45: return "roughly half"
        if f >= 0.25: return "a significant portion"
        if f >= 0.10: return "a minority"
        return "a small fraction"

    def shift_words(shift, ci_low, ci_high):
        if ci_low < 0 < ci_high:
            return None, f"no statistically significant change (the ~{abs(shift):.0f}{unit} difference is within noise)"
        improved = (shift < 0) == lower_is_better
        lo = min(abs(ci_low), abs(ci_high))
        hi = max(abs(ci_low), abs(ci_high))
        word = "improved" if improved else "regressed"
        return improved, f"{word} by {lo:.0f}–{hi:.0f}{unit}"

    def speed_label(loc, locs):
        sorted_locs = sorted(locs) if lower_is_better else sorted(locs, reverse=True)
        n = len(sorted_locs)
        rank = sorted_locs.index(loc)
        if n == 1: return ""
        if n == 2: return "fast" if rank == 0 else "slow"
        if rank == 0: return "fastest"
        if rank == n - 1: return "slowest"
        return "intermediate"

    n_base = len(base_peak_locs)
    n_new = len(new_peak_locs)
    all_locs = np.concatenate([base_peak_locs, new_peak_locs])
    overall_center = np.median(all_locs)

    total_modes = n_base + len(unmatched_new)
    if total_modes > 3:
        return (
            f"This benchmark shows {total_modes} distinct execution patterns, "
            f"which is too complex to summarise in plain language. "
            f"Please refer to the KDE chart for a visual overview."
        )

    # --- Unimodal simple case ---
    if n_base == 1 and n_new == 1 and len(pairs) == 1:
        shift, cl, ch = pair_shifts[0][2], pair_shifts[0][3], pair_shifts[0][4]
        is_improvement, phrase = shift_words(shift, cl, ch)
        base_loc, new_loc = base_peak_locs[0], new_peak_locs[0]
        if is_improvement is None:
            return (
                f"Runs behaved consistently in both revisions. "
                f"The typical time ({base_loc:.0f}{unit} → {new_loc:.0f}{unit}) shows {phrase}."
            )
        elif is_improvement:
            return (
                f"Performance improved clearly and consistently: "
                f"the typical run went from {base_loc:.0f}{unit} to {new_loc:.0f}{unit} ({phrase})."
            )
        else:
            return (
                f"Performance regressed: "
                f"the typical run slowed from {base_loc:.0f}{unit} to {new_loc:.0f}{unit} ({phrase})."
            )

    # --- Multimodal case ---
    parts = []

    if n_base == 1:
        intro = f"The base revision ran consistently at around {base_peak_locs[0]:.0f}{unit}."
    else:
        descs = [
            f"<b>Mode {base_letter.get(i, chr(ord('A')+i))}</b> at ~{loc:.0f}{unit} ({f:.0%})"
            for i, (loc, f) in enumerate(zip(base_peak_locs, base_fracs))
        ]
        intro = f"The base revision showed {n_base} distinct execution patterns: {', and '.join(descs)}."
    parts.append(intro)

    mode_lines = []
    sort_key = (lambda x: base_peak_locs[x[0]]) if lower_is_better else (lambda x: -base_peak_locs[x[0]])
    for bi, ni, shift, cl, ch in sorted(pair_shifts, key=sort_key):
        base_loc = base_peak_locs[bi]
        new_loc = new_peak_locs[ni]
        base_frac = base_fracs[bi]
        new_frac = new_fracs[ni]
        letter = base_letter.get(bi, "?")
        slabel = speed_label(base_loc, list(base_peak_locs))
        _, phrase = shift_words(shift, cl, ch)

        delta_frac = new_frac - base_frac
        if abs(delta_frac) >= 0.15:
            dir_word = "more" if delta_frac > 0 else "less"
            frac_tail = f" It is now {dir_word} common: {base_frac:.0%} → {new_frac:.0%} of runs."
        else:
            frac_tail = ""

        mode_lines.append(
            f"<b>Mode {letter}</b> ({slabel}, ~{base_loc:.0f}{unit}, {base_frac:.0%} of base runs)"
            f" {phrase} — now at ~{new_loc:.0f}{unit}.{frac_tail}"
        )

    for bi in unmatched_base:
        loc = base_peak_locs[bi]
        frac = base_fracs[bi]
        letter = base_letter.get(bi, "?")
        slabel = speed_label(loc, list(base_peak_locs))
        is_slow = (lower_is_better and loc >= overall_center) or (not lower_is_better and loc <= overall_center)
        valence = "This is a positive change — that slow behavior has been eliminated." if is_slow \
            else "A previously fast pattern no longer occurs — this may be worth investigating."
        mode_lines.append(
            f"<b>Mode {letter}</b> ({slabel}, ~{loc:.0f}{unit}, {frac_words(frac)} of base runs)"
            f" is absent from the new revision. {valence}"
        )

    for ni in unmatched_new:
        loc = new_peak_locs[ni]
        frac = new_fracs[ni]
        letter = new_letter.get(ni, "?")
        is_slow = (lower_is_better and loc >= overall_center) or (not lower_is_better and loc <= overall_center)
        valence = "This new slow behavior warrants investigation." if is_slow \
            else "This new fast pattern is a positive development."
        mode_lines.append(
            f"<b>Mode {letter}</b> (new, ~{loc:.0f}{unit}, {frac_words(frac)} of new runs): {valence}"
        )

    parts.append("<br>".join(mode_lines))
    return "<br><br>".join(parts)


def interpret_effect_size(effect_size):
    """Interpret Cliff's delta effect size."""
    if abs(effect_size) < 0.15:
        return "Negligible difference"
    elif abs(effect_size) < 0.33:
        return "Small difference"
    elif abs(effect_size) < 0.47:
        return "Moderate difference"
    else:
        return "Large difference"


def process_new(base_rev, new_rev, header, lower_is_better=True, unit="ms"):
    """Complete statistical analysis pipeline."""
    from scipy import stats

    display(Markdown(f"# {header}"))
    without_patch = base_rev.flatten()
    with_patch = new_rev.flatten()

    series = [
        {"name": "without patch", "data": without_patch},
        {"name": "with patch", "data": with_patch},
    ]

    display(Markdown("### Basic statistics, normality test"))
    display(Markdown("<https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test>"))
    for serie in series:
        df = summarize_data(serie["data"])
        serie["variance"] = df["Variance"]
        display(HTML(df.to_html(index=False)))
        stat, p = stats.shapiro(serie["data"])
        display(
            Markdown(
                f"Shapiro-Wilk result: {p:.2f}, {serie['name']} is {'**likely normal**' if p > PVALUE_THRESHOLD else '**not normal**'}"
            )
        )

    display(
        Markdown(
            "Using [Mann-Whitney U test](https://en.m.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#)"
        )
    )

    ks_stat, ks_p = ks_2samp(without_patch, with_patch)
    display(Markdown(f"KS test p-value: {ks_p:.4f}"))

    if ks_p < PVALUE_THRESHOLD:
        display(
            Markdown(
                "⚠️ Distributions seem to differ (KS test). Review KDE before drawing conclusions."
            )
        )

    mann_stat, mann_pvalue = mannwhitneyu(
        without_patch, with_patch, alternative="two-sided"
    )

    from cliffs_delta import cliffs_delta

    delta, _ = cliffs_delta(without_patch, with_patch)
    interpretation = interpret_effect_size(delta)

    cles = mann_stat / (len(with_patch) * len(without_patch))
    cles_direction = (
        f"{cles:.2f} → {cles * 100:.0f}% chance a value from `without_patch` is greater than a value from `with_patch`"
        if cles >= 0.5
        else f"{1 - cles:.2f} → {100 - cles * 100:.0f}% chance a value from `with_patch` is greater than a value from `without_patch`"
    )

    display(
        Markdown(f"**Mann-Whitney U Common Language Effect Size**: {cles_direction}")
    )
    display(
        Markdown(
            f"**p-value**: {mann_pvalue:.3f}, {'not' if mann_pvalue > PVALUE_THRESHOLD else ''} significant"
        )
    )
    display(Markdown(f"**Cliff's Delta**: {delta:.2f} → {interpretation}"))

    base_mode_count, base_peak_locs, _, _, base_boundaries = fit_kde_modes(without_patch)
    new_mode_count, new_peak_locs, _, _, new_boundaries = fit_kde_modes(with_patch)

    def mode_fractions(data, boundaries, n_modes):
        assignments = split_per_mode(data, boundaries)
        return [np.mean(assignments == i) for i in range(n_modes)]

    base_fracs = mode_fractions(without_patch, base_boundaries, base_mode_count)
    new_fracs = mode_fractions(with_patch, new_boundaries, new_mode_count)

    per_mode_without = split_per_mode(without_patch, base_boundaries)
    per_mode_with = split_per_mode(with_patch, new_boundaries)
    pairs, unmatched_base, unmatched_new = match_modes(base_peak_locs, base_fracs, new_peak_locs, new_fracs)

    # Assign letters A, B, C… to base modes sorted fast→slow
    base_sorted = sorted(range(base_mode_count),
                         key=lambda i: base_peak_locs[i] if lower_is_better else -base_peak_locs[i])
    base_letter = {idx: chr(ord('A') + rank) for rank, idx in enumerate(base_sorted)}

    # Matched new modes inherit the base letter; unmatched new get fresh letters
    new_letter = {new_i: base_letter[base_i] for base_i, new_i in pairs}
    next_ord = ord('A') + base_mode_count
    for new_i in unmatched_new:
        new_letter[new_i] = chr(next_ord)
        next_ord += 1

    base_mode_str = ", ".join(
        f"{base_letter[i]}: {loc:.1f} ({f:.0%})"
        for i, (loc, f) in enumerate(zip(base_peak_locs, base_fracs))
    )
    new_mode_str = ", ".join(
        f"{new_letter[i]}: {loc:.1f} ({f:.0%})"
        for i, (loc, f) in enumerate(zip(new_peak_locs, new_fracs))
    )
    print(f"Modes (Base): {base_mode_count} → {base_mode_str}")
    print(f"Modes (New):  {new_mode_count} → {new_mode_str}")

    if base_mode_count > 1:
        print("⚠️  Base revision distribution appears multimodal!")
    if new_mode_count > 1:
        print("⚠️  New revision distribution appears multimodal!")

    pair_shifts = []
    for base_i, new_i in pairs:
        base_loc = base_peak_locs[base_i]
        new_loc = new_peak_locs[new_i]
        letter = base_letter[base_i]
        ref_vals = without_patch[per_mode_without == base_i]
        new_vals = with_patch[per_mode_with == new_i]

        if len(ref_vals) < 2 or len(new_vals) < 2:
            print(f"Mode {letter}: Not enough data to compare.")
            continue

        shift, (ci_low, ci_high) = bootstrap_median_diff_ci(ref_vals, new_vals)
        pair_shifts.append((base_i, new_i, shift, ci_low, ci_high))
        print(f"Mode {letter} ({base_loc:.1f} → {new_loc:.1f}, base {base_fracs[base_i]:.0%}, new {new_fracs[new_i]:.0%}):")
        print(f"  Median shift: {shift:+.3f} (95% CI: {ci_low:+.3f} to {ci_high:+.3f})")
        print(f"  → ", end="")
        if ci_low > 0:
            print("Performance regressed (median increased)" if lower_is_better else "Performance improved (median increased)")
        elif ci_high < 0:
            print("Performance improved (median decreased)" if lower_is_better else "Performance regressed (median decreased)")
        else:
            print("No significant shift")

    for base_i in unmatched_base:
        print(f"Mode {base_letter[base_i]} ({base_peak_locs[base_i]:.1f}, {base_fracs[base_i]:.0%} of base): absent in new revision")

    for new_i in unmatched_new:
        print(f"Mode {new_letter[new_i]} ({new_peak_locs[new_i]:.1f}, {new_fracs[new_i]:.0%} of new): newly appeared")

    return generate_explanation(
        base_peak_locs, base_fracs,
        new_peak_locs, new_fracs,
        pairs, unmatched_base, unmatched_new,
        pair_shifts,
        lower_is_better,
        unit,
        base_letter,
        new_letter,
    )


def resolve_lando_id(lando_id):
    """Resolve a Lando landing job ID to a commit hash."""
    r = requests.get(f"https://api.lando.services.mozilla.com/landing_jobs/{lando_id}")
    if r.status_code != 200:
        raise ValueError(f"Failed to resolve lando ID {lando_id}: HTTP {r.status_code}")
    return r.json()["commit_id"]


def parse_perf_compare_url(url):
    """Extract parameters from a perf.compare URL."""
    parsed = urlparse(url)

    if "perf.compare" in parsed.netloc or "perf.compare" in parsed.path:
        # This is a frontend URL, need to extract params
        params = parse_qs(parsed.query)

        base_repo = params.get("baseRepo", [""])[0]
        new_repo = params.get("newRepo", [""])[0]
        framework = params.get("framework", [""])[0]
        search_term = params.get("search", [""])[0]

        if "compare-lando-results" in parsed.path:
            base_lando = params.get("baseLando", [""])[0]
            new_lando = params.get("newLando", [""])[0]
            if not all([base_repo, base_lando, new_repo, new_lando, framework]):
                raise ValueError("Missing required parameters in lando URL")
            print(f"Resolving lando IDs {base_lando} and {new_lando}...")
            base_rev = resolve_lando_id(base_lando)
            new_rev = resolve_lando_id(new_lando)
            print(f"  base: {base_rev}")
            print(f"  new:  {new_rev}")
        else:
            base_rev = params.get("baseRev", [""])[0]
            new_rev = params.get("newRev", [""])[0]

        if not all([base_repo, base_rev, new_repo, new_rev, framework]):
            raise ValueError("Missing required parameters in URL")

        is_subtests = "subtests-compare-results" in parsed.path
        if is_subtests:
            base_sig = params.get("baseParentSignature", [""])[0]
            new_sig = params.get("newParentSignature", [""])[0]
            if not all([base_sig, new_sig]):
                raise ValueError("Missing baseParentSignature or newParentSignature in subtests URL")
            api_url = (
                f"https://treeherder.mozilla.org/api/perfcompare/results/"
                f"?base_repository={base_repo}"
                f"&base_revision={base_rev}"
                f"&new_repository={new_repo}"
                f"&new_revision={new_rev}"
                f"&framework={framework}"
                f"&base_parent_signature={base_sig}"
                f"&new_parent_signature={new_sig}"
            )
        else:
            api_url = (
                f"https://treeherder.mozilla.org/api/perfcompare/results/"
                f"?base_repository={base_repo}"
                f"&base_revision={base_rev}"
                f"&new_repository={new_repo}"
                f"&new_revision={new_rev}"
                f"&framework={framework}"
                f"&no_subtests=true"
            )

        # Decode search term if present (handles URL encoding like + for space)
        if search_term:
            search_term = unquote(search_term.replace('+', ' '))

        return api_url, search_term
    else:
        # Assume it's already an API URL, extract search from it if present
        params = parse_qs(parsed.query)
        search_term = params.get("search", [""])[0]
        if search_term:
            search_term = unquote(search_term.replace('+', ' '))
        return url, search_term


def fetch_performance_data(url, use_replicates=True, use_cache=True):
    """Fetch performance data from the API with caching support."""
    # Note: The API seems to not support replicates=true properly
    # When use_replicates is True, we'll omit the parameter entirely
    # When False, we'll add replicates=false

    # Remove any existing replicates parameter
    if "replicates=" in url:
        import re
        url = re.sub(r'[&?]replicates=[^&]*', '', url)
        # Fix the query string if needed
        if '?' not in url and '&' in url:
            url = url.replace('&', '?', 1)

    # Only add replicates=false when explicitly requested
    if not use_replicates:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}replicates=false"

    # Try to load from cache first
    if use_cache:
        cached_data = load_cached_data(url)
        if cached_data is not None:
            return cached_data

    print(f"Fetching data from: {url}")

    try:
        data = get_data(url)
        # Save to cache for future use
        if use_cache:
            save_to_cache(url, data)
        return data
    except Exception as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        sys.exit(1)


def prepare_data_for_pipeline(item, use_replicates):
    """Prepare data from API response for analysis."""
    if use_replicates:
        # Try to use replicates data first
        base_replicates = item.get("base_runs_replicates", [])
        new_replicates = item.get("new_runs_replicates", [])

        # Use replicates if BOTH are available
        if base_replicates and new_replicates:
            base_data = np.array(base_replicates).flatten()
            new_data = np.array(new_replicates).flatten()
        else:
            # Fall back to regular runs if either replicate is missing
            base_data = np.array(item.get("base_runs", []))
            new_data = np.array(item.get("new_runs", []))
    else:
        # Use aggregated runs data
        base_data = np.array(item.get("base_runs", []))
        new_data = np.array(item.get("new_runs", []))

    return base_data, new_data


def extract_chart_data(base_data, new_data, lower_is_better=True):
    """Extract data for ECharts visualization."""
    from KDEpy import FFTKDE

    if len(base_data) == 0 and len(new_data) == 0:
        return None

    base_median = float(np.median(base_data)) if len(base_data) > 0 else 0
    new_median = float(np.median(new_data)) if len(new_data) > 0 else 0

    all_data = (
        np.concatenate([base_data, new_data])
        if len(base_data) > 0 and len(new_data) > 0
        else (base_data if len(base_data) > 0 else new_data)
    )
    x_min, x_max = np.min(all_data), np.max(all_data)
    padding = 0.05 * (x_max - x_min) if x_max > x_min else 1
    x_min, x_max = x_min - padding, x_max + padding

    x_grid = np.linspace(x_min, x_max, 200)

    base_peaks = []
    new_peaks = []
    try:
        def _letters(locs, letter_map):
            return [{"value": float(loc), "letter": letter_map[i]} for i, loc in enumerate(locs)]

        def _assign_letters(locs, lower_is_better):
            sorted_idxs = sorted(range(len(locs)),
                                 key=lambda i: locs[i] if lower_is_better else -locs[i])
            return {idx: chr(ord('A') + rank) for rank, idx in enumerate(sorted_idxs)}

        def _mode_fracs(data, boundaries, n):
            assignments = split_per_mode(data, boundaries)
            return [np.mean(assignments == i) for i in range(n)]

        if len(base_data) > 1 and len(new_data) > 1:
            base_n, base_locs, _, _, base_bounds = fit_kde_modes(base_data)
            new_n, new_locs, _, _, new_bounds = fit_kde_modes(new_data)
            base_fracs = _mode_fracs(base_data, base_bounds, base_n)
            new_fracs = _mode_fracs(new_data, new_bounds, new_n)
            pairs, _, unmatched_new = match_modes(base_locs, base_fracs, new_locs, new_fracs)
            base_letter = _assign_letters(base_locs, lower_is_better)
            new_letter = {new_i: base_letter[base_i] for base_i, new_i in pairs}
            next_ord = ord('A') + base_n
            for new_i in unmatched_new:
                new_letter[new_i] = chr(next_ord)
                next_ord += 1
            base_peaks = _letters(base_locs, base_letter)
            new_peaks = _letters(new_locs, new_letter)
        elif len(base_data) > 1:
            _, base_locs, _, _, _ = fit_kde_modes(base_data)
            base_peaks = _letters(base_locs, _assign_letters(base_locs, lower_is_better))
        elif len(new_data) > 1:
            _, new_locs, _, _, _ = fit_kde_modes(new_data)
            new_peaks = _letters(new_locs, _assign_letters(new_locs, lower_is_better))
    except Exception:
        pass

    chart_data = {
        "base": {
            "median": base_median,
            "sample_count": len(base_data),
            "kde_x": [],
            "kde_y": [],
            "peaks": base_peaks,
        },
        "new": {
            "median": new_median,
            "sample_count": len(new_data),
            "kde_x": [],
            "kde_y": [],
            "peaks": new_peaks,
        },
    }

    # Calculate KDE curves
    try:
        if len(base_data) > 1:
            kde_base = FFTKDE(bw="ISJ").fit(base_data)
            y_base = kde_base.evaluate(x_grid)
            chart_data["base"]["kde_x"] = x_grid.tolist()
            chart_data["base"]["kde_y"] = y_base.tolist()

        if len(new_data) > 1:
            kde_new = FFTKDE(bw="ISJ").fit(new_data)
            y_new = kde_new.evaluate(x_grid)
            chart_data["new"]["kde_x"] = x_grid.tolist()
            chart_data["new"]["kde_y"] = y_new.tolist()
    except Exception:
        # KDE failed, charts will show just medians
        pass

    return chart_data


def analyze_performance_change(item, base_data, new_data, compute_bootstrap=True):
    """Analyze performance change using our pipeline calculations only."""
    from cliffs_delta import cliffs_delta
    from scipy.stats import mannwhitneyu
    import time

    # Only get the direction preference from API (metadata, not calculation)
    lower_is_better = item.get("lower_is_better", True)

    # Calculate everything ourselves
    if len(base_data) > 0 and len(new_data) > 0:
        base_median = np.median(base_data)
        new_median = np.median(new_data)
        delta_value = new_median - base_median
        delta_percentage = (delta_value / base_median * 100) if base_median != 0 else 0

        # Calculate Cliff's delta for effect size
        cliffs_delta_value, _ = cliffs_delta(base_data, new_data)

        # Calculate Common Language Effect Size
        mann_stat, _ = mannwhitneyu(base_data, new_data, alternative="two-sided")
        cles = mann_stat / (len(base_data) * len(new_data))

        # Generate CLES explanation
        if cles >= 0.5:
            cles_explanation = (
                f"{cles:.0%} chance a base value is greater than a new value"
            )
        else:
            cles_explanation = (
                f"{1 - cles:.0%} chance a new value is greater than a base value"
            )

        # Check if distributions are multimodal and if per-mode analysis is available
        is_multimodal = False
        has_per_mode_analysis = False
        if compute_bootstrap and len(base_data) > 1 and len(new_data) > 1:
            try:
                base_mode_count, *_ = fit_kde_modes(base_data)
                new_mode_count, *_ = fit_kde_modes(new_data)
                is_multimodal = (base_mode_count > 1 or new_mode_count > 1)
                # Per-mode analysis only available if both have same number of modes
                has_per_mode_analysis = (base_mode_count == new_mode_count and is_multimodal)
            except:
                pass

        # Bootstrap confidence interval for median difference
        # Skip if multimodal AND per-mode analysis is available (to avoid confusion)
        # Show if unimodal OR if multimodal but no per-mode analysis available
        bootstrap_time = 0
        if compute_bootstrap and not has_per_mode_analysis and len(base_data) >= 2 and len(new_data) >= 2:
            start_time = time.perf_counter()
            _, ci = bootstrap_median_diff_ci(base_data, new_data, n_iter=1000)
            bootstrap_time = time.perf_counter() - start_time
            ci_low = float(ci.low)
            ci_high = float(ci.high)
        else:
            ci_low = None
            ci_high = None

        # Interpret effect size
        if abs(cliffs_delta_value) < 0.15:
            effect_size = "Negligible"
        elif abs(cliffs_delta_value) < 0.33:
            effect_size = "Small"
        elif abs(cliffs_delta_value) < 0.47:
            effect_size = "Moderate"
        else:
            effect_size = "Large"

        # Determine if change is better or worse
        if abs(delta_value) < 0.001:
            direction = "No change"
            color_class = "neutral"
        elif (lower_is_better and delta_value < 0) or (
            not lower_is_better and delta_value > 0
        ):
            direction = "Better"
            color_class = "better"
        else:
            direction = "Worse"
            color_class = "worse"
    else:
        delta_value = 0
        delta_percentage = 0
        cliffs_delta_value = 0
        cles = 0.5
        cles_explanation = "No data for comparison"
        effect_size = "Unknown"
        direction = "No data"
        color_class = "neutral"
        ci_low = None
        ci_high = None
        bootstrap_time = 0
        is_multimodal = False
        has_per_mode_analysis = False

    return {
        "direction": direction,
        "color_class": color_class,
        "effect_size": effect_size,
        "delta_value": delta_value,
        "delta_percentage": delta_percentage,
        "cliffs_delta": cliffs_delta_value,
        "cles": cles,
        "cles_explanation": cles_explanation,
        "lower_is_better": lower_is_better,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_time": bootstrap_time,
        "is_multimodal": is_multimodal,
        "has_per_mode_analysis": has_per_mode_analysis,
    }


def process_single_test(args):
    """Process a single test item for parallel processing."""
    item, index, use_replicates, compute_bootstrap = args

    warnings.filterwarnings("ignore")

    # Import here to avoid multiprocessing issues
    import io
    from contextlib import redirect_stdout
    import matplotlib

    matplotlib.use("Agg")

    suite = item.get("suite", "unknown")
    test = item.get("test", item.get("header_name", ""))
    platform = item.get("platform", "unknown")

    # Prepare data
    base_data, new_data = prepare_data_for_pipeline(item, use_replicates)

    # Skip if no valid data
    if len(base_data) == 0 and len(new_data) == 0:
        return None

    # Skip if missing either base or new data (can't compare)
    if len(base_data) == 0 or len(new_data) == 0:
        return None

    # Reshape for pipeline (expects 3D: tasks x iterations x runs)
    base_data_3d = (
        base_data.reshape(1, 1, -1) if len(base_data) > 0 else np.array([[[]]])
    )
    new_data_3d = new_data.reshape(1, 1, -1) if len(new_data) > 0 else np.array([[[]]])

    # Capture statistical analysis from pipeline
    output_buffer = io.StringIO()

    lower_is_better = item.get("lower_is_better", True)
    unit = item.get("base_measurement_unit", "ms") or "ms"
    explanation = ""

    try:
        with redirect_stdout(output_buffer):
            header = f"{suite} - {test} ({platform})"
            explanation = process_new(base_data_3d, new_data_3d, header, lower_is_better, unit)

        output_text = output_buffer.getvalue()
        lines = output_text.split("\n")
        cleaned_lines = []
        for line in lines:
            if "<IPython" not in line and line.strip():
                cleaned_lines.append(line)

        statistical_analysis = "\n".join(cleaned_lines)

    except Exception as e:
        print(f"\nError running pipeline for {suite} - {test}: {e}", file=sys.stderr)
        statistical_analysis = f"Statistical analysis failed: {str(e)}"

    # Analyze performance change (our pipeline only)
    perf_analysis = analyze_performance_change(item, base_data, new_data, compute_bootstrap)

    # Extract chart data for ECharts
    chart_data = extract_chart_data(base_data, new_data, lower_is_better)

    return {
        "suite": suite,
        "test": test,
        "platform": platform,
        "chart_data": chart_data,
        "explanation": explanation,
        "statistical_analysis": statistical_analysis,
        "base_sample_count": len(base_data),
        "new_sample_count": len(new_data),
        "perf_analysis": perf_analysis,
        "index": index,
    }


def categorize_results(results):
    """Categorize and sort results by performance impact."""
    categories = {
        "better": {"Large": [], "Moderate": [], "Small": [], "Negligible": []},
        "worse": {"Large": [], "Moderate": [], "Small": [], "Negligible": []},
        "neutral": {"No change": []},
    }

    for result in results:
        perf = result["perf_analysis"]
        direction = perf["direction"].lower()
        effect_size = perf["effect_size"]

        if direction in ["better", "worse"]:
            if effect_size in categories[direction]:
                categories[direction][effect_size].append(result)
            else:
                categories["neutral"]["No change"].append(result)
        else:
            categories["neutral"]["No change"].append(result)

    return categories


def process_results(data, use_replicates, limit=None, workers=None, compute_bootstrap=True):
    """Process the fetched performance data through parallel processing."""
    results = []

    # Filter to only items with data - require BOTH base and new for comparison
    # When use_replicates is True, we'll fall back to regular runs if replicates aren't available
    items_with_data = []
    for item in data:
        has_replicates = item.get("base_runs_replicates") and item.get("new_runs_replicates")
        has_runs = item.get("base_runs") and item.get("new_runs")

        if use_replicates:
            # Try replicates first, fall back to regular runs
            if has_replicates or has_runs:
                items_with_data.append(item)
        else:
            # Only use regular runs
            if has_runs:
                items_with_data.append(item)

    if not items_with_data:
        print("No data found to process (need both base and new runs for comparison).")
        return results

    if limit:
        items_with_data = items_with_data[:limit]

    # Determine number of workers
    if workers is None:
        workers = min(cpu_count(), len(items_with_data))
    else:
        workers = min(workers, len(items_with_data))

    print(
        f"\nProcessing {len(items_with_data)} test results using {workers} workers..."
    )
    if compute_bootstrap:
        print("Bootstrap CI enabled (1000 resamples per test)")

    # Prepare arguments for parallel processing
    task_args = [(item, i, use_replicates, compute_bootstrap) for i, item in enumerate(items_with_data)]

    # Process in parallel with progress bar
    import time
    start_time = time.perf_counter()

    with Pool(processes=workers) as pool:
        with tqdm(total=len(task_args), desc="Processing tests") as pbar:
            for result in pool.imap(process_single_test, task_args):
                if result is not None:
                    results.append(result)
                pbar.update()

    total_time = time.perf_counter() - start_time

    if compute_bootstrap:
        bootstrap_times = [r['perf_analysis']['bootstrap_time'] for r in results]
        total_bootstrap_time = sum(bootstrap_times)
        avg_bootstrap_time = np.mean(bootstrap_times)
        median_bootstrap_time = np.median(bootstrap_times)
        print(f"\nTiming Statistics:")
        print(f"  Wall clock time: {total_time:.2f}s ({len(results)/total_time:.1f} tests/sec)")
        print(f"  Parallelization: {workers} workers")
        print(f"\nBootstrap Performance (per test):")
        print(f"  Average: {avg_bootstrap_time*1000:.1f}ms")
        print(f"  Median:  {median_bootstrap_time*1000:.1f}ms")
        print(f"  Total (sum of all): {total_bootstrap_time:.2f}s")
        print(f"\nEfficiency: Bootstrap adds ~{avg_bootstrap_time*1000:.0f}ms per test")
        print(f"            Parallelized across {workers} workers = ~{(total_bootstrap_time/workers)*1000:.0f}ms wall clock overhead")

    return results


def generate_html_report(results, title="Performance Comparison Report", source_url=None):
    """Generate interactive HTML report with ECharts."""
    categories = categorize_results(results)

    # Collect chart data
    chart_data = {}
    for result in results:
        if result["chart_data"]:
            chart_data[f"chart-{result['index']}"] = result["chart_data"]

    # Generate index HTML
    index_html = []
    for direction in ["better", "worse", "neutral"]:
        direction_title = {
            "better": "Performance Improvements",
            "worse": "Performance Regressions",
            "neutral": "No Significant Change",
        }[direction]

        direction_categories = categories[direction]
        direction_total = sum(len(items) for items in direction_categories.values())

        if direction_total > 0:
            index_html.append(f'<div class="index-category">')
            index_html.append(
                f'<div class="index-title {direction}">{direction_title} <span class="index-count">({direction_total} tests)</span></div>'
            )
            index_html.append('<div class="index-links">')

            for effect_size, items in direction_categories.items():
                if items:
                    anchor = f"{direction}-{effect_size.lower().replace(' ', '-')}"
                    count = len(items)
                    index_html.append(
                        f'<a href="#{anchor}" class="index-link">{effect_size} <span class="index-count">({count})</span></a>'
                    )

            index_html.append("</div></div>")

    # Generate test results HTML
    test_results_html = []

    for direction in ["better", "worse", "neutral"]:
        direction_categories = categories[direction]
        direction_total = sum(len(items) for items in direction_categories.values())

        if direction_total > 0:
            direction_title = {
                "better": "Performance Improvements",
                "worse": "Performance Regressions",
                "neutral": "No Significant Change",
            }[direction]

            test_results_html.append(f'<div class="category-{direction}">')
            test_results_html.append(f"<h2>{direction_title}</h2>")

            for effect_size in [
                "Large",
                "Moderate",
                "Small",
                "Negligible",
                "No change",
            ]:
                items = direction_categories.get(effect_size, [])
                if items:
                    anchor = f"{direction}-{effect_size.lower().replace(' ', '-')}"
                    test_results_html.append(
                        f'<h3 id="{anchor}">{effect_size} Effect ({len(items)} tests)</h3>'
                    )

                    for item in items:
                        perf = item["perf_analysis"]
                        full_header = (
                            f"{item['suite']} - {item['test']}"
                            if item["test"]
                            else item["suite"]
                        )
                        direction_text = f"{'Lower' if perf['lower_is_better'] else 'Higher'} is better"

                        # Generate unique test ID based on suite, test, and platform
                        test_id = (
                            f"{item['suite']}-{item['test']}-{item['platform']}".replace(
                                " ", "_"
                            )
                            .replace("/", "_")
                            .replace("\\", "_")
                            .replace(":", "_")
                            .lower()
                        )

                        # Stats cards
                        base_data = (
                            item["chart_data"]["base"]
                            if item["chart_data"]
                            else {"median": 0, "sample_count": 0}
                        )
                        new_data = (
                            item["chart_data"]["new"]
                            if item["chart_data"]
                            else {"median": 0, "sample_count": 0}
                        )

                        test_results_html.append(f"""
                        <div class="test-result {perf["color_class"]}" id="{test_id}">
                            <div class="test-header {perf["color_class"]}">
                                <div class="test-title">{full_header}
                                    <span class="perf-badge {perf["color_class"]}">
                                        {perf["direction"]} ({perf["effect_size"]}) {perf["delta_percentage"]:+.1f}%
                                    </span>
                                </div>
                                <div class="test-meta">
                                    Platform: {item["platform"]} | 
                                    Samples: Base={item["base_sample_count"]}, New={item["new_sample_count"]}
                                </div>
                                <div class="direction-indicator">
                                    {direction_text}
                                </div>
                            </div>
                            
                            <div class="stats-grid">
                                <div class="stats-card">
                                    <div class="stats-title">Base (Without Patch)</div>
                                    <div class="stat-row"><span>Samples:</span><span>{base_data["sample_count"]}</span></div>
                                    <div class="stat-row"><span>Median:</span><span>{base_data["median"]:.3f}</span></div>
                                </div>
                                <div class="stats-card">
                                    <div class="stats-title">New (With Patch)</div>
                                    <div class="stat-row"><span>Samples:</span><span>{new_data["sample_count"]}</span></div>
                                    <div class="stat-row"><span>Median:</span><span>{new_data["median"]:.3f}</span></div>
                                    <div class="stat-row"><span>Delta:</span><span>{perf["delta_value"]:+.3f} ({perf["delta_percentage"]:+.1f}%)</span></div>
                                    {"" if perf["ci_low"] is None else f'<div class="stat-row"><span>95% CI:</span><span>[{perf["ci_low"]:+.3f}, {perf["ci_high"]:+.3f}]</span></div>'}
                                    <div class="stat-row"><span>Cliff's δ:</span><span>{perf["cliffs_delta"]:+.3f}</span></div>
                                </div>
                            </div>
                            
                            <div class="cles-section">
                                <div class="cles-explanation">
                                    <strong>Effect Size:</strong> {perf["cles_explanation"]}<br/>
                                    {"" if perf["ci_low"] is None else f'<strong>Confidence Interval:</strong> We are 95% confident the median difference is between <strong>{perf["ci_low"]:+.3f}</strong> and <strong>{perf["ci_high"]:+.3f}</strong>'}
                                </div>
                            </div>

                            {f'<div class="interpretation-section"><div class="interpretation-text">{item["explanation"]}</div></div>' if item.get("explanation") else ""}

                            <div class="analysis-section">
                                <h3>Statistical Analysis</h3>
                                <div class="analysis-text">{item["statistical_analysis"]}</div>
                            </div>
                            
                            <div class="chart-container" id="chart-{item["index"]}"></div>
                        </div>
                        """)

            test_results_html.append("</div>")

    # Complete HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Comparison Report</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            font-size: 2.5em;
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 25px;
            margin-bottom: 10px;
            font-size: 1.8em;
        }}
        h3 {{
            margin-top: 20px;
            margin-bottom: 8px;
            font-size: 1.3em;
        }}
        .category-better h2 {{ color: #27ae60; }}
        .category-worse h2 {{ color: #e74c3c; }}
        .category-neutral h2 {{ color: #7f8c8d; }}
        
        .test-result {{
            background: white;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .test-result.better {{ border-left: 6px solid #27ae60; }}
        .test-result.worse {{ border-left: 6px solid #e74c3c; }}
        .test-result.neutral {{ border-left: 6px solid #7f8c8d; }}
        
        .test-header {{
            color: white;
            padding: 15px;
            margin: -15px -15px 15px -15px;
        }}
        .test-header.better {{ background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); }}
        .test-header.worse {{ background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }}
        .test-header.neutral {{ background: linear-gradient(135deg, #7f8c8d 0%, #95a5a6 100%); }}
        
        .test-title {{
            font-size: 1.4em;
            font-weight: bold;
            margin-bottom: 8px;
            word-wrap: break-word;
        }}
        .test-meta {{
            font-size: 1em;
            opacity: 0.95;
        }}
        .perf-badge {{
            display: inline-block;
            padding: 3px 10px;
            font-size: 0.9em;
            font-weight: bold;
            margin-left: 10px;
            background: rgba(255,255,255,0.2);
            color: white;
        }}
        .chart-container {{
            width: 100%;
            height: 350px;
            margin: 15px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .cles-section {{
            background: #e8f4fd;
            border-left: 4px solid #2980b9;
            padding: 12px 15px;
            margin: 10px 0;
        }}
        .cles-explanation {{
            font-style: italic;
            color: #2c3e50;
            font-size: 1em;
            line-height: 1.4;
        }}
        .interpretation-section {{
            background: #fffbea;
            border-left: 4px solid #f39c12;
            padding: 14px 16px;
            margin: 10px 0;
        }}
        .interpretation-text {{
            font-size: 1.05em;
            line-height: 1.6;
            color: #2c3e50;
        }}
        .analysis-section {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 10px 0 0 0;
        }}
        .analysis-text {{
            white-space: pre-wrap;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            line-height: 1.5;
            color: #2c3e50;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 15px 0;
        }}
        .stats-card {{
            background: #f8f9fa;
            padding: 12px;
        }}
        .stats-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
        }}
        .summary {{
            background: #e8f4f8;
            padding: 15px;
            margin-bottom: 25px;
            font-size: 1.1em;
        }}
        .metric {{
            display: inline-block;
            margin-right: 30px;
            padding: 6px 12px;
            background: white;
        }}
        .metric-label {{
            font-weight: bold;
            color: #555;
        }}
        .metric-value {{
            color: #2c3e50;
            font-size: 1.2em;
        }}
        .index {{
            background: white;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .index-category {{
            margin-bottom: 20px;
        }}
        .index-title {{
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .index-title.better {{ color: #27ae60; }}
        .index-title.worse {{ color: #e74c3c; }}
        .index-title.neutral {{ color: #7f8c8d; }}
        .index-links {{
            margin-left: 20px;
        }}
        .index-link {{
            display: inline-block;
            margin-right: 15px;
            margin-bottom: 5px;
            padding: 4px 8px;
            color: #3498db;
            text-decoration: none;
            background: #f8f9fa;
            transition: background 0.3s;
        }}
        .index-link:hover {{
            background: #e8f4f8;
        }}
        .index-count {{
            color: #666;
            font-size: 0.9em;
        }}
        .direction-indicator {{
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 5px;
        }}
        .source-url {{
            font-size: 0.95em;
            margin-bottom: 15px;
            color: #555;
        }}
        .source-url a {{
            color: #3498db;
            word-break: break-all;
        }}
        .timestamp {{
            text-align: right;
            color: #777;
            font-size: 0.95em;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {f'<div class="source-url"><a href="{source_url}" target="_blank">{source_url}</a></div>' if source_url else ''}

    <div class="summary">
        <div class="metric">
            <span class="metric-label">Total Tests:</span>
            <span class="metric-value">{len(results)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Generated:</span>
            <span class="metric-value">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
        </div>
    </div>
    
    <!-- Index Section -->
    <div class="index">
        <h2>Test Results Index</h2>
        {"".join(index_html)}
    </div>
    
    {"".join(test_results_html)}
    
    <script>
    // Chart data embedded here
    const chartDataAll = {json.dumps(chart_data, indent=2)};
    
    // Function to create interactive charts
    function createChart(containerId, chartData) {{
        const chart = echarts.init(document.getElementById(containerId));
        
        const baseData = chartData.base;
        const newData = chartData.new;

        const modeGlobalIdx = {{}};
        [...(baseData.peaks || []).map((p, i) => ({{ x: p.value, key: `base_${{i}}` }})),
         ...(newData.peaks || []).map((p, i) => ({{ x: p.value, key: `new_${{i}}` }}))]
            .sort((a, b) => a.x - b.x)
            .forEach((m, gi) => {{ modeGlobalIdx[m.key] = gi; }});

        const option = {{
            title: {{
                text: 'Distribution Comparison (KDE)',
                left: 'center',
                textStyle: {{
                    fontSize: 16,
                    color: '#2c3e50'
                }}
            }},
            tooltip: {{
                trigger: 'axis',
                axisPointer: {{
                    type: 'cross'
                }},
                formatter: function(params) {{
                    if (params.length === 0) return '';
                    const x = params[0].value[0];
                    let tooltip = `Value: ${{x.toFixed(3)}}<br/>`;
                    params.forEach(param => {{
                        if (param.seriesName.includes('KDE')) {{
                            tooltip += `${{param.seriesName}}: ${{param.value[1].toFixed(4)}}<br/>`;
                        }}
                    }});
                    return tooltip;
                }}
            }},
            legend: {{
                data: ['Base (Without Patch)', 'New (With Patch)'],
                bottom: 10
            }},
            toolbox: {{
                feature: {{
                    restore: {{}},
                    saveAsImage: {{}}
                }},
                right: 10,
                top: 10
            }},
            dataZoom: [
                {{
                    type: 'slider',
                    show: true,
                    xAxisIndex: [0],
                    bottom: 60,
                    start: 0,
                    end: 100
                }},
                {{
                    type: 'inside',
                    xAxisIndex: [0],
                    start: 0,
                    end: 100
                }}
            ],
            grid: {{ top: 80 }},
            xAxis: {{
                type: 'value',
                name: 'Value',
                nameLocation: 'middle',
                nameGap: 30
            }},
            yAxis: {{
                type: 'value',
                name: 'Density',
                nameLocation: 'middle',
                nameGap: 50
            }},
            series: [
                // KDE curves
                {{
                    name: 'Base (Without Patch)',
                    type: 'line',
                    data: baseData.kde_x ? baseData.kde_x.map((x, i) => [x, baseData.kde_y[i]]) : [],
                    smooth: true,
                    lineStyle: {{
                        width: 3,
                        color: '#3498db'
                    }},
                    symbol: 'none'
                }},
                {{
                    name: 'New (With Patch)',
                    type: 'line',
                    data: newData.kde_x ? newData.kde_x.map((x, i) => [x, newData.kde_y[i]]) : [],
                    smooth: true,
                    lineStyle: {{
                        width: 3,
                        color: '#e67e22'
                    }},
                    symbol: 'none'
                }},
                // Median lines: only when no mode data available
                ...(!baseData.peaks || baseData.peaks.length === 0 ? [{{
                    name: 'Base Median',
                    type: 'line',
                    markLine: {{
                        silent: true,
                        data: [{{ xAxis: baseData.median, lineStyle: {{ color: '#3498db', type: 'dashed', width: 2 }}, label: {{ formatter: `Base: ${{baseData.median.toFixed(2)}}` }} }}]
                    }}
                }}] : []),
                ...(!newData.peaks || newData.peaks.length === 0 ? [{{
                    name: 'New Median',
                    type: 'line',
                    markLine: {{
                        silent: true,
                        data: [{{ xAxis: newData.median, lineStyle: {{ color: '#e67e22', type: 'dashed', width: 2 }}, label: {{ formatter: `New: ${{newData.median.toFixed(2)}}` }} }}]
                    }}
                }}] : []),
                // Mode lines
                ...((baseData.peaks || []).map((p, i) => ({{
                    name: `Base ${{p.letter}}`,
                    type: 'line',
                    markLine: {{
                        silent: true,
                        data: [{{ xAxis: p.value, lineStyle: {{ color: '#3498db', type: 'solid', width: 2 }}, label: {{ formatter: `Base ${{p.letter}}: ${{p.value.toFixed(0)}}`, distance: [0, modeGlobalIdx[`base_${{i}}`] * 15], color: '#3498db' }} }}]
                    }}
                }}))),
                ...((newData.peaks || []).map((p, i) => ({{
                    name: `New ${{p.letter}}`,
                    type: 'line',
                    markLine: {{
                        silent: true,
                        data: [{{ xAxis: p.value, lineStyle: {{ color: '#e67e22', type: 'solid', width: 2 }}, label: {{ formatter: `New ${{p.letter}}: ${{p.value.toFixed(0)}}`, distance: [0, modeGlobalIdx[`new_${{i}}`] * 15], color: '#e67e22' }} }}]
                    }}
                }}))),
            ]
        }};
        
        chart.setOption(option);
        
        // Make chart responsive
        window.addEventListener('resize', function() {{
            chart.resize();
        }});
        
        return chart;
    }}
    
    // Initialize all charts when page loads
    document.addEventListener('DOMContentLoaded', function() {{
        Object.keys(chartDataAll).forEach(chartId => {{
            if (document.getElementById(chartId)) {{
                createChart(chartId, chartDataAll[chartId]);
            }}
        }});
    }});
    </script>
    
    <div class="timestamp">
        Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</body>
</html>"""

    return html


def load_subtest_json(filepath):
    """Load and parse subtest JSON file format."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    results = []
    for item in data:
        for test_name, test_data_list in item.items():
            if not test_data_list:
                continue

            for test_data in test_data_list:
                test_info = {
                    'suite': test_data.get('suite', 'unknown'),
                    'test': test_data.get('test', test_name),
                    'platform': test_data.get('platform', 'unknown'),
                    'header_name': test_data.get('header_name', test_name),
                    'base_runs': test_data.get('base_runs', []),
                    'new_runs': test_data.get('new_runs', []),
                    'base_runs_replicates': test_data.get('base_runs_replicates', []),
                    'new_runs_replicates': test_data.get('new_runs_replicates', []),
                    'base_measurement_unit': test_data.get('base_measurement_unit', ''),
                    'new_measurement_unit': test_data.get('new_measurement_unit', ''),
                    'lower_is_better': test_data.get('lower_is_better', True),
                    'is_complete': test_data.get('is_complete', True),
                }
                results.append(test_info)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze performance comparison data from Mozilla perf.compare with interactive charts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://perf.compare/compare-results?newRev=0f8c07da4319&baseRepo=try&baseRev=7bedadac3ab7&newRepo=try&framework=13"
  %(prog)s <url> --no-replicates --workers 4
  %(prog)s <url> --limit 20 --output-file interactive_report.html
  %(prog)s <url> --search "sessionrestore many_windows"
  %(prog)s --from-file perf-compare-all-revisions.json --output-file report.html
        """,
    )

    parser.add_argument("url", nargs='?', help="perf.compare URL or API URL")
    parser.add_argument(
        "--no-replicates",
        action="store_true",
        help="Disable use of replicates data (use aggregated runs instead)",
    )
    parser.add_argument(
        "--limit", type=int, metavar="N", help="Process only first N test results"
    )
    parser.add_argument(
        "--output-file",
        metavar="FILE",
        help="Output file for HTML report (default: auto-generated)",
    )
    parser.add_argument(
        "--save-data", metavar="FILE", help="Save fetched JSON data to file"
    )
    parser.add_argument(
        "--workers",
        type=int,
        metavar="N",
        help=f"Number of parallel workers (default: auto, max: {cpu_count()})",
    )
    parser.add_argument(
        "--title",
        metavar="TITLE",
        help='Custom title for the report (default: "Performance Comparison Report")',
    )
    parser.add_argument(
        "--search",
        metavar="TERM",
        help="Search term to filter results (overrides URL search parameter)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache usage for fetching data",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the cache directory before fetching",
    )
    parser.add_argument(
        "--from-file",
        metavar="FILE",
        help="Load data from a local JSON file instead of fetching from URL",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Disable bootstrap confidence intervals (faster but less informative)",
    )

    args = parser.parse_args()

    if not args.url and not args.from_file:
        parser.error("Either provide a URL or use --from-file to load from a local file")

    # Clear cache if requested
    if args.clear_cache and CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"Cache cleared: {CACHE_DIR}")

    use_replicates = not args.no_replicates
    use_cache = not args.no_cache

    if args.from_file:
        print(f"Loading data from file: {args.from_file}")
        data = load_subtest_json(args.from_file)
        search_term = args.search
    else:
        # Parse URL and extract search term
        api_url, url_search_term = parse_perf_compare_url(args.url)

        # Use CLI search argument if provided, otherwise use URL search parameter
        search_term = args.search if args.search else url_search_term

        data = fetch_performance_data(api_url, use_replicates, use_cache)

    # Filter results based on search term
    if search_term:
        data = filter_results_by_search(data, search_term)

    if args.save_data:
        with open(args.save_data, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {args.save_data}")

    compute_bootstrap = not args.no_bootstrap
    results = process_results(data, use_replicates, args.limit, args.workers, compute_bootstrap)

    if results:
        report_title = args.title if args.title else "Performance Comparison Report"
        source_url = args.url if args.url else None
        html_content = generate_html_report(results, report_title, source_url)

        if args.output_file:
            filename = args.output_file
        else:
            filename = (
                f"perf_compare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )

        with open(filename, "w") as f:
            f.write(html_content)

        print(f"\nInteractive HTML report saved to: {filename}")
        print(f"Open in browser to view interactive charts")
    else:
        print("No results to display.")
        sys.exit(1)


if __name__ == "__main__":
    main()
