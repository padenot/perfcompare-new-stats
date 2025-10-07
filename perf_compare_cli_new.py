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
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import bootstrap
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


def count_modes(x, y, min_prom_frac=0.02, max_prom_frac=0.2):
    """Count modes in KDE distribution."""
    y_smooth = gaussian_filter1d(y, sigma=2)
    dy = np.gradient(y_smooth, x)
    noise_est = np.percentile(np.abs(dy), 90)
    max_y = np.max(y_smooth)
    prom_est = np.clip(noise_est, min_prom_frac * max_y, max_prom_frac * max_y)
    dx = x[1] - x[0]
    x_range = x[-1] - x[0]
    min_distance = max(1, int((x_range * 0.05) / dx))
    peaks, _ = find_peaks(y_smooth, prominence=prom_est, distance=min_distance)
    return len(peaks), x[peaks], prom_est


def find_mode_interval(x, y, peaks):
    """Find intervals between modes."""
    x = np.asarray(x)
    y = np.asarray(y)
    peak_xs = sorted(peaks)
    peak_idxs = [np.searchsorted(x, px) for px in peaks]
    if len(peaks) == 0:
        return [(x[0], x[-1])]

    valleys = []
    for i in range(len(peaks) - 1):
        start = peak_idxs[i]
        end = peak_idxs[i + 1]
        valley_idx = start + np.argmin(y[start : end + 1])
        valleys.append(valley_idx)

    edges = [0] + valleys + [len(x) - 1]
    intervals = [(x[edges[i]], x[edges[i + 1]]) for i in range(len(edges) - 1)]
    return intervals


def split_per_mode(data, intervals):
    """Split data into modes based on intervals."""
    assignments = []
    for val in data:
        for i, (start, end) in enumerate(intervals):
            if start <= val <= end:
                assignments.append(i)
                break
        else:
            assignments.append(None)
    return np.array(assignments)


def bootstrap_median_diff_ci(a, b, n_iter=1000, alpha=0.05):
    """Bootstrap confidence interval for median difference."""
    data = (a, b)
    res = bootstrap(
        data,
        statistic=lambda *args: np.median(args[1]) - np.median(args[0]),
        n_resamples=n_iter,
        confidence_level=1 - alpha,
        method="percentile",
        vectorized=False,
        paired=False,
        random_state=None,
    )
    return np.median(b) - np.median(a), res.confidence_interval


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


def process_new(base_rev, new_rev, header):
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

    x_without, y_without = (
        FFTKDE(kernel="gaussian", bw="silverman").fit(without_patch).evaluate()
    )
    x_with, y_with = (
        FFTKDE(kernel="gaussian", bw="silverman").fit(with_patch).evaluate()
    )

    base_mode_count, base_peak_locs, base_prom = count_modes(x_without, y_without)
    new_mode_count, new_peak_locs, new_prom = count_modes(x_with, y_with)

    display(
        Markdown(
            f"Estimated modes (Base): {base_mode_count} (location: {base_peak_locs}, prominence: {base_prom})"
        )
    )
    display(
        Markdown(
            f"Estimated modes (New): {new_mode_count} (location: {new_peak_locs}, prominence: {new_prom})"
        )
    )

    if base_mode_count > 1:
        display(Markdown("⚠️  Warning: Base revision distribution appears multimodal!"))
    if new_mode_count > 1:
        display(Markdown("⚠️  Warning: New revision distribution appears multimodal!"))

    if base_mode_count != new_mode_count:
        display(
            Markdown(
                "⚠️  Warning: mode count between base and new revision different, look at the KDE!"
            )
        )

    if base_mode_count == new_mode_count:
        base_intervals = find_mode_interval(x_without, y_without, base_peak_locs)
        new_intervals = find_mode_interval(x_with, y_with, new_peak_locs)
        per_mode_without = split_per_mode(without_patch, base_intervals)
        per_mode_with = split_per_mode(with_patch, new_intervals)
        for i, (start, end) in enumerate(base_intervals):
            ref_vals = without_patch[per_mode_without == i]
            new_vals = with_patch[per_mode_with == i]

            if len(ref_vals) == 0 or len(new_vals) == 0:
                print(
                    f"Mode {i + 1} [{start:.2f}, {end:.2f}]: Not enough data to compare."
                )
                continue

            shift, (ci_low, ci_high) = bootstrap_median_diff_ci(ref_vals, new_vals)
            print(f"Mode {i + 1} [{start:.2f}, {end:.2f}]:")
            print(
                f"  Median shift: {shift:+.3f} (95% CI: {ci_low:+.3f} to {ci_high:+.3f})"
            )
            print(f"  → Interpretation: ", end="")
            if ci_low > 0:
                print("Performance regressed (median increased)")
            elif ci_high < 0:
                print("Performance improved (median decreased)")
            else:
                print("No significant shift")


def parse_perf_compare_url(url):
    """Extract parameters from a perf.compare URL."""
    parsed = urlparse(url)

    if "perf.compare" in parsed.netloc or "perf.compare" in parsed.path:
        # This is a frontend URL, need to extract params
        params = parse_qs(parsed.query)

        base_repo = params.get("baseRepo", [""])[0]
        base_rev = params.get("baseRev", [""])[0]
        new_repo = params.get("newRepo", [""])[0]
        new_rev = params.get("newRev", [""])[0]
        framework = params.get("framework", [""])[0]
        search_term = params.get("search", [""])[0]

        if not all([base_repo, base_rev, new_repo, new_rev, framework]):
            raise ValueError("Missing required parameters in URL")

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


def extract_chart_data(base_data, new_data):
    """Extract data for ECharts visualization."""
    from KDEpy import FFTKDE

    if len(base_data) == 0 and len(new_data) == 0:
        return None

    # Calculate medians
    base_median = float(np.median(base_data)) if len(base_data) > 0 else 0
    new_median = float(np.median(new_data)) if len(new_data) > 0 else 0

    # Determine range for KDE
    all_data = (
        np.concatenate([base_data, new_data])
        if len(base_data) > 0 and len(new_data) > 0
        else (base_data if len(base_data) > 0 else new_data)
    )
    x_min, x_max = np.min(all_data), np.max(all_data)
    padding = 0.05 * (x_max - x_min) if x_max > x_min else 1
    x_min, x_max = x_min - padding, x_max + padding

    # Generate grid points
    x_grid = np.linspace(x_min, x_max, 200)

    chart_data = {
        "base": {
            "median": base_median,
            "sample_count": len(base_data),
            "kde_x": [],
            "kde_y": [],
        },
        "new": {
            "median": new_median,
            "sample_count": len(new_data),
            "kde_x": [],
            "kde_y": [],
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


def analyze_performance_change(item, base_data, new_data):
    """Analyze performance change using our pipeline calculations only."""
    from cliffs_delta import cliffs_delta
    from scipy.stats import mannwhitneyu

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
    }


def process_single_test(args):
    """Process a single test item for parallel processing."""
    item, index, use_replicates = args

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

    try:
        with redirect_stdout(output_buffer):
            # Create header for this test
            header = f"{suite} - {test} ({platform})"

            # process_new is now defined above

            # Run full pipeline processing to get statistical analysis
            process_new(base_data_3d, new_data_3d, header)

        # Parse and clean output
        output_text = output_buffer.getvalue()
        lines = output_text.split("\n")
        cleaned_lines = []
        for line in lines:
            # Skip object representations but keep meaningful content
            if "<IPython" not in line and line.strip():
                cleaned_lines.append(line)

        statistical_analysis = "\n".join(cleaned_lines)

    except Exception as e:
        print(f"\nError running pipeline for {suite} - {test}: {e}", file=sys.stderr)
        statistical_analysis = f"Statistical analysis failed: {str(e)}"

    # Analyze performance change (our pipeline only)
    perf_analysis = analyze_performance_change(item, base_data, new_data)

    # Extract chart data for ECharts
    chart_data = extract_chart_data(base_data, new_data)

    return {
        "suite": suite,
        "test": test,
        "platform": platform,
        "chart_data": chart_data,
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


def process_results(data, use_replicates, limit=None, workers=None):
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

    # Prepare arguments for parallel processing
    task_args = [(item, i, use_replicates) for i, item in enumerate(items_with_data)]

    # Process in parallel with progress bar
    with Pool(processes=workers) as pool:
        with tqdm(total=len(task_args), desc="Processing tests") as pbar:
            for result in pool.imap(process_single_test, task_args):
                if result is not None:
                    results.append(result)
                pbar.update()

    return results


def generate_html_report(results, title="Performance Comparison Report"):
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
                                    <div class="stat-row"><span>Cliff's δ:</span><span>{perf["cliffs_delta"]:+.3f}</span></div>
                                </div>
                            </div>
                            
                            <div class="cles-section">
                                <div class="cles-explanation">{perf["cles_explanation"]}</div>
                            </div>
                            
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
                // Median lines
                {{
                    name: 'Base Median',
                    type: 'line',
                    markLine: {{
                        silent: true,
                        data: [{{
                            xAxis: baseData.median,
                            lineStyle: {{
                                color: '#3498db',
                                type: 'dashed',
                                width: 2
                            }},
                            label: {{
                                formatter: `Base: ${{baseData.median.toFixed(2)}}`
                            }}
                        }}]
                    }}
                }},
                {{
                    name: 'New Median',
                    type: 'line',
                    markLine: {{
                        silent: true,
                        data: [{{
                            xAxis: newData.median,
                            lineStyle: {{
                                color: '#e67e22',
                                type: 'dashed',
                                width: 2
                            }},
                            label: {{
                                formatter: `New: ${{newData.median.toFixed(2)}}`
                            }}
                        }}]
                    }}
                }}
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
        """,
    )

    parser.add_argument("url", help="perf.compare URL or API URL")
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

    args = parser.parse_args()

    # Clear cache if requested
    if args.clear_cache and CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"Cache cleared: {CACHE_DIR}")

    # Parse URL and extract search term
    api_url, url_search_term = parse_perf_compare_url(args.url)

    # Use CLI search argument if provided, otherwise use URL search parameter
    search_term = args.search if args.search else url_search_term

    use_replicates = not args.no_replicates
    use_cache = not args.no_cache

    data = fetch_performance_data(api_url, use_replicates, use_cache)

    # Filter results based on search term
    if search_term:
        data = filter_results_by_search(data, search_term)

    if args.save_data:
        with open(args.save_data, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {args.save_data}")

    results = process_results(data, use_replicates, args.limit, args.workers)

    if results:
        report_title = args.title if args.title else "Performance Comparison Report"
        html_content = generate_html_report(results, report_title)

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
