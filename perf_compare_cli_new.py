#!/usr/bin/env python3
"""
Performance Compare CLI Tool with ECharts interactive visualization
Fetches performance data from Mozilla's perf.compare API and analyzes it using a
reusable statistical pipeline plus a separate display layer.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
import sys
from urllib.parse import parse_qs, unquote, urlparse
import warnings

import numpy as np
import requests
from retry import retry
from tqdm import tqdm

from perf_compare_display import generate_html_report
from perf_compare_stats import ProcessedTestResult, analyze_test_item


CACHE_DIR = Path.home() / ".cache" / "perf_compare_cli"


@retry(tries=3)
def get_data(url: str) -> object:
    """Fetch data from URL with retry logic."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    raise Exception(f"Failed to fetch data. HTTP Status Code: {response.status_code}")


def get_cache_key(url: str) -> str:
    """Generate cache key from URL."""
    return hashlib.md5(url.encode()).hexdigest()


def load_cached_data(url: str) -> object | None:
    """Load cached data if available."""
    cache_key = get_cache_key(url)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, "r") as file_handle:
            return json.load(file_handle)
    return None


def save_to_cache(url: str, data: object) -> None:
    """Save data to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = get_cache_key(url)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    with open(cache_file, "w") as file_handle:
        json.dump(data, file_handle, indent=2)
    print(f"Data cached to {cache_file}")


def fuzzy_match(text: str, search_terms: str) -> bool:
    """Simple fuzzy matching using substring search."""
    if not search_terms:
        return True

    text_lower = text.lower().replace("_", " ")
    search_lower = search_terms.lower().replace("_", " ")
    return all(term in text_lower for term in search_lower.split())


def filter_results_by_search(data: list[dict], search_term: str) -> list[dict]:
    """Filter results based on search term using fuzzy matching."""
    if not search_term:
        return data

    print(f"Filtering results for search term: '{search_term}'")

    filtered_data = []
    for item in data:
        searchable_text = (
            f"{item.get('suite', '')} {item.get('test', '')} "
            f"{item.get('platform', '')} {item.get('header_name', '')}"
        )
        if fuzzy_match(searchable_text, search_term):
            filtered_data.append(item)

    print(f"Found {len(filtered_data)} matching results out of {len(data)} total")
    return filtered_data


def resolve_lando_id(lando_id: str) -> str:
    """Resolve a Lando landing job ID to a commit hash."""
    response = requests.get(
        f"https://api.lando.services.mozilla.com/landing_jobs/{lando_id}"
    )
    if response.status_code != 200:
        raise ValueError(
            f"Failed to resolve lando ID {lando_id}: HTTP {response.status_code}"
        )
    return response.json()["commit_id"]


def parse_perf_compare_url(url: str) -> tuple[str, str]:
    """Extract API URL and search term from a perf.compare URL."""
    parsed = urlparse(url)

    if "perf.compare" in parsed.netloc or "perf.compare" in parsed.path:
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
                raise ValueError(
                    "Missing baseParentSignature or newParentSignature in subtests URL"
                )
            api_url = (
                "https://treeherder.mozilla.org/api/perfcompare/results/"
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
                "https://treeherder.mozilla.org/api/perfcompare/results/"
                f"?base_repository={base_repo}"
                f"&base_revision={base_rev}"
                f"&new_repository={new_repo}"
                f"&new_revision={new_rev}"
                f"&framework={framework}"
                f"&no_subtests=true"
            )

        if search_term:
            search_term = unquote(search_term.replace("+", " "))

        return api_url, search_term

    params = parse_qs(parsed.query)
    search_term = params.get("search", [""])[0]
    if search_term:
        search_term = unquote(search_term.replace("+", " "))
    return url, search_term


def fetch_performance_data(
    url: str, use_replicates: bool = True, use_cache: bool = True
) -> object:
    """Fetch performance data from the API with caching support."""
    if "replicates=" in url:
        import re

        url = re.sub(r"[&?]replicates=[^&]*", "", url)
        if "?" not in url and "&" in url:
            url = url.replace("&", "?", 1)

    if not use_replicates:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}replicates=false"

    if use_cache:
        cached_data = load_cached_data(url)
        if cached_data is not None:
            return cached_data

    print(f"Fetching data from: {url}")
    try:
        data = get_data(url)
        if use_cache:
            save_to_cache(url, data)
        return data
    except Exception as exc:
        print(f"Error fetching data: {exc}", file=sys.stderr)
        sys.exit(1)


def process_single_test(
    args: tuple[dict, int, bool, bool]
) -> ProcessedTestResult | None:
    """Process a single test item for parallel processing."""
    item, index, use_replicates, compute_bootstrap = args
    warnings.filterwarnings("ignore")

    try:
        return analyze_test_item(item, index, use_replicates, compute_bootstrap)
    except Exception as exc:
        suite = item.get("suite", "unknown")
        test = item.get("test", item.get("header_name", ""))
        print(
            f"\nError running pipeline for {suite} - {test}: {exc}",
            file=sys.stderr,
        )
        return None


def process_results(
    data: list[dict],
    use_replicates: bool,
    limit: int | None = None,
    workers: int | None = None,
    compute_bootstrap: bool = True,
) -> list[ProcessedTestResult]:
    """Process the fetched performance data through parallel processing."""
    results: list[ProcessedTestResult] = []

    items_with_data = []
    for item in data:
        has_replicates = item.get("base_runs_replicates") and item.get(
            "new_runs_replicates"
        )
        has_runs = item.get("base_runs") and item.get("new_runs")

        if use_replicates:
            if has_replicates or has_runs:
                items_with_data.append(item)
        elif has_runs:
            items_with_data.append(item)

    if not items_with_data:
        print("No data found to process (need both base and new runs for comparison).")
        return results

    if limit:
        items_with_data = items_with_data[:limit]

    worker_count = min(workers or cpu_count(), len(items_with_data))

    print(f"\nProcessing {len(items_with_data)} test results using {worker_count} workers...")
    if compute_bootstrap:
        print("Bootstrap CI enabled (1000 resamples per test)")

    task_args = [
        (item, index, use_replicates, compute_bootstrap)
        for index, item in enumerate(items_with_data)
    ]

    import time

    start_time = time.perf_counter()
    with Pool(processes=worker_count) as pool:
        with tqdm(total=len(task_args), desc="Processing tests") as progress:
            for result in pool.imap(process_single_test, task_args):
                if result is not None:
                    results.append(result)
                progress.update()

    total_time = time.perf_counter() - start_time
    if compute_bootstrap and results:
        bootstrap_times = [result.perf_analysis.bootstrap_time for result in results]
        total_bootstrap_time = float(sum(bootstrap_times))
        avg_bootstrap_time = float(np.mean(bootstrap_times))
        median_bootstrap_time = float(np.median(bootstrap_times))
        print("\nTiming Statistics:")
        print(f"  Wall clock time: {total_time:.2f}s ({len(results) / total_time:.1f} tests/sec)")
        print(f"  Parallelization: {worker_count} workers")
        print("\nBootstrap Performance (per test):")
        print(f"  Average: {avg_bootstrap_time * 1000:.1f}ms")
        print(f"  Median:  {median_bootstrap_time * 1000:.1f}ms")
        print(f"  Total (sum of all): {total_bootstrap_time:.2f}s")
        print("\nEfficiency: Bootstrap adds "
              f"~{avg_bootstrap_time * 1000:.0f}ms per test")
        print("            Parallelized across "
              f"{worker_count} workers = "
              f"~{(total_bootstrap_time / worker_count) * 1000:.0f}ms wall clock overhead")

    return results


def load_subtest_json(filepath: str) -> list[dict]:
    """Load and parse subtest JSON file format."""
    with open(filepath, "r") as file_handle:
        data = json.load(file_handle)

    results = []
    for item in data:
        for test_name, test_data_list in item.items():
            if not test_data_list:
                continue

            for test_data in test_data_list:
                results.append(
                    {
                        "suite": test_data.get("suite", "unknown"),
                        "test": test_data.get("test", test_name),
                        "platform": test_data.get("platform", "unknown"),
                        "header_name": test_data.get("header_name", test_name),
                        "base_runs": test_data.get("base_runs", []),
                        "new_runs": test_data.get("new_runs", []),
                        "base_runs_replicates": test_data.get(
                            "base_runs_replicates", []
                        ),
                        "new_runs_replicates": test_data.get(
                            "new_runs_replicates", []
                        ),
                        "base_measurement_unit": test_data.get(
                            "base_measurement_unit", ""
                        ),
                        "new_measurement_unit": test_data.get(
                            "new_measurement_unit", ""
                        ),
                        "lower_is_better": test_data.get("lower_is_better", True),
                        "is_complete": test_data.get("is_complete", True),
                    }
                )

    return results


def main() -> None:
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

    parser.add_argument("url", nargs="?", help="perf.compare URL or API URL")
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
        api_url, url_search_term = parse_perf_compare_url(args.url)
        search_term = args.search if args.search else url_search_term
        data = fetch_performance_data(api_url, use_replicates, use_cache)

    if search_term:
        data = filter_results_by_search(data, search_term)

    if args.save_data:
        with open(args.save_data, "w") as file_handle:
            json.dump(data, file_handle, indent=2)
        print(f"Data saved to {args.save_data}")

    compute_bootstrap = not args.no_bootstrap
    results = process_results(
        data,
        use_replicates,
        args.limit,
        args.workers,
        compute_bootstrap,
    )

    if not results:
        print("No results to display.")
        sys.exit(1)

    report_title = args.title if args.title else "Performance Comparison Report"
    source_url = args.url if args.url else None
    html_content = generate_html_report(results, report_title, source_url)

    if args.output_file:
        filename = args.output_file
    else:
        filename = f"perf_compare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    with open(filename, "w") as file_handle:
        file_handle.write(html_content)

    print(f"\nInteractive HTML report saved to: {filename}")
    print("Open in browser to view interactive charts")


if __name__ == "__main__":
    main()
