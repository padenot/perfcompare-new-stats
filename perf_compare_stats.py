#!/usr/bin/env python3
"""Reusable statistical analysis for perf.compare datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import numpy as np
from KDEpy import FFTKDE
from scipy.optimize import linear_sum_assignment
from scipy.signal import argrelmax
from scipy.stats import bootstrap
from scipy.stats import ks_2samp, mannwhitneyu, shapiro


PVALUE_THRESHOLD = 0.05


@dataclass(frozen=True)
class BasicStats:
    sample_count: int
    mean: float
    median: float
    variance: float
    standard_deviation: float
    minimum: float
    maximum: float
    shapiro_pvalue: float
    is_likely_normal: bool


@dataclass(frozen=True)
class PairShift:
    base_index: int
    new_index: int
    shift: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class ModeAnalysis:
    base_mode_count: int
    new_mode_count: int
    base_peak_locs: list[float]
    new_peak_locs: list[float]
    base_fracs: list[float]
    new_fracs: list[float]
    base_letters: dict[int, str]
    new_letters: dict[int, str]
    pairs: list[tuple[int, int]]
    unmatched_base: list[int]
    unmatched_new: list[int]
    pair_shifts: list[PairShift]


@dataclass(frozen=True)
class StatisticalComparison:
    base_stats: BasicStats
    new_stats: BasicStats
    ks_pvalue: float
    ks_distributions_differ: bool
    mann_stat: float
    mann_pvalue: float
    cliffs_delta: float
    cliffs_interpretation: str
    cles: float
    cles_direction: str
    mode_analysis: ModeAnalysis


@dataclass(frozen=True)
class ChartPeak:
    value: float
    letter: str


@dataclass(frozen=True)
class ChartSeries:
    median: float
    sample_count: int
    kde_x: list[float]
    kde_y: list[float]
    peaks: list[ChartPeak]


@dataclass(frozen=True)
class ChartData:
    base: ChartSeries
    new: ChartSeries


@dataclass(frozen=True)
class PerformanceAnalysis:
    direction: str
    color_class: str
    effect_size: str
    delta_value: float
    delta_percentage: float
    cliffs_delta: float
    cles: float
    cles_explanation: str
    lower_is_better: bool
    ci_low: float | None
    ci_high: float | None
    bootstrap_time: float
    is_multimodal: bool
    has_per_mode_analysis: bool


@dataclass(frozen=True)
class ProcessedTestResult:
    suite: str
    test: str
    platform: str
    header: str
    unit: str
    lower_is_better: bool
    base_sample_count: int
    new_sample_count: int
    chart_data: ChartData
    statistical_comparison: StatisticalComparison
    perf_analysis: PerformanceAnalysis
    index: int


def summarize_data(series: np.ndarray) -> BasicStats:
    """Summarize statistical data for a one-dimensional series."""
    values = np.asarray(series, dtype=float).flatten()
    if len(values) == 0:
        raise ValueError("Cannot summarize an empty series")

    if len(values) >= 3:
        _, shapiro_pvalue = shapiro(values)
        shapiro_pvalue = float(shapiro_pvalue)
        is_likely_normal = shapiro_pvalue > PVALUE_THRESHOLD
    else:
        shapiro_pvalue = float("nan")
        is_likely_normal = False

    variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    return BasicStats(
        sample_count=len(values),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        variance=variance,
        standard_deviation=standard_deviation,
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        shapiro_pvalue=shapiro_pvalue,
        is_likely_normal=is_likely_normal,
    )


def fit_kde_modes(
    data: np.ndarray,
    valley_threshold: float = 0.5,
    min_peak_fraction: float = 0.05,
    min_data_fraction: float = 0.05,
) -> tuple[int, np.ndarray, np.ndarray | None, np.ndarray | None, list[float]]:
    """Detect modes via KDE with ISJ bandwidth and a relative valley-depth criterion."""
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
        valley = y[prev : nxt + 1].min()
        if valley < valley_threshold * min(y[prev], y[nxt]):
            good.append(nxt)
        elif y[nxt] > y[good[-1]]:
            good[-1] = nxt

    boundaries = []
    for i in range(len(good) - 1):
        seg = y[good[i] : good[i + 1] + 1]
        boundaries.append(float(x[good[i] + np.argmin(seg)]))

    assignments = np.searchsorted(boundaries, data)
    keep = [
        i for i in range(len(good)) if np.mean(assignments == i) >= min_data_fraction
    ]
    if len(keep) < 2:
        return 1, np.array([x[good[np.argmax(y[good])]]]), x, y, []

    good = [good[i] for i in keep]
    boundaries = []
    for i in range(len(good) - 1):
        seg = y[good[i] : good[i + 1] + 1]
        boundaries.append(float(x[good[i] + np.argmin(seg)]))

    return len(good), x[np.array(good)], x, y, boundaries


def split_per_mode(data: np.ndarray, boundaries: list[float]) -> np.ndarray:
    """Assign each data point to a mode index using valley boundaries."""
    return np.searchsorted(boundaries, data)


def match_modes(
    base_locs: np.ndarray,
    base_fracs: list[float],
    new_locs: np.ndarray,
    new_fracs: list[float],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match base modes to new modes using combined location and weight cost."""
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


def bootstrap_median_diff_ci(
    base_data: np.ndarray,
    new_data: np.ndarray,
    n_iter: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, tuple[float, float]]:
    """Bootstrap confidence interval for the median difference."""

    def statistic(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.median(y, axis=axis) - np.median(x, axis=axis)

    res = bootstrap(
        (base_data, new_data),
        statistic=statistic,
        n_resamples=n_iter,
        confidence_level=1 - alpha,
        method="percentile",
        vectorized=True,
        paired=False,
        random_state=42,
    )
    return (
        float(np.median(new_data) - np.median(base_data)),
        (float(res.confidence_interval.low), float(res.confidence_interval.high)),
    )


def interpret_effect_size(effect_size: float) -> str:
    """Interpret Cliff's delta effect size."""
    magnitude = abs(effect_size)
    if magnitude < 0.15:
        return "Negligible difference"
    if magnitude < 0.33:
        return "Small difference"
    if magnitude < 0.47:
        return "Moderate difference"
    return "Large difference"


def effect_size_label(effect_size: float) -> str:
    """Return the categorical effect size label used by the report."""
    return interpret_effect_size(effect_size).replace(" difference", "")


def prepare_data_for_pipeline(
    item: dict[str, Any], use_replicates: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare data from API response for analysis."""
    if use_replicates:
        base_replicates = item.get("base_runs_replicates", [])
        new_replicates = item.get("new_runs_replicates", [])

        if base_replicates and new_replicates:
            base_data = np.array(base_replicates, dtype=float).flatten()
            new_data = np.array(new_replicates, dtype=float).flatten()
        else:
            base_data = np.array(item.get("base_runs", []), dtype=float)
            new_data = np.array(item.get("new_runs", []), dtype=float)
    else:
        base_data = np.array(item.get("base_runs", []), dtype=float)
        new_data = np.array(item.get("new_runs", []), dtype=float)

    return base_data, new_data


def _mode_fractions(
    data: np.ndarray, boundaries: list[float], n_modes: int
) -> list[float]:
    assignments = split_per_mode(data, boundaries)
    return [float(np.mean(assignments == i)) for i in range(n_modes)]


def _assign_mode_letters(locs: np.ndarray, lower_is_better: bool) -> dict[int, str]:
    sorted_idxs = sorted(
        range(len(locs)),
        key=lambda i: float(locs[i]) if lower_is_better else -float(locs[i]),
    )
    return {idx: chr(ord("A") + rank) for rank, idx in enumerate(sorted_idxs)}


def build_mode_analysis(
    base_data: np.ndarray,
    new_data: np.ndarray,
    lower_is_better: bool,
) -> ModeAnalysis:
    """Build multimodal analysis for a base/new comparison."""
    base_mode_count, base_peak_locs, _, _, base_boundaries = fit_kde_modes(base_data)
    new_mode_count, new_peak_locs, _, _, new_boundaries = fit_kde_modes(new_data)

    base_fracs = _mode_fractions(base_data, base_boundaries, base_mode_count)
    new_fracs = _mode_fractions(new_data, new_boundaries, new_mode_count)

    per_mode_base = split_per_mode(base_data, base_boundaries)
    per_mode_new = split_per_mode(new_data, new_boundaries)
    pairs, unmatched_base, unmatched_new = match_modes(
        base_peak_locs,
        base_fracs,
        new_peak_locs,
        new_fracs,
    )

    base_letters = _assign_mode_letters(base_peak_locs, lower_is_better)
    new_letters = {new_i: base_letters[base_i] for base_i, new_i in pairs}
    next_ord = ord("A") + base_mode_count
    for new_i in unmatched_new:
        new_letters[new_i] = chr(next_ord)
        next_ord += 1

    pair_shifts: list[PairShift] = []
    for base_i, new_i in pairs:
        ref_vals = base_data[per_mode_base == base_i]
        candidate_vals = new_data[per_mode_new == new_i]

        if len(ref_vals) < 2 or len(candidate_vals) < 2:
            continue

        shift, (ci_low, ci_high) = bootstrap_median_diff_ci(ref_vals, candidate_vals)
        pair_shifts.append(
            PairShift(
                base_index=base_i,
                new_index=new_i,
                shift=shift,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )

    return ModeAnalysis(
        base_mode_count=base_mode_count,
        new_mode_count=new_mode_count,
        base_peak_locs=[float(value) for value in base_peak_locs],
        new_peak_locs=[float(value) for value in new_peak_locs],
        base_fracs=base_fracs,
        new_fracs=new_fracs,
        base_letters=base_letters,
        new_letters=new_letters,
        pairs=pairs,
        unmatched_base=unmatched_base,
        unmatched_new=unmatched_new,
        pair_shifts=pair_shifts,
    )


def analyze_statistical_comparison(
    base_data: np.ndarray,
    new_data: np.ndarray,
    lower_is_better: bool,
) -> StatisticalComparison:
    """Compute the reusable statistical comparison model for one test."""
    from cliffs_delta import cliffs_delta

    base_stats = summarize_data(base_data)
    new_stats = summarize_data(new_data)

    _, ks_pvalue = ks_2samp(base_data, new_data)
    mann_stat, mann_pvalue = mannwhitneyu(base_data, new_data, alternative="two-sided")

    cliffs_delta_value, _ = cliffs_delta(base_data, new_data)
    cles = float(mann_stat / (len(base_data) * len(new_data)))
    if cles >= 0.5:
        cles_direction = (
            f"{cles:.2f} -> {cles * 100:.0f}% chance a value from without patch "
            f"is greater than a value from with patch"
        )
    else:
        cles_direction = (
            f"{1 - cles:.2f} -> {100 - cles * 100:.0f}% chance a value from with patch "
            f"is greater than a value from without patch"
        )

    mode_analysis = build_mode_analysis(base_data, new_data, lower_is_better)

    return StatisticalComparison(
        base_stats=base_stats,
        new_stats=new_stats,
        ks_pvalue=float(ks_pvalue),
        ks_distributions_differ=bool(ks_pvalue < PVALUE_THRESHOLD),
        mann_stat=float(mann_stat),
        mann_pvalue=float(mann_pvalue),
        cliffs_delta=float(cliffs_delta_value),
        cliffs_interpretation=interpret_effect_size(cliffs_delta_value),
        cles=cles,
        cles_direction=cles_direction,
        mode_analysis=mode_analysis,
    )


def build_chart_data(
    base_data: np.ndarray,
    new_data: np.ndarray,
    lower_is_better: bool,
    mode_analysis: ModeAnalysis | None = None,
) -> ChartData:
    """Build chart-ready KDE and mode markers for the display layer."""
    if len(base_data) == 0 and len(new_data) == 0:
        raise ValueError("Cannot build chart data without any samples")

    base_median = float(np.median(base_data)) if len(base_data) > 0 else 0.0
    new_median = float(np.median(new_data)) if len(new_data) > 0 else 0.0

    all_data = (
        np.concatenate([base_data, new_data])
        if len(base_data) > 0 and len(new_data) > 0
        else (base_data if len(base_data) > 0 else new_data)
    )
    x_min, x_max = float(np.min(all_data)), float(np.max(all_data))
    padding = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    x_grid = np.linspace(x_min - padding, x_max + padding, 200)

    base_peaks: list[ChartPeak] = []
    new_peaks: list[ChartPeak] = []

    if mode_analysis is not None:
        if len(base_data) > 1:
            base_peaks = [
                ChartPeak(value=float(loc), letter=mode_analysis.base_letters[i])
                for i, loc in enumerate(mode_analysis.base_peak_locs)
            ]
        if len(new_data) > 1:
            new_peaks = [
                ChartPeak(value=float(loc), letter=mode_analysis.new_letters[i])
                for i, loc in enumerate(mode_analysis.new_peak_locs)
            ]
    else:
        if len(base_data) > 1:
            _, base_locs, _, _, _ = fit_kde_modes(base_data)
            base_letters = _assign_mode_letters(base_locs, lower_is_better)
            base_peaks = [
                ChartPeak(value=float(loc), letter=base_letters[i])
                for i, loc in enumerate(base_locs)
            ]
        if len(new_data) > 1:
            _, new_locs, _, _, _ = fit_kde_modes(new_data)
            new_letters = _assign_mode_letters(new_locs, lower_is_better)
            new_peaks = [
                ChartPeak(value=float(loc), letter=new_letters[i])
                for i, loc in enumerate(new_locs)
            ]

    base_kde_x: list[float] = []
    base_kde_y: list[float] = []
    new_kde_x: list[float] = []
    new_kde_y: list[float] = []

    try:
        if len(base_data) > 1:
            y_base = FFTKDE(bw="ISJ").fit(base_data).evaluate(x_grid)
            base_kde_x = x_grid.tolist()
            base_kde_y = np.asarray(y_base).tolist()

        if len(new_data) > 1:
            y_new = FFTKDE(bw="ISJ").fit(new_data).evaluate(x_grid)
            new_kde_x = x_grid.tolist()
            new_kde_y = np.asarray(y_new).tolist()
    except Exception:
        pass

    return ChartData(
        base=ChartSeries(
            median=base_median,
            sample_count=len(base_data),
            kde_x=base_kde_x,
            kde_y=base_kde_y,
            peaks=base_peaks,
        ),
        new=ChartSeries(
            median=new_median,
            sample_count=len(new_data),
            kde_x=new_kde_x,
            kde_y=new_kde_y,
            peaks=new_peaks,
        ),
    )


def analyze_performance_change(
    base_data: np.ndarray,
    new_data: np.ndarray,
    lower_is_better: bool,
    mode_analysis: ModeAnalysis,
    compute_bootstrap: bool = True,
) -> PerformanceAnalysis:
    """Produce the report-level summary metrics for one comparison."""
    from cliffs_delta import cliffs_delta

    if len(base_data) == 0 or len(new_data) == 0:
        return PerformanceAnalysis(
            direction="No data",
            color_class="neutral",
            effect_size="Unknown",
            delta_value=0.0,
            delta_percentage=0.0,
            cliffs_delta=0.0,
            cles=0.5,
            cles_explanation="No data for comparison",
            lower_is_better=lower_is_better,
            ci_low=None,
            ci_high=None,
            bootstrap_time=0.0,
            is_multimodal=False,
            has_per_mode_analysis=False,
        )

    base_median = float(np.median(base_data))
    new_median = float(np.median(new_data))
    delta_value = new_median - base_median
    delta_percentage = (delta_value / base_median * 100) if base_median != 0 else 0.0

    cliffs_delta_value, _ = cliffs_delta(base_data, new_data)
    mann_stat, _ = mannwhitneyu(base_data, new_data, alternative="two-sided")
    cles = float(mann_stat / (len(base_data) * len(new_data)))

    if cles >= 0.5:
        cles_explanation = f"{cles:.0%} chance the new value is smaller than the base value"
    else:
        cles_explanation = (
            f"{1 - cles:.0%} chance the new value is greater than the base value"
        )

    is_multimodal = (
        mode_analysis.base_mode_count > 1 or mode_analysis.new_mode_count > 1
    )
    has_per_mode_analysis = (
        mode_analysis.base_mode_count == mode_analysis.new_mode_count and is_multimodal
    )

    bootstrap_time = 0.0
    ci_low: float | None = None
    ci_high: float | None = None
    if (
        compute_bootstrap
        and not has_per_mode_analysis
        and len(base_data) >= 2
        and len(new_data) >= 2
    ):
        start_time = time.perf_counter()
        _, (ci_low, ci_high) = bootstrap_median_diff_ci(base_data, new_data, n_iter=1000)
        bootstrap_time = time.perf_counter() - start_time

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

    return PerformanceAnalysis(
        direction=direction,
        color_class=color_class,
        effect_size=effect_size_label(cliffs_delta_value),
        delta_value=float(delta_value),
        delta_percentage=float(delta_percentage),
        cliffs_delta=float(cliffs_delta_value),
        cles=cles,
        cles_explanation=cles_explanation,
        lower_is_better=lower_is_better,
        ci_low=ci_low,
        ci_high=ci_high,
        bootstrap_time=bootstrap_time,
        is_multimodal=is_multimodal,
        has_per_mode_analysis=has_per_mode_analysis,
    )


def analyze_test_item(
    item: dict[str, Any],
    index: int,
    use_replicates: bool = True,
    compute_bootstrap: bool = True,
) -> ProcessedTestResult | None:
    """Run the full reusable stats pipeline for a single API item."""
    suite = item.get("suite", "unknown")
    test = item.get("test", item.get("header_name", ""))
    platform = item.get("platform", "unknown")

    base_data, new_data = prepare_data_for_pipeline(item, use_replicates)
    if len(base_data) == 0 or len(new_data) == 0:
        return None

    lower_is_better = bool(item.get("lower_is_better", True))
    unit = item.get("base_measurement_unit", "ms") or "ms"
    header = f"{suite} - {test} ({platform})"

    statistical_comparison = analyze_statistical_comparison(
        base_data,
        new_data,
        lower_is_better,
    )
    perf_analysis = analyze_performance_change(
        base_data,
        new_data,
        lower_is_better,
        statistical_comparison.mode_analysis,
        compute_bootstrap,
    )
    chart_data = build_chart_data(
        base_data,
        new_data,
        lower_is_better,
        statistical_comparison.mode_analysis,
    )

    return ProcessedTestResult(
        suite=suite,
        test=test,
        platform=platform,
        header=header,
        unit=unit,
        lower_is_better=lower_is_better,
        base_sample_count=len(base_data),
        new_sample_count=len(new_data),
        chart_data=chart_data,
        statistical_comparison=statistical_comparison,
        perf_analysis=perf_analysis,
        index=index,
    )
