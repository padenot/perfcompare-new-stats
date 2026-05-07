#!/usr/bin/env python3
"""Display-only rendering for perf.compare statistical results."""

from __future__ import annotations

from datetime import datetime
import html
import json
import math

import numpy as np

from perf_compare_stats import ProcessedTestResult


def generate_explanation(result: ProcessedTestResult) -> str:
    """Produce a plain-English summary of the performance change."""
    mode_analysis = result.statistical_comparison.mode_analysis
    base_peak_locs = np.array(mode_analysis.base_peak_locs)
    new_peak_locs = np.array(mode_analysis.new_peak_locs)
    base_fracs = mode_analysis.base_fracs
    new_fracs = mode_analysis.new_fracs
    pairs = mode_analysis.pairs
    unmatched_base = mode_analysis.unmatched_base
    unmatched_new = mode_analysis.unmatched_new
    pair_shifts = mode_analysis.pair_shifts
    lower_is_better = result.lower_is_better
    unit = result.unit
    base_letter = mode_analysis.base_letters
    new_letter = mode_analysis.new_letters

    def fmt_loc(value: float) -> str:
        absolute = abs(value)
        if absolute < 10:
            return f"{value:.2f}"
        if absolute < 100:
            return f"{value:.1f}"
        return f"{value:.0f}"

    def frac_words(fraction: float) -> str:
        if fraction >= 0.92:
            return "virtually all"
        if fraction >= 0.75:
            return "the large majority"
        if fraction >= 0.55:
            return "most"
        if fraction >= 0.45:
            return "roughly half"
        if fraction >= 0.25:
            return "a significant portion"
        if fraction >= 0.10:
            return "a minority"
        return "a small fraction"

    def shift_words(shift: float, ci_low: float, ci_high: float) -> tuple[bool | None, str]:
        if ci_low < 0 < ci_high:
            return (
                None,
                f"no statistically significant change (the ~{fmt_loc(abs(shift))}{unit} difference is within noise)",
            )
        improved = (shift < 0) == lower_is_better
        lo = min(abs(ci_low), abs(ci_high))
        hi = max(abs(ci_low), abs(ci_high))
        word = "improved" if improved else "regressed"
        return improved, f"{word} by {fmt_loc(lo)}-{fmt_loc(hi)}{unit}"

    def speed_label(loc: float, locs: list[float]) -> str:
        sorted_locs = sorted(locs) if lower_is_better else sorted(locs, reverse=True)
        rank = sorted_locs.index(loc)
        if len(sorted_locs) == 1:
            return ""
        if len(sorted_locs) == 2:
            return "fast" if rank == 0 else "slow"
        if rank == 0:
            return "fastest"
        if rank == len(sorted_locs) - 1:
            return "slowest"
        return "intermediate"

    n_base = len(base_peak_locs)
    n_new = len(new_peak_locs)
    all_locs = np.concatenate([base_peak_locs, new_peak_locs])
    overall_center = float(np.median(all_locs))

    total_modes = n_base + len(unmatched_new)
    if total_modes > 3:
        return (
            f"This benchmark shows {total_modes} distinct execution patterns, "
            f"which is too complex to summarise in plain language. "
            f"Please refer to the KDE chart for a visual overview."
        )

    if n_base == 1 and n_new == 1 and len(pair_shifts) == 1:
        pair_shift = pair_shifts[0]
        is_improvement, phrase = shift_words(
            pair_shift.shift,
            pair_shift.ci_low,
            pair_shift.ci_high,
        )
        base_loc = float(base_peak_locs[0])
        new_loc = float(new_peak_locs[0])
        if is_improvement is None:
            return (
                f"Runs behaved consistently in both revisions. "
                f"The typical time ({fmt_loc(base_loc)}{unit} -> {fmt_loc(new_loc)}{unit}) shows {phrase}."
            )
        if is_improvement:
            return (
                f"Performance improved clearly and consistently: "
                f"the typical run went from {fmt_loc(base_loc)}{unit} to {fmt_loc(new_loc)}{unit} ({phrase})."
            )
        return (
            f"Performance regressed: "
            f"the typical run slowed from {fmt_loc(base_loc)}{unit} to {fmt_loc(new_loc)}{unit} ({phrase})."
        )

    parts: list[str] = []
    if n_base == 1:
        parts.append(
            f"The base revision ran consistently at around {fmt_loc(float(base_peak_locs[0]))}{unit}."
        )
    else:
        descs = [
            f"<b>Mode {base_letter.get(i, chr(ord('A') + i))}</b> at ~{fmt_loc(float(loc))}{unit} ({fraction:.1%})"
            for i, (loc, fraction) in enumerate(zip(base_peak_locs, base_fracs))
        ]
        parts.append(
            f"The base revision showed {n_base} distinct execution patterns: {', and '.join(descs)}."
        )

    mode_lines: list[str] = []
    pair_lookup = {(pair.base_index, pair.new_index): pair for pair in pair_shifts}
    sort_key = (
        lambda pair: float(base_peak_locs[pair[0]])
        if lower_is_better
        else -float(base_peak_locs[pair[0]])
    )
    for base_i, new_i in sorted(pairs, key=sort_key):
        pair_shift = pair_lookup.get((base_i, new_i))
        if pair_shift is None:
            continue

        base_loc = float(base_peak_locs[base_i])
        new_loc = float(new_peak_locs[new_i])
        base_frac = base_fracs[base_i]
        new_frac = new_fracs[new_i]
        letter = base_letter.get(base_i, "?")
        slabel = speed_label(base_loc, [float(value) for value in base_peak_locs])
        _, phrase = shift_words(pair_shift.shift, pair_shift.ci_low, pair_shift.ci_high)

        delta_frac = new_frac - base_frac
        if abs(delta_frac) >= 0.15:
            dir_word = "more" if delta_frac > 0 else "less"
            frac_tail = (
                f" It is now {dir_word} common: {base_frac:.1%} -> {new_frac:.1%} of runs."
            )
        else:
            frac_tail = ""

        mode_lines.append(
            f"<b>Mode {letter}</b> ({slabel}, ~{fmt_loc(base_loc)}{unit}, {base_frac:.1%} of base runs) "
            f"{phrase} - now at ~{fmt_loc(new_loc)}{unit}.{frac_tail}"
        )

    for base_i in unmatched_base:
        loc = float(base_peak_locs[base_i])
        fraction = base_fracs[base_i]
        letter = base_letter.get(base_i, "?")
        slabel = speed_label(loc, [float(value) for value in base_peak_locs])
        is_slow = (lower_is_better and loc >= overall_center) or (
            not lower_is_better and loc <= overall_center
        )
        valence = (
            "This is a positive change - that slow behavior has been eliminated."
            if is_slow
            else "A previously fast pattern no longer occurs - this may be worth investigating."
        )
        mode_lines.append(
            f"<b>Mode {letter}</b> ({slabel}, ~{fmt_loc(loc)}{unit}, {frac_words(fraction)} of base runs) "
            f"is absent from the new revision. {valence}"
        )

    for new_i in unmatched_new:
        loc = float(new_peak_locs[new_i])
        fraction = new_fracs[new_i]
        letter = new_letter.get(new_i, "?")
        is_slow = (lower_is_better and loc >= overall_center) or (
            not lower_is_better and loc <= overall_center
        )
        valence = (
            "This new slow behavior warrants investigation."
            if is_slow
            else "This new fast pattern is a positive development."
        )
        mode_lines.append(
            f"<b>Mode {letter}</b> (new, ~{fmt_loc(loc)}{unit}, {frac_words(fraction)} of new runs): {valence}"
        )

    if mode_lines:
        parts.append("<br>".join(mode_lines))

    return "<br><br>".join(parts)


def render_statistical_analysis(result: ProcessedTestResult) -> str:
    """Render the statistical model as a plain-text report."""
    comparison = result.statistical_comparison
    mode_analysis = comparison.mode_analysis

    def fmt_value(value: float) -> str:
        if math.isnan(value):
            return "n/a"
        absolute = abs(value)
        if absolute < 10:
            return f"{value:.3f}"
        if absolute < 100:
            return f"{value:.2f}"
        return f"{value:.1f}"

    def fmt_pvalue(value: float) -> str:
        return "n/a" if math.isnan(value) else f"{value:.4f}"

    def stats_block(label: str, stats) -> list[str]:
        normality = (
            "likely normal" if stats.is_likely_normal else "not normal"
            if not math.isnan(stats.shapiro_pvalue)
            else "normality unavailable"
        )
        return [
            f"{label}:",
            f"  Samples: {stats.sample_count}",
            f"  Mean: {fmt_value(stats.mean)}",
            f"  Median: {fmt_value(stats.median)}",
            f"  Variance: {fmt_value(stats.variance)}",
            f"  Standard deviation: {fmt_value(stats.standard_deviation)}",
            f"  Min/Max: {fmt_value(stats.minimum)} / {fmt_value(stats.maximum)}",
            f"  Shapiro-Wilk p-value: {fmt_pvalue(stats.shapiro_pvalue)} -> {normality}",
        ]

    lines = ["Basic statistics and normality test"]
    lines.extend(stats_block("without patch", comparison.base_stats))
    lines.append("")
    lines.extend(stats_block("with patch", comparison.new_stats))
    lines.append("")
    lines.append("Distribution comparison")
    lines.append(f"  KS test p-value: {comparison.ks_pvalue:.4f}")
    if comparison.ks_distributions_differ:
        lines.append("  Distributions differ enough that the KDE view matters.")
    lines.append(f"  Mann-Whitney U p-value: {comparison.mann_pvalue:.4f}")
    lines.append(f"  Common language effect size: {comparison.cles_direction}")
    lines.append(
        f"  Cliff's delta: {comparison.cliffs_delta:+.3f} -> {comparison.cliffs_interpretation}"
    )
    lines.append("")

    base_mode_str = ", ".join(
        f"{mode_analysis.base_letters[i]}: {loc:.1f} ({fraction:.0%})"
        for i, (loc, fraction) in enumerate(
            zip(mode_analysis.base_peak_locs, mode_analysis.base_fracs)
        )
    )
    new_mode_str = ", ".join(
        f"{mode_analysis.new_letters[i]}: {loc:.1f} ({fraction:.0%})"
        for i, (loc, fraction) in enumerate(
            zip(mode_analysis.new_peak_locs, mode_analysis.new_fracs)
        )
    )
    lines.append(f"Modes (Base): {mode_analysis.base_mode_count} -> {base_mode_str}")
    lines.append(f"Modes (New):  {mode_analysis.new_mode_count} -> {new_mode_str}")
    if mode_analysis.base_mode_count > 1:
        lines.append("  Base revision distribution appears multimodal.")
    if mode_analysis.new_mode_count > 1:
        lines.append("  New revision distribution appears multimodal.")

    pair_lookup = {
        (pair.base_index, pair.new_index): pair for pair in mode_analysis.pair_shifts
    }
    for base_i, new_i in mode_analysis.pairs:
        letter = mode_analysis.base_letters[base_i]
        base_loc = mode_analysis.base_peak_locs[base_i]
        new_loc = mode_analysis.new_peak_locs[new_i]
        pair_shift = pair_lookup.get((base_i, new_i))
        if pair_shift is None:
            lines.append(f"Mode {letter}: Not enough data to compare.")
            continue

        lines.append(
            f"Mode {letter} ({base_loc:.1f} -> {new_loc:.1f}, "
            f"base {mode_analysis.base_fracs[base_i]:.0%}, new {mode_analysis.new_fracs[new_i]:.0%}):"
        )
        lines.append(
            f"  Median shift: {pair_shift.shift:+.3f} "
            f"(95% CI: {pair_shift.ci_low:+.3f} to {pair_shift.ci_high:+.3f})"
        )
        if pair_shift.ci_low > 0:
            direction = (
                "Performance regressed (median increased)"
                if result.lower_is_better
                else "Performance improved (median increased)"
            )
        elif pair_shift.ci_high < 0:
            direction = (
                "Performance improved (median decreased)"
                if result.lower_is_better
                else "Performance regressed (median decreased)"
            )
        else:
            direction = "No significant shift"
        lines.append(f"  {direction}")

    for base_i in mode_analysis.unmatched_base:
        lines.append(
            f"Mode {mode_analysis.base_letters[base_i]} "
            f"({mode_analysis.base_peak_locs[base_i]:.1f}, {mode_analysis.base_fracs[base_i]:.0%} of base): "
            f"absent in new revision"
        )

    for new_i in mode_analysis.unmatched_new:
        lines.append(
            f"Mode {mode_analysis.new_letters[new_i]} "
            f"({mode_analysis.new_peak_locs[new_i]:.1f}, {mode_analysis.new_fracs[new_i]:.0%} of new): "
            f"newly appeared"
        )

    return "\n".join(lines)


def categorize_results(
    results: list[ProcessedTestResult],
) -> dict[str, dict[str, list[ProcessedTestResult]]]:
    """Categorize and sort results by performance impact."""
    categories = {
        "better": {"Large": [], "Moderate": [], "Small": [], "Negligible": []},
        "worse": {"Large": [], "Moderate": [], "Small": [], "Negligible": []},
        "neutral": {"No change": []},
    }

    for result in results:
        perf = result.perf_analysis
        direction = perf.direction.lower()
        effect_size = perf.effect_size

        if direction in {"better", "worse"} and effect_size in categories[direction]:
            categories[direction][effect_size].append(result)
        else:
            categories["neutral"]["No change"].append(result)

    return categories


def _chart_data_to_dict(result: ProcessedTestResult) -> dict[str, object]:
    chart_data = result.chart_data
    return {
        "base": {
            "median": chart_data.base.median,
            "sample_count": chart_data.base.sample_count,
            "kde_x": chart_data.base.kde_x,
            "kde_y": chart_data.base.kde_y,
            "peaks": [
                {"value": peak.value, "letter": peak.letter}
                for peak in chart_data.base.peaks
            ],
        },
        "new": {
            "median": chart_data.new.median,
            "sample_count": chart_data.new.sample_count,
            "kde_x": chart_data.new.kde_x,
            "kde_y": chart_data.new.kde_y,
            "peaks": [
                {"value": peak.value, "letter": peak.letter}
                for peak in chart_data.new.peaks
            ],
        },
    }


def _test_dom_id(result: ProcessedTestResult) -> str:
    return (
        f"{result.suite}-{result.test}-{result.platform}"
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .lower()
    )


def generate_html_report(
    results: list[ProcessedTestResult],
    title: str = "Performance Comparison Report",
    source_url: str | None = None,
) -> str:
    """Generate the interactive HTML report."""
    categories = categorize_results(results)
    chart_data = {
        f"chart-{result.index}": _chart_data_to_dict(result) for result in results
    }

    index_html: list[str] = []
    for direction in ["better", "worse", "neutral"]:
        direction_title = {
            "better": "Performance Improvements",
            "worse": "Performance Regressions",
            "neutral": "No Significant Change",
        }[direction]
        direction_categories = categories[direction]
        direction_total = sum(len(items) for items in direction_categories.values())

        if direction_total == 0:
            continue

        index_html.append('<div class="index-category">')
        index_html.append(
            f'<div class="index-title {direction}">{direction_title} '
            f'<span class="index-count">({direction_total} tests)</span></div>'
        )
        index_html.append('<div class="index-links">')

        for effect_size, items in direction_categories.items():
            if not items:
                continue
            anchor = f"{direction}-{effect_size.lower().replace(' ', '-')}"
            count = len(items)
            index_html.append(
                f'<a href="#{anchor}" class="index-link">{effect_size} '
                f'<span class="index-count">({count})</span></a>'
            )

        index_html.append("</div></div>")

    test_results_html: list[str] = []
    for direction in ["better", "worse", "neutral"]:
        direction_categories = categories[direction]
        direction_total = sum(len(items) for items in direction_categories.values())
        if direction_total == 0:
            continue

        direction_title = {
            "better": "Performance Improvements",
            "worse": "Performance Regressions",
            "neutral": "No Significant Change",
        }[direction]

        test_results_html.append(f'<div class="category-{direction}">')
        test_results_html.append(f"<h2>{direction_title}</h2>")

        for effect_size in ["Large", "Moderate", "Small", "Negligible", "No change"]:
            items = direction_categories.get(effect_size, [])
            if not items:
                continue

            anchor = f"{direction}-{effect_size.lower().replace(' ', '-')}"
            test_results_html.append(
                f'<h3 id="{anchor}">{effect_size} Effect ({len(items)} tests)</h3>'
            )

            for item in items:
                perf = item.perf_analysis
                explanation = generate_explanation(item)
                statistical_analysis = html.escape(render_statistical_analysis(item))
                full_header = (
                    f"{item.suite} - {item.test}" if item.test else item.suite
                )
                direction_text = (
                    f"{'Lower' if perf.lower_is_better else 'Higher'} is better"
                )

                test_results_html.append(
                    f"""
                        <details class="test-result {perf.color_class}" id="{_test_dom_id(item)}"
                            data-cles="{perf.cles:.6f}"
                            data-delta-pct="{perf.delta_percentage:.4f}"
                            data-delta-abs="{perf.delta_value:.6f}"
                            data-cliffs-delta="{perf.cliffs_delta:.4f}"
                            data-lower-is-better="{1 if perf.lower_is_better else 0}">
                            <summary class="test-header {perf.color_class}">
                                <div class="test-title">{html.escape(full_header)}
                                    <span class="perf-badge {perf.color_class}">
                                        {perf.direction} ({perf.effect_size}) {perf.delta_percentage:+.1f}%
                                    </span>
                                </div>
                                <div class="test-meta">
                                    Platform: {html.escape(item.platform)} |
                                    Samples: Base={item.base_sample_count}, New={item.new_sample_count}
                                </div>
                                <div class="direction-indicator">
                                    {direction_text}
                                </div>
                                <span class="expand-indicator">▶</span>
                            </summary>

                            <div class="details-body">
                            <div class="stats-grid">
                                <div class="stats-card">
                                    <div class="stats-title">Base (Without Patch)</div>
                                    <div class="stat-row"><span>Samples:</span><span>{item.chart_data.base.sample_count}</span></div>
                                    <div class="stat-row"><span>Median:</span><span>{item.chart_data.base.median:.3f}</span></div>
                                </div>
                                <div class="stats-card">
                                    <div class="stats-title">New (With Patch)</div>
                                    <div class="stat-row"><span>Samples:</span><span>{item.chart_data.new.sample_count}</span></div>
                                    <div class="stat-row"><span>Median:</span><span>{item.chart_data.new.median:.3f}</span></div>
                                    <div class="stat-row"><span>Delta:</span><span>{perf.delta_value:+.3f} ({perf.delta_percentage:+.1f}%)</span></div>
                                    {"" if perf.ci_low is None else f'<div class="stat-row"><span>95% CI:</span><span>[{perf.ci_low:+.3f}, {perf.ci_high:+.3f}]</span></div>'}
                                    <div class="stat-row"><span>Cliff&apos;s δ:</span><span>{perf.cliffs_delta:+.3f}</span></div>
                                </div>
                            </div>

                            <div class="cles-section">
                                <div class="cles-explanation">
                                    <strong>Effect Size:</strong> {html.escape(perf.cles_explanation)}<br/>
                                    {"" if perf.ci_low is None else f'<strong>Confidence Interval:</strong> We are 95% confident the median difference is between <strong>{perf.ci_low:+.3f}</strong> and <strong>{perf.ci_high:+.3f}</strong>'}
                                </div>
                            </div>

                            {f'<div class="interpretation-section"><div class="interpretation-text">{explanation}</div></div>' if explanation else ""}

                            <div class="analysis-section">
                                <h3>Statistical Analysis</h3>
                                <div class="analysis-text">{statistical_analysis}</div>
                            </div>

                            <div class="chart-container" id="chart-{item.index}"></div>
                            </div>
                        </details>
                    """
                )

        test_results_html.append("</div>")

    source_url_html = (
        f'<div class="source-url"><a href="{html.escape(source_url)}" target="_blank">{html.escape(source_url)}</a></div>'
        if source_url
        else ""
    )
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
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
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .test-result.better {{ border-left: 6px solid #27ae60; }}
        .test-result.worse {{ border-left: 6px solid #e74c3c; }}
        .test-result.neutral {{ border-left: 6px solid #7f8c8d; }}
        details.test-result > summary {{
            list-style: none;
            cursor: pointer;
            display: block;
        }}
        details.test-result > summary::-webkit-details-marker {{ display: none; }}
        .details-body {{
            padding: 15px;
        }}
        .test-header {{
            color: white;
            padding: 15px;
            position: relative;
        }}
        .test-header.better {{ background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); }}
        .test-header.worse {{ background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }}
        .test-header.neutral {{ background: linear-gradient(135deg, #7f8c8d 0%, #95a5a6 100%); }}
        .expand-indicator {{
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1.1em;
            opacity: 0.85;
            transition: transform 0.15s;
        }}
        details[open] > summary .expand-indicator {{
            transform: translateY(-50%) rotate(90deg);
        }}
        .expand-collapse-btns {{
            margin-bottom: 15px;
        }}
        .expand-collapse-btns button {{
            margin-right: 8px;
            padding: 6px 14px;
            background: #3498db;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 0.95em;
        }}
        .expand-collapse-btns button:hover {{ background: #2980b9; }}
        .sort-controls {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-left: 16px;
        }}
        .sort-controls label {{ font-size: 0.95em; }}
        .sort-controls select {{
            padding: 5px 10px;
            border-radius: 4px;
            border: 1px solid #ccc;
            font-size: 0.95em;
            cursor: pointer;
        }}

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
    <h1>{html.escape(title)}</h1>
    {source_url_html}

    <div class="summary">
        <div class="metric">
            <span class="metric-label">Total Tests:</span>
            <span class="metric-value">{len(results)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Generated:</span>
            <span class="metric-value">{generated_at}</span>
        </div>
    </div>

    <div class="expand-collapse-btns">
        <button id="expand-all">Expand all</button>
        <button id="collapse-all">Collapse all</button>
        <span class="sort-controls">
            <label for="sort-by">Sort by:</label>
            <select id="sort-by">
                <option value="default">Default</option>
                <option value="cles">Common language effect size</option>
                <option value="deltaPct">% delta</option>
                <option value="deltaAbs">Delta</option>
                <option value="cliffsDelta">Cliff's δ</option>
            </select>
            <label><input type="checkbox" id="sort-abs"> |absolute|</label>
            <label><input type="checkbox" id="sort-asc"> ascending</label>
        </span>
    </div>
    <div id="sorted-view" style="display:none"></div>

    <div class="index">
        <h2>Test Results Index</h2>
        {"".join(index_html)}
    </div>

    {"".join(test_results_html)}

    <script>
    const chartDataAll = {json.dumps(chart_data, indent=2)};
    const initializedCharts = new Set();

    function fmtVal(v) {{
        const a = Math.abs(v);
        if (a < 10) return v.toFixed(2);
        if (a < 100) return v.toFixed(1);
        return v.toFixed(0);
    }}

    function initChartIfNeeded(containerId) {{
        if (initializedCharts.has(containerId) || !chartDataAll[containerId]) return;
        initializedCharts.add(containerId);
        createChart(containerId, chartDataAll[containerId]);
    }}

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
                        if (param.seriesName.includes('Patch)')) {{
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
                ...(!baseData.peaks || baseData.peaks.length === 0 ? [{{
                    name: 'Base Median',
                    type: 'line',
                    markLine: {{
                        silent: true,
                        data: [{{ xAxis: baseData.median, lineStyle: {{ color: '#3498db', type: 'dashed', width: 2 }}, label: {{ formatter: `Base: ${{fmtVal(baseData.median)}}` }} }}]
                    }}
                }}] : []),
                ...(!newData.peaks || newData.peaks.length === 0 ? [{{
                    name: 'New Median',
                    type: 'line',
                    markLine: {{
                        silent: true,
                        data: [{{ xAxis: newData.median, lineStyle: {{ color: '#e67e22', type: 'dashed', width: 2 }}, label: {{ formatter: `New: ${{fmtVal(newData.median)}}` }} }}]
                    }}
                }}] : []),
                ...((baseData.peaks || []).map((p, i) => ({{
                    name: `Base ${{p.letter}}`,
                    type: 'line',
                    markLine: {{
                        silent: true,
                        data: [{{ xAxis: p.value, lineStyle: {{ color: '#3498db', type: 'solid', width: 2 }}, label: {{ formatter: `Base ${{p.letter}}: ${{fmtVal(p.value)}}`, distance: [0, modeGlobalIdx[`base_${{i}}`] * 15], color: '#3498db' }} }}]
                    }}
                }}))),
                ...((newData.peaks || []).map((p, i) => ({{
                    name: `New ${{p.letter}}`,
                    type: 'line',
                    markLine: {{
                        silent: true,
                        data: [{{ xAxis: p.value, lineStyle: {{ color: '#e67e22', type: 'solid', width: 2 }}, label: {{ formatter: `New ${{p.letter}}: ${{fmtVal(p.value)}}`, distance: [0, modeGlobalIdx[`new_${{i}}`] * 15], color: '#e67e22' }} }}]
                    }}
                }}))),
            ]
        }};

        chart.setOption(option);
        window.addEventListener('resize', function() {{
            chart.resize();
        }});
        return chart;
    }}

    document.addEventListener('DOMContentLoaded', function() {{
        document.querySelectorAll('details.test-result').forEach(details => {{
            details.addEventListener('toggle', function() {{
                if (this.open) {{
                    const chartEl = this.querySelector('.chart-container');
                    if (chartEl) initChartIfNeeded(chartEl.id);
                }}
            }});
        }});

        document.getElementById('expand-all').addEventListener('click', function() {{
            document.querySelectorAll('details.test-result').forEach(d => {{
                d.open = true;
                const chartEl = d.querySelector('.chart-container');
                if (chartEl) initChartIfNeeded(chartEl.id);
            }});
        }});

        document.getElementById('collapse-all').addEventListener('click', function() {{
            document.querySelectorAll('details.test-result').forEach(d => d.open = false);
        }});

        const sortedView = document.getElementById('sorted-view');
        const allResults = Array.from(document.querySelectorAll('details.test-result'));
        const origParent = new Map();
        const origNext = new Map();
        allResults.forEach(el => {{
            origParent.set(el, el.parentNode);
            origNext.set(el, el.nextSibling);
        }});

        function applySort() {{
            const val = document.getElementById('sort-by').value;
            const useAbs = document.getElementById('sort-abs').checked;
            const asc = document.getElementById('sort-asc').checked;

            if (val === 'default') {{
                sortedView.style.display = 'none';
                document.querySelectorAll('.category-better, .category-worse, .category-neutral')
                    .forEach(el => el.style.display = '');
                allResults.forEach(el => {{
                    const parent = origParent.get(el);
                    const next = origNext.get(el);
                    if (next && next.parentNode === parent) {{
                        parent.insertBefore(el, next);
                    }} else {{
                        parent.appendChild(el);
                    }}
                }});
            }} else {{
                const keyMap = {{cles: 'cles', deltaPct: 'deltaPct', deltaAbs: 'deltaAbs', cliffsDelta: 'cliffsDelta'}};
                const key = keyMap[val];
                function normalizedVal(el) {{
                    const v = parseFloat(el.dataset[key]);
                    const lib = el.dataset.lowerIsBetter === '1';
                    if (key === 'deltaPct' || key === 'deltaAbs') return lib ? -v : v;
                    if (key === 'cliffsDelta') return lib ? v : -v;
                    if (key === 'cles') return lib ? v : 1 - v;
                    return v;
                }}
                const sorted = [...allResults].sort((a, b) => {{
                    let av = normalizedVal(a);
                    let bv = normalizedVal(b);
                    if (useAbs) {{ av = Math.abs(av); bv = Math.abs(bv); }}
                    return asc ? av - bv : bv - av;
                }});
                sortedView.replaceChildren(...sorted);
                sortedView.style.display = '';
                document.querySelectorAll('.category-better, .category-worse, .category-neutral')
                    .forEach(el => el.style.display = 'none');
            }}
        }}

        ['sort-by', 'sort-abs', 'sort-asc'].forEach(id =>
            document.getElementById(id).addEventListener('change', applySort)
        );
    }});
    </script>

    <div class="timestamp">
        Report generated on {generated_at}
    </div>
</body>
</html>"""
