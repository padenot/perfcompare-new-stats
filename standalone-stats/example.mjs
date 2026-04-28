#!/usr/bin/env node
/**
 * Standalone example: FFTKDE (ISJ) + bootstrap CI + interactive chart widget.
 *
 * Dependencies (all local, no npm install needed):
 *   kde.js          — FFTKDE with ISJ bandwidth (port of KDEpy)
 *   bootstrap-ci.js — percentile bootstrap CI for median difference
 *   kde-widget.js   — chart widget (mode detection, slider, blurb) extracted
 *                     from the comparison-report-generator template
 *
 * Outputs: example-output.html — open directly in browser, no server needed.
 */
import { writeFileSync, readFileSync } from "fs";
import { resolve }                     from "path";
import { fftkde }                      from "./kde.js";
import { bootstrapMedianDiffCI }       from "./bootstrap-ci.js";

// ── Seeded Gaussian samples ───────────────────────────────────────────────────

function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 0x100000000;
  };
}

function gaussian(rng, n, mean, std) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const u = 1 - rng(), v = rng();
    out.push(mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v));
  }
  return out;
}

// ── Data: bimodal base, unimodal new (slow path eliminated, fast path faster) -

const rng     = mulberry32(7);
const base    = [...gaussian(rng, 65, 120, 10), ...gaussian(rng, 35, 280, 20)];
const newData =   gaussian(rng, 100, 95, 9);

// ── KDE ───────────────────────────────────────────────────────────────────────

function downsample(arr, n = 512) {
  if (arr.length <= n) return arr;
  const step = arr.length / n;
  return Array.from({ length: n }, (_, i) => arr[Math.round(i * step)]);
}

const bKde = fftkde(base,    "ISJ", undefined, 1024);
const nKde = fftkde(newData, "ISJ", undefined, 1024);

const allX = [...base, ...newData];
const kdePayload = JSON.stringify({
  series: [
    { name: "base", x: downsample(bKde.x), y: downsample(bKde.y) },
    { name: "new",  x: downsample(nKde.x), y: downsample(nKde.y) },
  ],
  rawSamples: [base, newData],
  xMin: Math.min(...allX),
  xMax: Math.max(...allX),
});

// ── Bootstrap CI (overall, for the summary line) ──────────────────────────────

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

const ci  = bootstrapMedianDiffCI(base, newData);
const fmt = (n) => (n >= 0 ? "+" : "") + n.toFixed(1);
const summary =
  `Δ median = ${fmt(ci.medianDiff)} ms  95% CI [${fmt(ci.ciLow)}, ${fmt(ci.ciHigh)}]` +
  (ci.significant ? "" : "  (not significant)");

// ── Embed the chart widget JS ─────────────────────────────────────────────────

const widgetJs = readFileSync(new URL("./kde-widget.js", import.meta.url), "utf-8");

// ── HTML ──────────────────────────────────────────────────────────────────────

const dataAttr = kdePayload.replace(/"/g, "&quot;");

const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Perf stats example</title>
<style>
body { font: 12px/1.5 monospace; margin: 24px; color: #111; max-width: 860px }
h2 { font-size: 1.1em; border-bottom: 1px solid #bbb; padding-bottom: 3px }
.summary { font-weight: bold; color: ${ci.significant ? (ci.medianDiff < 0 ? "#060" : "#b00") : "#888"}; margin: 8px 0 }
details { margin: 6px 0 }
details summary { cursor: pointer; color: #444; padding: 2px 0 }
details[open] > summary { font-weight: bold }
.kde-chart { width: 100%; height: 320px }
.kde-controls { font-size: 11px; color: #555; margin: 4px 0 2px;
                display: flex; align-items: center; gap: 8px }
.kde-controls input[type=range] { width: 140px; cursor: pointer }
.kde-blurb { font-size: 11px; line-height: 1.6; margin: 4px 0 6px; padding: 6px 8px;
             background: #f9f9f9; border-left: 3px solid #bbb; display: none }
.kde-blurb.visible { display: block }
</style>
</head>
<body>
<h2>base (bimodal, n=${base.length}) vs new (unimodal, n=${newData.length})</h2>
<p class="summary">${summary}</p>
<p style="color:#666">base median: ${median(base).toFixed(1)} ms &nbsp;|&nbsp; new median: ${median(newData).toFixed(1)} ms</p>
<details open>
  <summary>Sample distribution (KDE)</summary>
  <div class="kde-chart" data-kde="${dataAttr}"></div>
</details>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<script>
${widgetJs}
</script>
</body>
</html>`;

const out = resolve(import.meta.dirname, "example-output.html");
writeFileSync(out, html);
console.log(`file://${out}`);
console.log(summary);
