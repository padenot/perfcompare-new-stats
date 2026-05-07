#!/usr/bin/env node
/**
 * Standalone example: FFTKDE (ISJ) + bootstrap CI + interactive chart widget.
 *
 * Dependencies (all local, no npm install needed):
 *   kde.js        — FFTKDE with ISJ bandwidth (port of KDEpy)
 *   kde-widget.js — chart widget (mode detection, slider, blurb)
 *
 * Outputs: example-output.html — open directly in browser, no server needed.
 */
import { writeFileSync, readFileSync } from "fs";
import { resolve }                     from "path";
import { fftkde }                      from "./kde.js";

// ── Seeded PRNG + sample generators ──────────────────────────────────────────

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

function mix(rng, ...components) {
  return components.flatMap(([n, mean, std]) => gaussian(rng, n, mean, std));
}

// ── Scenarios ─────────────────────────────────────────────────────────────────

const rng = mulberry32(42);

const scenarios = [
  {
    title: "Both unimodal — clear improvement",
    base: mix(rng, [120, 130, 12]),
    new:  mix(rng, [120,  95,  10]),
  },
  {
    title: "Both unimodal — clear regression",
    base: mix(rng, [120, 100, 10]),
    new:  mix(rng, [120, 140, 12]),
  },
  {
    title: "Both unimodal — no significant change",
    base: mix(rng, [120, 110, 15]),
    new:  mix(rng, [120, 112, 15]),
  },
  {
    title: "Base bimodal, new unimodal — slow path eliminated, fast path faster",
    base: mix(rng, [80, 120, 10], [40, 290, 20]),
    new:  mix(rng, [120, 95, 9]),
  },
  {
    title: "Base unimodal, new bimodal — slow path appeared",
    base: mix(rng, [120, 100, 10]),
    new:  mix(rng, [80, 95, 9], [40, 260, 18]),
  },
  {
    title: "Both bimodal — both modes faster (overall improvement)",
    base: mix(rng, [70, 130, 12], [50, 280, 18]),
    new:  mix(rng, [70, 100, 10], [50, 240, 15]),
  },
  {
    title: "Both bimodal — mixed: fast path regressed, slow path improved",
    base: mix(rng, [70, 100, 10], [50, 280, 18]),
    new:  mix(rng, [70, 130, 10], [50, 240, 15]),
  },
  {
    title: "Both trimodal — all three modes faster",
    base: mix(rng, [50, 100, 8], [40, 200, 12], [30, 350, 20]),
    new:  mix(rng, [50,  80, 8], [40, 170, 12], [30, 300, 18]),
  },
  {
    title: "Base trimodal, new unimodal — fast path only, all slow paths gone",
    base: mix(rng, [50, 110, 10], [40, 220, 15], [30, 380, 22]),
    new:  mix(rng, [120, 95, 9]),
  },
  {
    title: "Both unimodal — high variance, not significant",
    base: mix(rng, [80, 150, 60]),
    new:  mix(rng, [80, 140, 65]),
  },
  {
    title: "Small N (n=15) — real improvement but few samples",
    base: mix(rng, [15, 120, 12]),
    new:  mix(rng, [15,  88, 10]),
  },
];

// ── KDE + payload builder ─────────────────────────────────────────────────────

function downsample(arr, n = 512) {
  if (arr.length <= n) return arr;
  const step = arr.length / n;
  return Array.from({ length: n }, (_, i) => arr[Math.round(i * step)]);
}

function makePayload(base, newData) {
  const bKde = fftkde(base,    "ISJ", undefined, 1024);
  const nKde = fftkde(newData, "ISJ", undefined, 1024);
  const allX = [...base, ...newData];
  return JSON.stringify({
    series: [
      { name: "base", x: downsample(bKde.x), y: downsample(bKde.y) },
      { name: "new",  x: downsample(nKde.x), y: downsample(nKde.y) },
    ],
    rawSamples: [base, newData],
    unit: "ms",
    xMin: Math.min(...allX),
    xMax: Math.max(...allX),
  });
}

// ── Embed widget JS once ──────────────────────────────────────────────────────

const widgetJs = readFileSync(new URL("./kde-widget.js", import.meta.url), "utf-8");

// ── HTML ──────────────────────────────────────────────────────────────────────

const sections = scenarios.map(({ title, base, new: newData }, i) => {
  const payload = makePayload(base, newData);
  const dataAttr = payload.replace(/"/g, "&quot;");
  return `<section>
<h2>${i + 1}. ${title}</h2>
<details open>
  <summary>Sample distribution (KDE)</summary>
  <div class="kde-chart" data-kde="${dataAttr}"></div>
</details>
</section>`;
}).join("\n");

const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Perf stats examples</title>
<style>
body { font: 12px/1.5 monospace; margin: 24px; color: #111; max-width: 900px }
h1 { font-size: 1.2em; margin-bottom: 16px }
h2 { font-size: 1.3em; border-bottom: 1px solid #bbb; padding-bottom: 4px; margin: 28px 0 6px }
section { margin-bottom: 32px }
details { margin: 6px 0 }
details summary { cursor: pointer; color: #444; padding: 2px 0 }
details[open] > summary { font-weight: bold }
.kde-chart { width: 100%; height: 320px }
.kde-controls { font-size: 11px; color: #555; margin: 4px 0 2px;
                display: flex; align-items: center; gap: 8px }
.kde-controls input[type=range] { width: 140px; cursor: pointer }
.kde-blurb { font-size: 11px; line-height: 1.7; margin: 4px 0 6px; padding: 8px 10px;
             background: #f9f9f9; border-left: 3px solid #bbb; display: none }
.kde-blurb.visible { display: block }
</style>
</head>
<body>
<h1>Perf comparison widget — scenario gallery</h1>
${sections}
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<script>
${widgetJs}
</script>
</body>
</html>`;

const out = resolve(import.meta.dirname, "example-output.html");
writeFileSync(out, html);
console.log(`file://${out}`);
