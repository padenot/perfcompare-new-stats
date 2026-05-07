/**
 * Interactive KDE chart widget for browser embedding.
 *
 * Depends on: echarts (CDN global), kde.js (mode fitting re-implemented inline)
 * Consumed by: example.mjs (inlined into generated HTML)
 *
 * Self-contained IIFE — attaches to .kde-chart[data-kde] elements, initialises
 * an echarts chart with valley-depth slider, bootstrap CI blurb, and mode labels.
 * Mode-fitting logic (fitModes, argrelmax, areaFracs) is inlined rather than
 * imported so the widget can be embedded as a single <script> block.
 */
(function() {
  echarts.registerTheme('instant', { animation: false, animationDuration: 0, animationDurationUpdate: 0 });
  var COLORS = ['#3498db', '#e67e22', '#27ae60', '#9b59b6', '#e74c3c'];
  var initialized = new WeakSet();

  // ── pure helpers ────────────────────────────────────────────────────────────

  function fmtVal(v) {
    var a = Math.abs(v);
    return a < 10 ? v.toFixed(2) : a < 100 ? v.toFixed(1) : v.toFixed(0);
  }

  function argrelmax(y, order) {
    var out = [];
    for (var i = order; i < y.length - order; i++) {
      var ok = true;
      for (var j = 1; j <= order && ok; j++)
        if (y[i] <= y[i - j] || y[i] <= y[i + j]) ok = false;
      if (ok) out.push(i);
    }
    return out;
  }

  // integrate KDE area per mode bucket (trapezoid rule)
  function areaFracs(x, y, boundaries) {
    var buckets = new Array(boundaries.length + 1).fill(0), total = 0;
    for (var i = 1; i < x.length; i++) {
      var area = 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
      total += area;
      var m = 0;
      while (m < boundaries.length && x[i] > boundaries[m]) m++;
      buckets[m] += area;
    }
    return total > 0 ? buckets.map(function(b) { return b / total; })
                     : buckets.map(function() { return 1 / buckets.length; });
  }

  function fitModes(x, y, vt, mpf, mdf) {
    mpf = mpf || 0.05; mdf = mdf || 0.05;
    var yMax = 0;
    for (var i = 0; i < y.length; i++) if (y[i] > yMax) yMax = y[i];
    var peaks = argrelmax(y, 3).filter(function(i) { return y[i] >= mpf * yMax; });

    if (!peaks.length) {
      var gm = 0;
      for (var i = 1; i < y.length; i++) if (y[i] > y[gm]) gm = i;
      return { peakLocs: [x[gm]], boundaries: [] };
    }

    var good = [peaks[0]];
    for (var k = 1; k < peaks.length; k++) {
      var nxt = peaks[k], prev = good[good.length - 1], vm = y[prev];
      for (var j = prev; j <= nxt; j++) if (y[j] < vm) vm = y[j];
      if (vm < vt * Math.min(y[prev], y[nxt])) good.push(nxt);
      else if (y[nxt] > y[good[good.length - 1]]) good[good.length - 1] = nxt;
    }

    function bounds(ps) {
      var bs = [];
      for (var i = 0; i < ps.length - 1; i++) {
        var mi = ps[i];
        for (var j = ps[i]; j <= ps[i + 1]; j++) if (y[j] < y[mi]) mi = j;
        bs.push(x[mi]);
      }
      return bs;
    }

    var bs0 = bounds(good), fr0 = areaFracs(x, y, bs0);
    var keep = good.map(function(_, i) { return i; })
                   .filter(function(i) { return fr0[i] >= mdf; });
    if (keep.length < 2) {
      var bp = good.reduce(function(a, b) { return y[a] > y[b] ? a : b; });
      return { peakLocs: [x[bp]], boundaries: [] };
    }
    var fg = keep.map(function(i) { return good[i]; }), fb = bounds(fg);
    var locs = fg.map(function(i) { return x[i]; });

    // Require modes to be meaningfully separated: at least 2 samples apart.
    // Peaks closer than this are KDE artefacts on near-integer data (e.g. {0,1}).
    var dataRange = x[x.length - 1] - x[0];
    var minSep = Math.max(2, dataRange * 0.05);
    for (var k = 1; k < locs.length; k++) {
      if (locs[k] - locs[k - 1] < minSep) {
        // Modes too close — collapse to single mode at the higher peak
        var bestIdx = fg.reduce(function(a, b) { return y[a] > y[b] ? a : b; });
        return { peakLocs: [x[bestIdx]], boundaries: [] };
      }
    }
    return { peakLocs: locs, boundaries: fb };
  }

  // A = lowest value (fastest/cheapest for profiler sample counts)
  function assignLetters(locs) {
    var idx = locs.map(function(_, i) { return i; })
                  .sort(function(a, b) { return locs[a] - locs[b]; });
    var out = new Array(locs.length);
    idx.forEach(function(i, rank) { out[i] = String.fromCharCode(65 + rank); });
    return out;
  }

  // bitmask DP min-cost matching (n,m ≤ 8 modes — always the case here)
  //
  // Why match modes at all?
  // -----------------------
  // When comparing base vs. new, mode A in the base may correspond to mode B
  // in the new run (same code path, just re-ordered by peak location or density
  // shift).  Naively comparing by index would pair the wrong paths.
  //
  // This function solves a minimum-cost assignment problem: pair each base mode
  // to a new mode so that the total "distance" (75% location, 25% density
  // fraction) is minimised.  It uses dynamic programming over subsets (bitmask
  // DP), which is exact and fast for the small mode counts seen in practice.
  // Unmatched modes (a path that appeared or disappeared) are returned separately
  // as ub (unmatched base) and un (unmatched new).
  function matchModes(bLocs, bFracs, nLocs, nFracs) {
    var n = bLocs.length, m = nLocs.length;
    if (!n || !m) return { pairs: [], ub: range(n), un: range(m) };
    var all = bLocs.concat(nLocs);
    var span = Math.max.apply(null, all) - Math.min.apply(null, all) || 1;
    var cost = bLocs.map(function(bl, i) {
      return nLocs.map(function(nl, j) {
        return 0.75 * Math.abs(bl - nl) / span + 0.25 * Math.abs(bFracs[i] - nFracs[j]);
      });
    });
    // match all min(n,m) base to new
    if (n > m) {
      var sw = matchModes(nLocs, nFracs, bLocs, bFracs);
      return { pairs: sw.pairs.map(function(p) { return [p[1], p[0]]; }), ub: sw.un, un: sw.ub };
    }
    // n <= m: DP over subsets of new modes of size n
    var INF = 1e9, states = 1 << m;
    var dp = new Float64Array(states).fill(INF), prev = new Int16Array(states).fill(-1);
    dp[0] = 0;
    for (var mask = 0; mask < states; mask++) {
      if (dp[mask] === INF) continue;
      var i = popcount(mask); if (i >= n) continue;
      for (var j = 0; j < m; j++) {
        if (mask >> j & 1) continue;
        var nm = mask | (1 << j), c = dp[mask] + cost[i][j];
        if (c < dp[nm]) { dp[nm] = c; prev[nm] = j; }
      }
    }
    var best = -1, bc = INF;
    for (var mask = 0; mask < states; mask++)
      if (popcount(mask) === n && dp[mask] < bc) { bc = dp[mask]; best = mask; }
    var pairs = [], cur = best;
    for (var i = n - 1; i >= 0; i--) {
      var j = prev[cur]; pairs.unshift([i, j]); cur ^= (1 << j);
    }
    var mNew = new Set(pairs.map(function(p) { return p[1]; }));
    return { pairs: pairs, ub: [], un: range(m).filter(function(j) { return !mNew.has(j); }) };
  }

  function popcount(x) { var c = 0; while (x) { c += x & 1; x >>= 1; } return c; }
  function range(n) { return Array.from({ length: n }, function(_, i) { return i; }); }

  // ── CLES (Vargha-Delaney A) ──────────────────────────────────────────────────
  // P(comp < base): fraction of random (base, comp) pairs where comp is faster.
  // 0.5 = no difference, >0.5 = comp faster, <0.5 = comp slower.
  function cles(base, comp) {
    var n = base.length, m = comp.length;
    if (!n || !m) return NaN;
    var u = 0;
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < m; j++) {
        if (comp[j] < base[i]) u += 1;
        else if (comp[j] === base[i]) u += 0.5;
      }
    }
    return u / (n * m);
  }


  // Two-tailed permutation test for the rank-sum statistic.
  // Works for any N — exact for small samples, equivalent to Mann-Whitney for large.
  // O((n+m) log(n+m)) setup + O(n+m) per iteration.
  function permutationP(base, comp, nIter) {
    nIter = nIter || 999;
    var n = base.length, m = comp.length, total = n + m;
    if (!n || !m) return NaN;
    // Assign ranks to the pooled array (average ranks for ties)
    var pool = Array.prototype.slice.call(base).concat(Array.prototype.slice.call(comp));
    var order = pool.map(function(_, i) { return i; }).sort(function(a, b) { return pool[a] - pool[b]; });
    var poolRanks = new Array(total);
    for (var i = 0; i < total; ) {
      var j = i;
      while (j < total && pool[order[j]] === pool[order[i]]) j++;
      var avgRank = (i + j + 1) / 2;
      for (var k = i; k < j; k++) poolRanks[order[k]] = avgRank;
      i = j;
    }
    // Observed rank sum for comp (last m entries in pool)
    var obsSum = 0;
    for (var i = n; i < total; i++) obsSum += poolRanks[i];
    var expected = m * (total + 1) / 2;
    var obsDeviation = Math.abs(obsSum - expected);
    // Permutation: repeatedly draw m indices without replacement, sum their ranks
    var indices = pool.map(function(_, i) { return i; });
    var extreme = 0;
    for (var iter = 0; iter < nIter; iter++) {
      // Partial Fisher-Yates: shuffle only the first m slots
      for (var i = 0; i < m; i++) {
        var j = i + Math.floor(Math.random() * (total - i));
        var tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
      }
      var rankSum = 0;
      for (var i = 0; i < m; i++) rankSum += poolRanks[indices[i]];
      if (Math.abs(rankSum - expected) >= obsDeviation) extreme++;

    }
    return (extreme + 1) / (nIter + 1);
  }

  // ── bootstrap CI ─────────────────────────────────────────────────────────────
  // Simplified variant of bootstrapMedianDiffCI from bootstrap-ci.js.
  // Uses Math.random() (non-reproducible) because the widget is interactive
  // and re-runs on slider changes; exact reproducibility is not required here.
  function arrMedian(arr) {
    if (!arr.length) return NaN;
    var s = arr.slice().sort(function(a, b) { return a - b; });
    var m = s.length >> 1;
    return s.length & 1 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  function bootSample(arr) {
    var out = new Array(arr.length);
    for (var i = 0; i < arr.length; i++) out[i] = arr[Math.floor(Math.random() * arr.length)];
    return out;
  }

  function bootstrapMedianCI(base, comp, nIter) {
    nIter = nIter || 500;
    if (base.length < 2 || comp.length < 2) return null;
    var shifts = new Array(nIter);
    for (var i = 0; i < nIter; i++)
      shifts[i] = arrMedian(bootSample(comp)) - arrMedian(bootSample(base));
    shifts.sort(function(a, b) { return a - b; });
    return {
      shift: arrMedian(comp) - arrMedian(base),
      lo: shifts[Math.floor(0.025 * nIter)],
      hi: shifts[Math.ceil(0.975 * nIter) - 1]
    };
  }

  function splitByMode(data, boundaries) {
    var buckets = boundaries.map(function() { return []; });
    buckets.push([]);
    data.forEach(function(v) {
      var m = 0;
      while (m < boundaries.length && v > boundaries[m]) m++;
      buckets[m].push(v);
    });
    return buckets;
  }

  // ── English blurb ────────────────────────────────────────────────────────────
  // Generates an HTML summary of the comparison for display below the chart.
  // Single-mode: one line showing direction, magnitude, and CI.
  // Multi-mode: a verdict header followed by one row per matched/unmatched mode,
  // each with its bootstrap CI, fraction of runs, and path label (fast/mid/slow).
  function generateBlurb(kd, modesArr) {
    var empty = { header: '', detail: '' };
    if (modesArr.length < 2) return empty;
    var base = modesArr[0], comp = modesArr[1];
    if (!base.peakLocs.length || !comp.peakLocs.length) return empty;

    var match = matchModes(base.peakLocs, base.fracs, comp.peakLocs, comp.fracs);

    var baseSplits = kd.rawSamples && kd.rawSamples[0]
      ? splitByMode(kd.rawSamples[0], base.boundaries) : null;
    var compSplits = kd.rawSamples && kd.rawSamples[1]
      ? splitByMode(kd.rawSamples[1], comp.boundaries) : null;

    // path label: A = fastest, last letter = slowest
    function pathLabel(letter, totalModes) {
      var rank = letter.charCodeAt(0) - 65;
      if (totalModes === 1) return '';
      if (rank === 0) return 'fast\u00a0path';
      if (rank === totalModes - 1) return 'slow\u00a0path';
      return 'mid\u00a0path';
    }

    function ciLine(ci95, sig, baseLoc) {
      if (!ci95) return '';
      var pct = baseLoc > 0 ? (ci95.shift / baseLoc * 100) : 0;
      var col = sig ? (ci95.shift < 0 ? '#060' : '#b00') : '#555';
      var arrow = sig ? (ci95.shift < 0 ? '▼\u00a0faster' : '▲\u00a0slower') : 'no\u00a0reliable\u00a0change';
      return '<span style="color:' + col + ';font-weight:' + (sig ? 'bold' : 'normal') + '">' + arrow + '</span>' +
        '\u2002' + (ci95.shift >= 0 ? '+' : '') + fmtVal(ci95.shift) + '\u00a0' + (kd.unit || 'samples/iter') +
        (sig && baseLoc > 0 ? '\u00a0(' + (pct >= 0 ? '+' : '') + pct.toFixed(1) + '%)' : '') +
        '\u2002<span style="color:#555">95%\u00a0CI\u00a0[' +
        (ci95.lo >= 0 ? '+' : '') + fmtVal(ci95.lo) + ',\u2009' +
        (ci95.hi >= 0 ? '+' : '') + fmtVal(ci95.hi) + ']</span>';
    }

    // compute all pair CIs up front for the overall verdict
    var pairData = match.pairs.map(function(p) {
      var bi = p[0], ci = p[1];
      var ci95 = (baseSplits && compSplits)
        ? bootstrapMedianCI(baseSplits[bi], compSplits[ci]) : null;
      // Significant if CI is entirely on one side, with at least one strictly non-zero bound.
      // This rejects [0,0] (degenerate bootstrap on constant integer data) while keeping [-2,0].
      var sig = ci95
        ? ((ci95.hi <= 0 && (ci95.lo < 0 || ci95.hi < 0)) ||
           (ci95.lo >= 0 && (ci95.lo > 0 || ci95.hi > 0)))
        : false;
      return { bi: bi, ci: ci, ci95: ci95, sig: sig };
    });

    var multimodal = base.peakLocs.length > 1 || comp.peakLocs.length > 1;
    var sigCount = pairData.filter(function(r) { return r.sig; }).length;
    var improvements = pairData.filter(function(r) { return r.sig && r.ci95 && r.ci95.shift < 0; });
    var regressions  = pairData.filter(function(r) { return r.sig && r.ci95 && r.ci95.shift > 0; });
    var rawBase = kd.rawSamples && kd.rawSamples[0];
    var rawComp = kd.rawSamples && kd.rawSamples[1];
    var clesPct = (rawBase && rawComp) ? cles(rawBase, rawComp) : NaN;
    var unit = kd.unit || 'samples/iter';
    var headerLines = [];
    var detailLines = [];

    // ── prominent header (always shown) ──────────────────────────────────────
    if (!isNaN(clesPct)) {
      var clesPctRnd = Math.round(clesPct * 100);
      var clesCol = clesPct > 0.5 ? '#060' : clesPct < 0.5 ? '#b00' : '#555';
      var mwp = (rawBase && rawComp) ? permutationP(rawBase, rawComp) : NaN;
      var mwSig = !isNaN(mwp) && mwp < 0.05;
      headerLines.push(
        '<span style="font-size:1.4em;font-weight:bold;color:' + clesCol + '">' + clesPctRnd + '%</span>' +
        ' <span style="color:#555">CLES</span>' +
        (function() {
        var dir = clesPct >= 0.5 ? 'faster' : 'slower';
        var pct2 = clesPct >= 0.5 ? clesPctRnd : (100 - clesPctRnd);
        var adv = mwSig ? 'significantly ' : '';
        return ' <span style="color:#888;font-size:0.9em">new is ' + adv + dir + ' than base in ' + pct2 + '% of head-to-head run comparisons</span>';
      })() +
        ' <span style="color:#888">(p=' + (isNaN(mwp) ? 'n/a' : mwp < 0.001 ? '<0.001' : mwp.toFixed(3)) + ')</span>' +
        (!mwSig ? ' <span style="color:#888">— within noise</span>' : '')
      );
    }

    if (!multimodal) {
      // ── single-mode: Δ median prominently ──────────────────────────────────
      var r = pairData[0];
      if (r && r.ci95) {
        var shift = r.ci95.shift;
        var bl = base.peakLocs[r.bi];
        var col = r.sig ? (shift < 0 ? '#060' : '#b00') : '#555';
        var pct = bl > 0 ? (shift / bl * 100) : 0;
        headerLines.push(
          'Δ median: <span style="font-size:1.15em;font-weight:bold;color:' + col + '">' +
          (shift >= 0 ? '+' : '') + fmtVal(shift) + ' ' + unit + '</span>' +
          (r.sig && bl > 0 ? ' <span style="color:' + col + '">(' + (pct >= 0 ? '+' : '') + pct.toFixed(1) + '%)</span>' : '') +
          ' <span style="color:#888">95% CI [' +
          (r.ci95.lo >= 0 ? '+' : '') + fmtVal(r.ci95.lo) + ', ' +
          (r.ci95.hi >= 0 ? '+' : '') + fmtVal(r.ci95.hi) + ']</span>' +
          (!r.sig ? ' <span style="color:#888">— within noise</span>' : '')
        );
      }
    } else {
      // ── multi-mode: verdict header + per-mode rows ─────────────────────────
      var verdict;
      if (match.ub.length === 0 && match.un.length === 0 && sigCount === 0) {
        verdict = '<span style="color:#555">No reliable change in any mode</span>';
      } else {
        var newSlowPaths  = match.un.filter(function(ci) { return ci === comp.peakLocs.length - 1; }).length;
        var lostFastPaths = match.ub.filter(function(bi) { return bi === 0 && base.peakLocs.length > 1; }).length;
        var elimSlowPaths = match.ub.filter(function(bi) { return bi === base.peakLocs.length - 1; }).length;
        if (regressions.length === 0 && newSlowPaths === 0 && lostFastPaths === 0 &&
            (improvements.length > 0 || elimSlowPaths > 0)) {
          verdict = '<span style="color:#060;font-weight:bold">▼ Overall faster</span>';
        } else if (improvements.length === 0 && elimSlowPaths === 0 &&
                   (regressions.length > 0 || newSlowPaths > 0 || lostFastPaths > 0)) {
          verdict = '<span style="color:#b00;font-weight:bold">▲ Overall slower</span>';
        } else {
          verdict = '<span style="color:#a60;font-weight:bold">⚠ Mixed results</span>';
        }
      }
      var bm = base.peakLocs.length, cm = comp.peakLocs.length;
      var modeStr = (bm === 1 ? '1 mode' : bm + ' modes') + ' base · ' + (cm === 1 ? '1 mode' : cm + ' modes') + ' new';
      detailLines.push(verdict + ' <span style="color:#888">' + modeStr + '</span>');

      pairData.forEach(function(r) {
        var letter = base.letters[r.bi];
        var bl = base.peakLocs[r.bi];
        var bf = base.fracs[r.bi], cf = comp.fracs[r.ci];
        var df = cf - bf;
        var pl = pathLabel(letter, base.peakLocs.length);
        var fracStr = Math.abs(df) >= 0.03
          ? Math.round(bf * 100) + '% → ' + Math.round(cf * 100) + '%'
          : Math.round(bf * 100) + '% of runs';
        detailLines.push(
          '<b style="font-size:1.05em">Mode ' + letter + '</b>' +
          (pl ? ' <span style="color:#666">' + pl + '</span>' : '') +
          ' ~' + fmtVal(bl) + ' samples ' + fracStr + '<br>' +
          ' ' + (ciLine(r.ci95, r.sig, bl) || '<span style="color:#888">no CI available</span>')
        );
      });

      match.ub.forEach(function(bi) {
        var letter = base.letters[bi], bl = base.peakLocs[bi], frac = base.fracs[bi];
        var pl = pathLabel(letter, base.peakLocs.length);
        // Find the nearest comp peak to determine where these runs likely went
        var nearestComp = comp.peakLocs.reduce(function(a, b) {
          return Math.abs(b - bl) < Math.abs(a - bl) ? b : a;
        });
        var isImprovement = bl > nearestComp;
        var status = isImprovement
          ? '<span style="color:#060;font-weight:bold">✓ gone — these runs are now faster (merged into a quicker path)</span>'
          : '<span style="color:#b00;font-weight:bold">⚠ gone — these runs are now slower (merged into a slower path)</span>';
        detailLines.push(
          '<b>Mode ' + letter + '</b> ' + pl +
          ' ~' + fmtVal(bl) + ' samples ' + Math.round(frac * 100) + '% of base runs<br>' +
          ' ' + status
        );
      });

      match.un.forEach(function(ci) {
        var letter = comp.letters[ci], cl = comp.peakLocs[ci], frac = comp.fracs[ci];
        var pl = pathLabel(letter, comp.peakLocs.length);
        // Find the nearest base peak to determine where these runs came from
        var nearestBase = base.peakLocs.reduce(function(a, b) {
          return Math.abs(b - cl) < Math.abs(a - cl) ? b : a;
        });
        var isImprovement = cl < nearestBase;
        var status = isImprovement
          ? '<span style="color:#060;font-weight:bold">✓ new path — these runs are now faster than before</span>'
          : '<span style="color:#b00;font-weight:bold">⚠ new path — these runs are now slower than before</span>';
        detailLines.push(
          '<b>Mode ' + letter + '</b> ' + pl +
          ' ~' + fmtVal(cl) + ' samples ' + Math.round(frac * 100) + '% of new runs<br>' +
          ' ' + status
        );
      });
    }

    return {
      header: headerLines.filter(Boolean).join('<br>'),
      detail: detailLines.filter(Boolean).join('<br>')
    };
  }

  // ── chart update ─────────────────────────────────────────────────────────────

  function buildSeries(kd, vt, chartWidth) {
    var modesArr = kd.series.map(function(s) {
      var m = fitModes(s.x, s.y, vt);
      var fracs = areaFracs(s.x, s.y, m.boundaries);
      var letters = assignLetters(m.peakLocs);
      return { peakLocs: m.peakLocs, boundaries: m.boundaries, fracs: fracs, letters: letters };
    });

    // Assign vertical stagger levels: only offset labels that would overlap horizontally.
    // Labels within 13% of the x-range of each other get different levels; others get 0.
    var allPeaks = [];
    modesArr.forEach(function(m, si) {
      m.peakLocs.forEach(function(loc) { allPeaks.push({ loc: loc, si: si, level: 0 }); });
    });
    allPeaks.sort(function(a, b) { return a.loc - b.loc; });
    var xSpan = (kd.xMax !== undefined && kd.xMin !== undefined)
      ? kd.xMax - kd.xMin
      : (allPeaks.length > 1 ? allPeaks[allPeaks.length - 1].loc - allPeaks[0].loc : 1);
    var labelPx = 20 * 6;
    var threshold = xSpan * (chartWidth > 0 ? labelPx / chartWidth : 0.13);
    allPeaks.forEach(function(p, idx) {
      var used = {};
      for (var k = 0; k < idx; k++) {
        if (Math.abs(allPeaks[k].loc - p.loc) < threshold) used[allPeaks[k].level] = true;
      }
      var level = 0;
      while (used[level]) level++;
      p.level = level;
    });
    var globalIdx = {};
    allPeaks.forEach(function(p) {
      if (!globalIdx[p.si]) globalIdx[p.si] = {};
      globalIdx[p.si][p.loc] = p.level;
    });

    var series = [];
    kd.series.forEach(function(s, i) {
      var color = COLORS[i % COLORS.length];
      var modes = modesArr[i];
      series.push({
        name: s.name, type: 'line', smooth: false, symbol: 'none', z: 2,
        lineStyle: { width: 2.5, color: color },
        areaStyle: { opacity: 0.08, color: color },
        data: s.x.map(function(x, j) { return [x, s.y[j]]; })
      });

      if (modes.peakLocs.length > 0) {
        modes.peakLocs.forEach(function(loc, pi) {
          var gi = (globalIdx[i] || {})[loc] || 0;
          series.push({
            name: '_' + i + '_' + pi, type: 'line', data: [],
            markLine: {
              silent: true, symbol: 'none',
              data: [{ xAxis: loc }],
              lineStyle: { color: color, type: 'solid', width: 1.5 },
              label: {
                formatter: s.name.split(' ')[0] + '\u00a0' + modes.letters[pi] +
                  ':\u00a0' + fmtVal(loc) + '\u00a0(' + Math.round(modes.fracs[pi] * 100) + '%)',
                distance: [0, gi * 13],
                color: color, fontSize: 10
              }
            }
          });
        });
      } else {
        var med = s.x[Math.round(s.x.length / 2)]; // fallback centre
        series.push({
          name: '_' + i + '_med', type: 'line', data: [],
          markLine: {
            silent: true, symbol: 'none',
            data: [{ xAxis: med }],
            lineStyle: { color: color, type: 'dashed', width: 1.5 },
            label: { formatter: s.name.split(' ')[0], color: color, fontSize: 10 }
          }
        });
      }
    });

    var maxGi = allPeaks.reduce(function(m, p) { return Math.max(m, p.level); }, 0);
    return { series: series, modesArr: modesArr, gridTop: 24 + maxGi * 14 };
  }

  function recompute(kd, chart, headerEl, detailEl, vt) {
    var result = buildSeries(kd, vt, chart.getWidth());
    chart.setOption({ animation: false, animationDuration: 0, animationDurationUpdate: 0,
      grid: { top: result.gridTop },
      series: result.series }, { replaceMerge: ['series'] });
    var blurb = generateBlurb(kd, result.modesArr);
    if (blurb.header) {
      headerEl.innerHTML = blurb.header;
      headerEl.classList.add('visible');
    } else {
      headerEl.classList.remove('visible');
    }
    if (blurb.detail) {
      detailEl.innerHTML = blurb.detail;
      detailEl.classList.add('visible');
    } else {
      detailEl.classList.remove('visible');
    }
  }

  // ── init ─────────────────────────────────────────────────────────────────────

  function initKdeChart(el) {
    if (initialized.has(el)) return;
    var raw = el.getAttribute('data-kde');
    if (!raw) return;
    var kd; try { kd = JSON.parse(raw); } catch(e) { return; }
    initialized.add(el);

    // headerEl/detailEl go outside any <details> wrapper; controls stay inside
    var wrapper = el.closest('details') || el.parentNode;
    var outerParent = wrapper.parentNode;

    var headerEl = document.createElement('div');
    headerEl.className = 'kde-blurb';
    outerParent.insertBefore(headerEl, wrapper);

    var controls = document.createElement('div');
    controls.className = 'kde-controls';
    var vt = 0.5;
    controls.innerHTML =
      'Valley\u00a0depth\u00a0threshold ' +
      '<abbr title="A valley between two peaks must be shallower than this fraction of the shorter peak to count as a mode boundary. Higher = more splits detected.">[?]</abbr>: ' +
      '<input type="range" min="10" max="99" value="50" step="1">' +
      '<span>50%</span>';
    var slider = controls.querySelector('input');
    var valLabel = controls.querySelector('span');
    el.parentNode.insertBefore(controls, el);

    var detailEl = document.createElement('div');
    detailEl.className = 'kde-blurb';
    outerParent.insertBefore(detailEl, wrapper.nextSibling);

    var initialGridTop = buildSeries(kd, vt, el.clientWidth).gridTop;
    var chart = echarts.init(el, 'instant');
    chart.setOption({
      animation: false,
      animationDuration: 0,
      animationDurationUpdate: 0,
      tooltip: {
        trigger: 'axis', axisPointer: { type: 'cross' },
        formatter: function(params) {
          var pts = params.filter(function(p) { return p.seriesName[0] !== '_'; });
          if (!pts.length) return '';
          return 'x\u202f=\u202f' + fmtVal(pts[0].value[0]) + '<br>' +
            pts.map(function(p) {
              return p.marker + p.seriesName + ': ' + p.value[1].toExponential(2);
            }).join('<br>');
        }
      },
      legend: { data: kd.series.map(function(s) { return s.name; }), bottom: 40, textStyle: { fontSize: 11 } },
      toolbox: { feature: { restore: {}, saveAsImage: {} }, right: 8, top: 4, itemSize: 12 },
      dataZoom: [
        { type: 'slider', xAxisIndex: [0], bottom: 8, height: 20, start: 0, end: 100 },
        { type: 'inside', xAxisIndex: [0] }
      ],
      grid: { top: initialGridTop, bottom: 72, left: 8, right: 8, containLabel: true },
      xAxis: (function() {
        var margin = kd.xMax !== undefined
          ? Math.max(0.5, (kd.xMax - kd.xMin) * 0.15) : 0;
        return {
          type: 'value',
          min: kd.xMin !== undefined ? Math.max(0, kd.xMin - margin) : 0,
          max: kd.xMax !== undefined ? kd.xMax + margin : undefined,
          name: kd.unit || 'samples/iter', nameLocation: 'middle', nameGap: 24,
          nameTextStyle: { fontSize: 10 }, axisLabel: { fontSize: 10 }
        };
      })(),
      yAxis: { type: 'value', show: false }
    });

    recompute(kd, chart, headerEl, detailEl, vt);

    slider.addEventListener('input', function() {
      vt = parseInt(this.value) / 100;
      valLabel.textContent = this.value + '%';
      recompute(kd, chart, headerEl, detailEl, vt);
    });

    new ResizeObserver(function() { chart.resize(); }).observe(el);
  }

  function initAllVisible() {
    document.querySelectorAll('.kde-chart[data-kde]').forEach(function(el) {
      var d = el.closest('details');
      if (!d || d.open) initKdeChart(el);
    });
    document.querySelectorAll('details').forEach(function(d) {
      d.addEventListener('toggle', function() {
        if (d.open) d.querySelectorAll('.kde-chart[data-kde]').forEach(initKdeChart);
      });
    });
  }

  if (typeof echarts !== 'undefined') initAllVisible();
  else document.querySelector('script[src*="echarts"]').addEventListener('load', initAllVisible);
})();
