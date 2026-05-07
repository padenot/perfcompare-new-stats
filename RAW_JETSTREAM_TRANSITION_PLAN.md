# Raw JetStream Data Transition Plan

## Goal

Preserve the full JetStream measurement hierarchy end-to-end so perfcompare can:

- keep current `-Average` style comparisons working during the transition
- expose unsmoothed raw distributions to humans now
- later perform statistically defensible grouped analysis

This specifically means preserving temporal structure inside a task, because effects like warmup, JIT tiering, and GC can make early and late iterations behave differently.

The immediate target is backward-compatible transport of structured raw data from JetStream to Treeherder to perfcompare.

## Current State

### JetStream

The top patch in [`../jetstream/JetStreamDriver.js`] emits:

- regular perfherder-style subtests and summary values
- a new `rawResults` object with:
  - `iterationCount`
  - `worstCaseCount`
  - `times`
  - `subScores`
  - `subTimes`

This is a good first step, but it still does not preserve the grouping boundary needed for rigorous downstream analysis.

In particular, raw timings should remain grouped as arrays, not flattened, because:

- warmup effects change behavior across iterations
- GC pauses create bursts and tails that matter
- averaging destroys ordering and regime changes
- later analysis may want per-benchmark-run summaries rather than one pooled sample

### Treeherder

Treeherder currently stores:

- one `PerformanceDatum` per `(job, signature, push, timestamp)`
- zero or more flat `PerformanceDatumReplicate(value)` rows per datum

Important consequences:

- `job_id` is already available on each datum
- `task_id` is already recoverable through `Job -> TaskclusterMetadata`
- `machine` is already recoverable through `Job -> Machine`
- replicate grouping is not preserved today; Treeherder flattens replicates when building perfcompare responses

### Perfcompare

Perfcompare currently exposes:

- `base_runs` / `new_runs`: one value per task/retrigger
- `base_runs_replicates` / `new_runs_replicates`: one flat list of all replicate values

This is enough for visualization of raw distributions, but not enough for grouped inference.

## Target Data Model

The logical hierarchy for JetStream raw timing data should be:

- task
- benchmark_run
- iteration

Conceptually, one side of a comparison should be representable as:

```text
list[list[list[float]]]
```

Using symbols:

- `N`: task / retrigger / external run
- `M`: benchmark-internal run within one task
- `K`: iteration/sample within one benchmark-internal run

So the fully faithful structure is:

```text
N tasks -> M benchmark runs -> K iterations
```

or `[[[float]]]`.

## Terminology

Avoid the word `run` by itself. It is overloaded across JetStream, Treeherder, and perfcompare.

Use:

- `task`: one Taskcluster task / retrigger / Treeherder datum
- `benchmark_run`: one internal JetStream run/segment inside a task
- `iteration`: one innermost float sample

If existing code or UI already says `replicate`, keep it only for legacy flat arrays.

## Concrete Wire Format Recommendation

Use the following naming consistently across producer, storage, and API layers.

### Per subtest in the perfherder artifact

Recommended new field:

```json
{
  "name": "Foo-RawTime",
  "value": 123.4,
  "replicates": [123.1, 123.5],
  "benchmarkRuns": [
    [1.0, 1.1, 1.2],
    [0.9, 1.0, 1.1]
  ]
}
```

Why `benchmarkRuns`:

- it is more precise than `rawRuns`
- it avoids collision with Treeherder's existing notion of task-level runs
- it makes the middle level explicit

### Per datum in Treeherder storage

Recommended stored JSON shape:

```json
{
  "benchmark_runs": [
    [1.0, 1.1, 1.2],
    [0.9, 1.0, 1.1]
  ],
  "iteration_count": 120,
  "worst_case_count": 4
}
```

### In the perfcompare API

Recommended new fields:

- `base_task_runs_raw`
- `new_task_runs_raw`

Each should be:

```json
[
  {
    "job_id": 123,
    "task_id": "abc...",
    "machine": "t-linux-...",
    "benchmark_runs": [
      [1.0, 1.1, 1.2],
      [0.9, 1.0, 1.1]
    ]
  }
]
```

This gives a stable mapping:

- artifact producer: `benchmarkRuns`
- Treeherder storage: `benchmark_runs`
- perfcompare API: per-task objects with `benchmark_runs`

### Legacy compatibility

Keep these old fields during transition:

- summary `value`
- flat `replicates`
- current perfcompare `base_runs_replicates` / `new_runs_replicates`

But define them as compatibility views, not the canonical raw representation.

## Compatibility Principle

Do not replace existing fields during the transition.

Add new structured fields in parallel with the old flat fields:

- keep current summary subtests such as `-Average`
- keep current flat `replicates` storage and API fields if needed for old clients
- add structured raw payloads beside them

This allows:

- old consumers to keep working
- new tooling to opt into structured raw data
- incremental rollout across repos

For the API layer, the preferred transition is a new endpoint rather than mutating the existing compare endpoint aggressively.

## Endpoint Strategy

Recommended approach:

- keep the current perfcompare endpoint as the compatibility endpoint
- add a new endpoint for structured raw data

Suggested split:

- legacy endpoint:
  - `/api/perfcompare/results/`
- new structured endpoint:
  - `/api/perfcompare/results-raw/`

Alternative names if needed:

- `/api/perfcompare/results-structured/`
- `/api/perfcompare/results-v3/`

Current recommendation: `results-raw`

Why a new endpoint:

- avoids breaking current clients
- avoids changing payload expectations on a hot path
- allows iteration on shape and metadata during the transition
- makes it clear which endpoint returns canonical grouped raw data
- lets old and new UIs coexist cleanly

The new endpoint should return:

- the same row identity/signature metadata needed to match existing compare rows
- optional summary stats if convenient
- task-level grouped raw data with richer tagging

The old endpoint should remain stable and unchanged except for bug fixes.

## Proposed Rollout

### Phase 1: Producer-side Structure

Producer goal: emit grouped raw data from JetStream.

Preferred shape for each subtest in the artifact:

```json
{
  "name": "Foo-RawTime",
  "value": 123.4,
  "replicates": [123.1, 123.5],
  "benchmarkRuns": [
    [1.0, 1.1, 1.2],
    [0.9, 1.0, 1.1]
  ]
}
```

Where:

- `value` stays whatever summary value existing dashboards expect
- `replicates` stays flat for compatibility if required
- `benchmarkRuns` is the new grouped representation of benchmark runs within one task

Important:

- `benchmarkRuns` must remain an array of arrays
- do not replace it with one flat raw list
- if order is meaningful, preserve order within each inner array
- if JetStream can distinguish warmup vs steady-state slices explicitly, preserve that too

Open question for Firefox-side integration:

- the user mentioned a Firefox `jj` workspace at `../firefox`, but that path was not present from this repo
- once the correct path is provided, identify where the perfherder artifact is assembled and thread `benchmarkRuns` through there without disturbing existing summary output

### Phase 2: Treeherder Ingestion

Treeherder should ingest structured raw data without breaking current consumers.

Preferred approach:

1. Keep `PerformanceDatumReplicate` for legacy flat replicate access.
2. Add structured raw storage per datum.

Recommended storage shape:

- either a JSON field on `PerformanceDatum`
- or a side table keyed by `PerformanceDatum`

Suggested stored JSON payload:

```json
{
  "benchmark_runs": [
    [1.0, 1.1, 1.2],
    [0.9, 1.0, 1.1]
  ],
  "iteration_count": 120,
  "worst_case_count": 4
}
```

Rationale:

- one `PerformanceDatum` already maps to one `job`
- task and machine metadata are already recoverable from the datum's job
- no need to duplicate `task_id` or `machine` inside the stored raw blob

The important property is that the grouped arrays survive ingestion unchanged. Flattened legacy replicates can still be stored separately for old code paths, but they are not the source of truth.

Implementation recommendation:

- use `models.JSONField` first
- prefer JSON storage over normalized tables for the first rollout

Why JSON first:

- the producer already emits nested structured data
- Treeherder already uses `models.JSONField`
- the first goal is transport and visualization, not SQL analytics over benchmark runs
- it keeps the migration smaller and easier to roll back if needed

Suggested model choices, in order:

1. add nullable `raw_data = models.JSONField(null=True, blank=True)` to `PerformanceDatum`
2. if keeping `PerformanceDatum` smaller is preferred, add a one-to-one side model such as `PerformanceDatumRaw`

For the initial rollout, either is acceptable. JSON should be treated as canonical storage for grouped raw data, while `PerformanceDatumReplicate` remains a compatibility path.

Schema work likely needed in Treeherder:

- extend [`../treeherder/schemas/performance-artifact.json`] to allow the new structured field
- extend validation coverage in [`../treeherder/treeherder/log_parser/utils.py`]
- extend schema tests in [`../treeherder/tests/etl/test_perf_schema.py`]
- extend ingestion in [`../treeherder/treeherder/etl/perf.py`] to store it
- add tests in Treeherder ETL and API suites

### Phase 3: Treeherder API

Perfcompare API should expose both old and new views.

Keep existing fields on the legacy endpoint:

- `base_runs`
- `new_runs`
- `base_runs_replicates`
- `new_runs_replicates`

Expose structured data on the new endpoint:

- `base_task_runs_raw`
- `new_task_runs_raw`

Recommended API shape:

```json
[
  {
    "job_id": 123,
    "task_id": "abc...",
    "machine": "t-linux-...",
    "benchmark_runs": [
      [1.0, 1.1, 1.2],
      [0.9, 1.0, 1.1]
    ]
  }
]
```

This gives:

- grouping by task
- grouping by machine
- enough structure to reconstruct `[[[float]]]`

API policy:

- do not remove or redefine existing flat fields on the legacy endpoint during the transition
- treat the new endpoint as the canonical structured raw endpoint
- old clients keep reading flat arrays unchanged
- new clients prefer the new endpoint
- only add structured fields to the old endpoint if there is a very strong operational reason

That avoids a hard migration on existing consumers while still giving the new design a clean surface.

If payload size becomes a problem, add an opt-in query parameter later, for example:

- `structured_raw=true`

But the first recommendation is still: separate endpoint first, query param second.

Likely patch points:

- [`../treeherder/treeherder/webapp/api/performance_data.py`]
- [`../treeherder/treeherder/webapp/api/performance_serializers.py`]
- [`../treeherder/tests/webapp/api/test_perfcompare_api.py`]

### Phase 4: perfcompare Consumer

Perfcompare should:

1. keep using current fields when the new structured data is absent
2. visualize grouped raw data when present
3. clearly distinguish:
   - task-level samples
   - benchmark-run groups
   - flattened legacy replicate lists

Near-term UI/analysis goals:

- show grouped raw distributions per task
- show within-task per-benchmark-run arrays
- make warmup patterns and GC-heavy tails visible to the developer
- allow squint-test inspection by humans
- avoid pretending flattened `N*M*K` samples are fully independent

Later analysis goals:

- task-level tests
- benchmark-run summaries within task
- mixed/hierarchical models if needed

## Statistical Guidance

Until grouping is preserved, flattened raw values are useful for visualization but dangerous for inference.

Rules of thumb:

- use flattened raw data for distribution inspection
- prefer grouped raw arrays for any serious inspection of warmup or GC behavior
- do not claim extra certainty from `N*M*K` flattened samples
- prefer task-level inference
- if needed, summarize benchmark runs within each task before cross-revision comparison

## Concrete Work Items

### JetStream

- Keep current `rawResults`.
- Add grouped raw timings at benchmark/subtest level.
- Make sure naming is explicit enough that consumers can tell:
  - summary values
  - flat compatibility values
  - grouped raw benchmark runs

Recommended refinement of the current `rawResults` object:

- keep `times` only as a compatibility view if something already consumes it
- add `benchmarkRuns` as the canonical grouped field
- if `subTimes` is already grouped, either rename it or document it precisely so consumers do not infer the wrong semantics

### Firefox

- Find the producer path in the `jj` workspace that converts JetStream output into perfherder artifact JSON.
- Thread grouped raw timings through unchanged.
- Preserve old fields during rollout.

Status:

- blocked on correct workspace path from user

### Treeherder

- Extend performance artifact schema for structured raw benchmark-run data.
- Persist structured raw data per datum.
- Keep flat replicate ingestion working.
- Expose grouped raw task metadata in a new perfcompare endpoint.
- Add tests for:
  - schema validation
  - ETL ingestion
  - API serialization
  - backward compatibility

Concrete code areas already identified:

- schema validation:
  - [`../treeherder/schemas/performance-artifact.json`]
  - [`../treeherder/treeherder/log_parser/utils.py`]
  - [`../treeherder/tests/etl/test_perf_schema.py`]
- ingestion:
  - [`../treeherder/treeherder/etl/perf.py`]
- compare API:
  - [`../treeherder/treeherder/webapp/api/performance_data.py`]
  - [`../treeherder/treeherder/webapp/api/performance_serializers.py`]
  - [`../treeherder/tests/webapp/api/test_perfcompare_api.py`]

### perfcompare-new-stats

- Accept the new structured fields when present.
- Fall back to existing flat fields when absent.
- Add reporting that distinguishes:
  - legacy flat raw list
  - grouped-by-task raw data
  - grouped-by-benchmark-run raw data

## Recommended Order

1. Finalize the producer-side shape in Firefox/JetStream.
2. Add Treeherder schema support and raw storage.
3. Add a new structured perfcompare endpoint in Treeherder.
4. Update this repo to consume the new endpoint.
5. After the transition period, decide whether old flat replicate fields can be deprecated.

## Open Questions

- What is the correct path to the Firefox `jj` workspace?
- What exact JetStream structure should be emitted for grouped raw data:
  - current recommendation: `benchmarkRuns`
  - only change this if Firefox-side constraints force a different name
- Should Treeherder persist structured raw data in JSON or in normalized tables?
  - current recommendation: JSON first
  - revisit normalized tables only if server-side querying over benchmark runs becomes a real need
- Do we want machine grouping only as metadata, or do we also want a dedicated API aggregation by machine?

## Recommendation

For the transition period, prefer the smallest design that preserves truth:

- emit grouped raw benchmark-run data from JetStream/Firefox
- store it as JSON structured data per `PerformanceDatum`
- expose it through a new structured perfcompare endpoint
- keep all current fields alive until consumers have moved

That gets the full `task -> benchmark_run -> iteration` hierarchy into the system without forcing a flag day.
