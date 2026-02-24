# new stats for perf.compare

## Setup

```
uv sync
```

## Usage

```
uv run perfcompare <perf.compare URL> [options]
```

Options:

| Flag | Description |
|------|-------------|
| `--output-file FILE` | Output HTML file (default: auto-generated name) |
| `--title TITLE` | Report title |
| `--search TERM` | Filter results by test name |
| `--limit N` | Process only first N results |
| `--workers N` | Parallel workers (default: auto) |
| `--no-replicates` | Use aggregated runs instead of replicates |
| `--no-bootstrap` | Skip bootstrap CIs (faster) |
| `--save-data FILE` | Save fetched JSON to file |
| `--from-file FILE` | Load from local JSON instead of fetching |
| `--no-cache` | Disable request cache |
| `--clear-cache` | Clear cache before fetching |

## Example

```
uv run perfcompare "https://perf.compare/compare-results?newRev=aa76d5841f94&baseRepo=try&baseRev=7bedadac3ab7&newRepo=try&framework=13" \
  --output-file report.html --title "My comparison"
```
