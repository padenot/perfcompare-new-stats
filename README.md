# new stats for perf.compare

```
uv venv
source .venv/bin/activate
uv pip -r install requirements.txt
python3 perf_compare_cli_new.py https://perf.compare/compare-results\?newRev\=aa76d5841f94617a02ea612c32ca8e29a756cd05\&baseRepo\=try\&baseRev\=7bedadac3ab75ae4d8593814e80ac10b3ebb9e72\&newRepo\=try\&framework\=13  --output-file filename.html --title "Report title"
```
