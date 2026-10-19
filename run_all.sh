#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 build_dataset.py
python3 findings.py
python3 eval/planted.py
python3 build_app.py
python3 export_bns.py
python3 test_engine.py
python3 eval/logic_eval.py
echo "Готово: $(pwd)/out/provable.html"
