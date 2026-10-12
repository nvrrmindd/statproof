#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 build_dataset.py
python3 findings.py
python3 build_app.py
python3 test_engine.py
echo "Готово: $(pwd)/out/provable.html"
