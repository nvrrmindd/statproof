#!/bin/bash
# Двойной щелчок — запускает Provable и открывает его в браузере.
cd "$(dirname "$0")"
python3 serve.py --open
