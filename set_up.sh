#!/bin/bash

set -ex

pip install -r requirements.txt
pip uninstall -y flashinfer-jit-cache || true
pip install nvidia-cudnn-cu12==9.16.0.29
pip install numpy==1.26.4

if command -v playwright >/dev/null 2>&1; then
  playwright install chromium
else
  python -m playwright install chromium
fi

# Create a symbolic link for libstdc++.so.6 to ensure compatibility with certain libraries
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /root/miniconda3/lib/libstdc++.so.6
