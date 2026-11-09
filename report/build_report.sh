#!/bin/sh
# report/*.md -> report.pdf (pandoc + headless Chrome). Run from anywhere.
set -e
cd "$(dirname "$0")"
pandoc -f markdown+lists_without_preceding_blankline title.md q1_eda.md q2_summary.md q3_architecture.md q4_client.md \
  -s --metadata title="AHU Fault Detection and Diagnosis" -c style.css -o report.html
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu \
  --no-pdf-header-footer --print-to-pdf=../report.pdf "file://$PWD/report.html" 2>/dev/null
echo "../report.pdf"
