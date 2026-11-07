#!/bin/bash
# macOS: double-click this file in Finder.
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
cd -- "$ROOT" || exit 1
if [[ -x ".venv/bin/python" ]]; then
    .venv/bin/python src/stem_local/launcher.py --root "$ROOT" "$@"
    RESULT=$?
else
    echo 'Install Python 3.11, uv and FFmpeg (with ffprobe and libmp3lame).'
    echo 'Then run in this folder: uv sync --frozen'
    echo 'See docs/launch.md for macOS requirements and installation instructions.'
    RESULT=1
fi
if [[ "$RESULT" -ne 0 && -t 0 ]]; then
    read -r -p 'Press Enter to close...' _
fi
exit "$RESULT"
