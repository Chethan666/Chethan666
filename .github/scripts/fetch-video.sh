#!/usr/bin/env bash
# Downloads a produced story's video folder (the artifact its production run saved) to out/story-<id>.
# Needs GH_TOKEN, REPO and CONFIG.
set -euo pipefail
story=$1
run=$(python - "$story" <<'PY'
import json, sys
from story_factory import db
from story_factory.config import load_config
import os
row = db.get_story(db.connect(load_config(os.environ["CONFIG"])["database"]), int(sys.argv[1]))
production = json.loads(row["production"] or "{}") if row else {}
print(production.get("run") or "")
PY
)
if [ -z "$run" ]; then
  echo "Story #$story has no production run recorded" >&2
  exit 1
fi
artifact=$(gh api "repos/$REPO/actions/runs/$run/artifacts" --jq \
  '[.artifacts[] | select(.expired == false and (.name | startswith("video-")))] | first | .id // empty')
if [ -z "$artifact" ]; then
  echo "Run $run's video artifact has expired or is missing (artifacts are kept 14 days): make the video again" >&2
  exit 1
fi
mkdir -p out
gh api "repos/$REPO/actions/artifacts/$artifact/zip" > "$RUNNER_TEMP/video.zip"
unzip -o -q "$RUNNER_TEMP/video.zip" -d out
ls -la "out/story-$story"
