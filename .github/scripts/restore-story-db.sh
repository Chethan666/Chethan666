#!/usr/bin/env bash
# Restores data/ from the story database the most recently finished run saved.
#
# Looks run by run (newest finished first) instead of using the repository-wide
# artifact list, which can lag behind a run that finished seconds ago and would
# hand back an older database. The newest run gets a few retries while its
# upload settles. Needs GH_TOKEN, REPO and GITHUB_RUN_ID.
set -euo pipefail

runs=$(gh api "repos/$REPO/actions/workflows/story-factory.yml/runs?status=completed&per_page=30" --jq \
  "[.workflow_runs[] | select(.id != $GITHUB_RUN_ID and .conclusion != \"cancelled\")]
   | sort_by(.updated_at) | reverse | .[].id")

first=1
for run in $runs; do
  tries=$([ "$first" = 1 ] && echo 4 || echo 1)
  first=0
  for _ in $(seq "$tries"); do
    id=$(gh api "repos/$REPO/actions/runs/$run/artifacts" --jq \
      '[.artifacts[] | select(.expired == false and (.name == "story-db" or (.name | startswith("stories-"))))][0].id // empty')
    if [ -n "$id" ]; then
      mkdir -p data
      gh api "repos/$REPO/actions/artifacts/$id/zip" > "$RUNNER_TEMP/story-db.zip"
      unzip -o "$RUNNER_TEMP/story-db.zip" -d data
      echo "Restored the story database saved by run $run (artifact $id)"
      exit 0
    fi
    [ "$tries" -gt 1 ] && sleep 10
  done
done
echo "No saved story database yet; starting a new one"
