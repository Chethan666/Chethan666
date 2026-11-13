#!/usr/bin/env bash
# Pushes small review images of a run (contact sheet, thumbnails, frames from the
# video every 20 s, four full-size frames, Shorts) and its reports to the review-media branch, under run-<number>/,
# so they can be looked at without downloading the whole artifact. No audio, no full video.
set -euo pipefail
src=out
dest=$RUNNER_TEMP/review/run-$GITHUB_RUN_NUMBER
mkdir -p "$dest"
[ -d "$src" ] || { echo "Nothing to share"; exit 0; }
shrink() {  # picture -> JPEG at most 1280 px wide
  ffmpeg -nostdin -hide_banner -loglevel error -y -i "$1" -vf "scale='min(1280,iw)':-2" -q:v 4 "$2" || true
}
find "$src" -type f \( -name '*.png' -o -name '*.jpg' \) ! -path '*/work/*' | while read -r f; do
  name=$(echo "${f#$src/}" | tr '/' '_'); shrink "$f" "$dest/${name%.*}.jpg"
done
find "$src" -type f -name 'video.mp4' | while read -r v; do
  dir=$(dirname "${v#$src/}" | tr '/' '_')
  ffmpeg -nostdin -hide_banner -loglevel error -y -i "$v" -vf "fps=1/20,scale=640:-2,tile=4x3" -q:v 5 "$dest/${dir}_filmstrip_%02d.jpg" || true
  # Four frames at full size, so sharpness, grain and colour can be judged as the viewer sees them.
  length=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$v" | cut -d. -f1)
  for part in 15 40 65 90; do
    ffmpeg -nostdin -hide_banner -loglevel error -y -ss "$(( ${length:-0} * part / 100 ))" -i "$v" -frames:v 1 -q:v 2 \
      "$dest/${dir}_frame_${part}.jpg" || true
  done
done
find "$src" -type f -name 'short_*.mp4' | while read -r v; do
  name=$(echo "${v#$src/}" | tr '/' '_')
  ffmpeg -nostdin -hide_banner -loglevel error -y -ss 5 -i "$v" -frames:v 1 -vf scale=360:-2 "$dest/${name%.*}.jpg" || true
done
find "$src" -type f \( -name 'report.json' -o -name 'beats.json' -o -name 'metadata.json' \) \
  ! -path '*/work/*' | while read -r f; do cp "$f" "$dest/$(echo "${f#$src/}" | tr '/' '_')"; done
cd "$RUNNER_TEMP/review"
git init -q
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
remote="https://x-access-token:${GH_TOKEN}@github.com/${REPO}.git"
if git fetch -q --depth 1 "$remote" review-media 2>/dev/null; then
  git checkout -q -b review-media FETCH_HEAD
else
  git checkout -q --orphan review-media
fi
git add "run-$GITHUB_RUN_NUMBER"
if git diff --cached --quiet; then echo "Nothing to share"; exit 0; fi
git commit -q -m "Review media for run $GITHUB_RUN_NUMBER"
git push -q "$remote" HEAD:review-media
echo "Shared $(ls "run-$GITHUB_RUN_NUMBER" | wc -l) files on the review-media branch"
