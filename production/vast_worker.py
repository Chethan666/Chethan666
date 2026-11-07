"""Runs on a rented Vast.ai GPU: open models draw, animate and score a shot list.

    python vast_worker.py job.json pictures   # Qwen-Image-Edit-2509: style frame, character sheets, shots
    python vast_worker.py job.json clips      # Wan 2.2 I2V A14B: a clip from every shot picture
    python vast_worker.py job.json audio      # MMAudio: foley from every clip, room tone, score beds, stinger

Self-contained (no imports from this repository): the runner copies this file and job.json
to the machine. Every stage writes its files into job["out"] and prints one line per item
so the runner can follow progress; an item that fails is skipped, never fatal.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
import traceback
import zlib
from pathlib import Path

NEGATIVE_PICTURE = ("photo, photograph, realistic, 3d render, painting, colourful, colour illustration, green, "
                    "blue, yellow, grey gradient shading, text, letters, words, watermark, signature, frame, "
                    "border, panels, collage, detailed realistic face")
NEGATIVE_CLIP = ("colour, colorful, realistic, photo, 3d render, shading, gradient, text, watermark, blurry, "
                 "distorted, deformed, morphing, melting, extra limbs, extra figures, flicker, static frame")


def log(*parts: object) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def chalk_check(image) -> tuple[bool, float]:
    """Mostly black, some white chalk, no colour but red. Returns (ok, colour share) to pick the best."""
    import numpy as np

    hsv = np.asarray(image.convert("RGB").resize((320, 180)).convert("HSV"), dtype=np.float32) / 255.0
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    dark = float(np.mean(v < 0.3))
    chalk = float(np.mean((v > 0.6) & (s < 0.25)))
    other = float(np.mean((s > 0.35) & (v > 0.3) & ~((h < 0.06) | (h > 0.93))))
    return dark >= 0.4 and chalk >= 0.003 and other <= 0.02, other + (0.5 if dark < 0.4 else 0.0)


def free() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def pictures(job: dict, out: Path) -> None:
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from PIL import Image

    W, H = job.get("picture_size", [1664, 928])
    steps = int(job.get("picture_steps", 40))
    pipe = QwenImageEditPlusPipeline.from_pretrained(job["picture_model"], torch_dtype=torch.bfloat16).to("cuda")
    log("picture model loaded")

    def draw(images: list, prompt: str, seed: int):
        return pipe(image=images, prompt=prompt, negative_prompt=NEGATIVE_PICTURE, true_cfg_scale=4.0,
                    num_inference_steps=steps, height=H, width=W,
                    generator=torch.Generator("cuda").manual_seed(seed)).images[0]

    def best(images: list, prompt: str, name: str, tries: int) -> Image.Image | None:
        kept, score = None, 9.0
        for k in range(tries):
            try:
                picture = draw(images, prompt, 1000 + 97 * k + zlib.crc32(name.encode()) % 997)
            except Exception:  # noqa: BLE001 - one bad item never stops the stage
                log("FAILED", name, traceback.format_exc(limit=2).replace("\n", " | "))
                continue
            ok, bad = chalk_check(picture)
            log("picture", name, "try", k + 1, "ok" if ok else f"rejected ({bad:.3f})")
            if ok:
                return picture
            if bad < score:
                kept, score = picture, bad
        return kept

    board = Image.new("RGB", (W, H), (14, 14, 14))
    style = best([board], job["style_frame_prompt"], "style_frame", 3) or board
    style.save(out / "style_frame.png")
    sheets: dict[str, Image.Image] = {}
    for c in job.get("sheets", []):
        sheet = best([style], c["prompt"], f"sheet {c['name']}", 2)
        if sheet is not None:
            sheets[c["name"]] = sheet
            sheet.save(out / f"sheet_{c['file']}.png")
    for shot in job["shots"]:
        refs = [style] + [sheets[n] for n in shot.get("characters", []) if n in sheets][:2]
        prompt = shot["picture_prompt"]
        for k, n in enumerate([n for n in shot.get("characters", []) if n in sheets][:2], 2):
            prompt = prompt.replace(f"{{sheet:{n}}}", f"image {k}")
        picture = best(refs, prompt, f"shot_{shot['index']:03d}", int(job.get("picture_tries", 3)))
        if picture is not None:
            picture.save(out / f"shot_{shot['index']:03d}.png")
    pipe.to("cpu")
    pipe = None  # noqa: F841 - frees the GPU for the clip model
    free()


def clips(job: dict, out: Path, shard: int = 0, shards: int = 1) -> None:
    import torch
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    from PIL import Image

    W, H = job.get("clip_size", [1280, 720])
    pipe = WanImageToVideoPipeline.from_pretrained(job["clip_model"], torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    log("clip model loaded")
    for shot in job["shots"][shard::shards]:
        src = out / f"shot_{shot['index']:03d}.png"
        dst = out / f"clip_{shot['index']:03d}.mp4"
        if not src.exists() or dst.exists():
            continue
        started = time.time()
        try:
            image = Image.open(src).convert("RGB")
            scale = max(W / image.width, H / image.height)
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
            x, y = (image.width - W) // 2, (image.height - H) // 2
            image = image.crop((x, y, x + W, y + H))
            frames = pipe(image=image, prompt=shot["motion_prompt"], negative_prompt=NEGATIVE_CLIP, height=H,
                          width=W, num_frames=int(shot.get("frames", 81)), guidance_scale=3.5,
                          num_inference_steps=int(job.get("clip_steps", 30)),
                          generator=torch.Generator("cuda").manual_seed(7 + shot["index"])).frames[0]
            export_to_video(frames, str(dst), fps=16)
            log("clip", shot["index"], f"done in {time.time() - started:.0f} s")
        except Exception:  # noqa: BLE001
            log("FAILED clip", shot["index"], traceback.format_exc(limit=2).replace("\n", " | "))
            free()
    del pipe
    free()


def audio(job: dict, out: Path) -> None:
    """MMAudio's own command line: foley from a clip (with its prompt), or sound from text alone."""
    root = Path(job.get("mmaudio_dir", "/workspace/MMAudio"))
    for item in job.get("audio", []):
        target = out / f"{item['name']}.flac"
        if target.exists():
            continue
        folder = out / "mm" / item["name"]
        folder.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, "demo.py", "--prompt", item["prompt"], "--duration", str(item["duration"]),
                "--output", str(folder), "--negative_prompt", item.get("negative", "music, speech, voices, talking")]
        if item.get("video"):
            video = out / item["video"]
            if not video.exists():
                continue
            args += ["--video", str(video)]
        result = subprocess.run(args, cwd=root, capture_output=True, text=True)
        made = sorted(folder.rglob("*.flac"), key=lambda p: p.stat().st_mtime)
        if result.returncode != 0 or not made:
            log("FAILED audio", item["name"], result.stderr[-300:].replace("\n", " | "))
            continue
        made[-1].replace(target)
        log("audio", item["name"], "done")


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    stage = sys.argv[2]
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    log("stage", stage, "starting")
    if stage == "pictures":
        pictures(job, out)
    elif stage == "clips":
        shard, shards = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
        clips(job, out, shard, shards)
    elif stage == "audio":
        audio(job, out)
    else:
        raise SystemExit(f"unknown stage {stage}")
    log("stage", stage, "finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
