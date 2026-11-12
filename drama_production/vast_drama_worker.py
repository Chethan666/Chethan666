"""Runs on a rented Vast.ai GPU: open models voice, draw and film one drama episode.

    python vast_worker.py job.json voices    # Chatterbox (MIT): every line, cloned from its character's voice
    python vast_worker.py job.json pictures  # Qwen-Image-Edit-2511 (Apache-2.0): references, then every shot
    python vast_worker.py job.json talk      # Wan2.2-S2V-14B (Apache-2.0): lip-synced clips of on-screen speakers
    python vast_worker.py job.json move      # Wan 2.2 I2V A14B (Apache-2.0): every other shot, and talk fallbacks
    python vast_worker.py job.json sound     # MMAudio: the room's ambience

Self-contained (no imports from this repository): the runner copies this file and job.json
to the machine. Every stage writes into job["out"] and logs one line per item; an item that
fails is skipped, never fatal, so the runner can always finish the episode.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

NEGATIVE_PICTURE = ("cartoon, anime, illustration, painting, drawing, 3d render, cgi, plastic skin, waxy skin, "
                    "deformed face, deformed hands, extra fingers, extra limbs, duplicate person, text, letters, "
                    "captions, subtitles, watermark, logo, frame, border, collage, blurry, lowres, oversaturated")
NEGATIVE_CLIP = ("cartoon, anime, 3d render, text, captions, watermark, blurry, distorted face, deformed, morphing, "
                 "melting, extra limbs, extra people, flicker, jitter, static frame, sudden cut")
EXAGGERATION = {"whisper": (0.35, 0.5), "calm": (0.4, 0.5), "warm": (0.45, 0.5), "cold": (0.4, 0.45),
                "sad": (0.55, 0.45), "tense": (0.6, 0.4), "pleading": (0.7, 0.35), "afraid": (0.75, 0.35),
                "shocked": (0.75, 0.35), "angry": (0.85, 0.3), "mocking": (0.6, 0.4)}


def log(*parts: object) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def free() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def pad(audio, rate: int, lead: float = 0.15, tail: float = 0.45, least: float = 2.0):
    """The same breathing room the edit gives every line, so a talking clip matches its shot."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    piece = np.concatenate([np.zeros(int(lead * rate), np.float32), audio, np.zeros(int(tail * rate), np.float32)])
    if piece.size < int(least * rate):
        piece = np.concatenate([piece, np.zeros(int(least * rate) - piece.size, np.float32)])
    return piece


# --- voices -----------------------------------------------------------------------------------
def voices(job: dict, out: Path) -> None:
    """Chatterbox runs in its own environment (it pins its own torch); see the remote script."""
    import numpy as np
    import soundfile
    import torch
    from chatterbox.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(device="cuda")
    log("voice model loaded")
    for line in job["lines"]:
        dst = out / f"line_{line['index']:03d}.wav"
        if dst.exists():
            continue
        ref = out / line["reference"]
        ex, cfg = EXAGGERATION.get(line.get("emotion", ""), (0.5, 0.5))
        best = None
        for attempt in range(2):  # a second take if the first comes out empty or far too long
            try:
                torch.manual_seed(11 + line["index"] + 100 * attempt)
                wav = model.generate(line["text"], audio_prompt_path=str(ref), exaggeration=ex, cfg_weight=cfg)
                audio = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
            except Exception:  # noqa: BLE001
                log("FAILED voice", line["index"], traceback.format_exc(limit=2).replace("\n", " | "))
                continue
            seconds = audio.size / model.sr
            words = max(len(line["text"].split()), 1)
            if 0.25 * words <= seconds <= 1.2 * words + 1.5:
                best = audio
                break
            log("voice", line["index"], f"take {attempt + 1} rejected ({seconds:.1f} s for {words} words)")
            best = best if best is not None else audio
        if best is None:
            continue
        peak = float(np.max(np.abs(best))) or 1.0
        soundfile.write(dst, pad(best * min(1.0, 0.9 / peak), model.sr), model.sr)
        log("voice", line["index"], "done")


# --- pictures ---------------------------------------------------------------------------------
def pictures(job: dict, out: Path) -> None:
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from PIL import Image

    W, H = job.get("picture_size", [928, 1664])
    steps = int(job.get("picture_steps", 40))
    pipe = QwenImageEditPlusPipeline.from_pretrained(job["picture_model"], torch_dtype=torch.bfloat16).to("cuda")
    log("picture model loaded")

    def draw(images: list, prompt: str, seed: int, name: str):
        try:
            return pipe(image=images, prompt=prompt, negative_prompt=NEGATIVE_PICTURE, true_cfg_scale=4.0,
                        num_inference_steps=steps, height=H, width=W,
                        generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        except Exception:  # noqa: BLE001 - one bad item never stops the stage
            log("FAILED", name, traceback.format_exc(limit=2).replace("\n", " | "))
            return None

    canvas = Image.new("RGB", (W, H), (96, 96, 96))
    refs: dict[str, Image.Image] = {}
    for k, ref in enumerate(job["references"]):  # bases first, so a derived character can start from one
        path = out / ref["file"]
        if path.exists():
            refs[ref["name"]] = Image.open(path).convert("RGB")
            continue
        start = [refs[ref["based_on"]]] if ref.get("based_on") in refs else [canvas]
        picture = draw(start, ref["prompt"], 500 + 31 * k, ref["name"])
        if picture is not None:
            picture.save(path)
            refs[ref["name"]] = picture
            log("reference", ref["name"], "done")
    for shot in job["shots"]:
        path = out / f"shot_{shot['index']:03d}.png"
        if path.exists():
            continue
        names = [n for n in shot["refs"] if n in refs][:3]
        prompt = shot["picture_prompt"]
        for k, n in enumerate(names, 1):
            prompt = prompt.replace(f"{{ref:{n}}}", f"Picture {k}")
        picture = draw([refs[n] for n in names] or [canvas], prompt, 1000 + 7 * shot["index"], f"shot {shot['index']}")
        if picture is not None:
            picture.save(path)
            log("picture", shot["index"], "done")
    del pipe
    free()


# --- talking clips (Wan2.2-S2V) -------------------------------------------------------------------
SHIM = """
import runpy, sys
import wan.modules.attention as att
if not (att.FLASH_ATTN_2_AVAILABLE or getattr(att, 'FLASH_ATTN_3_AVAILABLE', False)):
    def _sdpa(q, k, v, q_lens=None, k_lens=None, *a, **kw):
        return att.attention(q, k, v, q_lens=q_lens, k_lens=k_lens, **{x: y for x, y in kw.items()
                             if x in ('dropout_p', 'softmax_scale', 'q_scale', 'causal', 'window_size',
                                      'deterministic', 'dtype')})
    original = att.flash_attention
    for module in list(sys.modules.values()):  # wan/__init__ already imported the models, which bound the name
        if getattr(module, 'flash_attention', None) is original:
            module.flash_attention = _sdpa
    print('flash-attn missing: using PyTorch attention', flush=True)
sys.argv = ['generate.py'] + sys.argv[1:]
runpy.run_path('generate.py', run_name='__main__')
"""


def talk(job: dict, out: Path) -> None:
    repo = Path(job.get("wan_repo", "/workspace/Wan2.2"))
    (repo / "run_s2v.py").write_text(SHIM)
    python = job.get("wan_python", sys.executable)
    for shot in [s for s in job["shots"] if s["mode"] == "talk"]:
        src, wav = out / f"shot_{shot['index']:03d}.png", out / f"line_{shot['index']:03d}.wav"
        dst = out / f"clip_{shot['index']:03d}.mp4"
        if (out / f"clip_{shot['index']:03d}.talk").exists() or not src.exists() or not wav.exists():
            continue
        started = time.time()
        args = [python, "run_s2v.py", "--task", "s2v-14B", "--size", job.get("talk_size", "1024*704"),
                "--ckpt_dir", job["talk_dir"], "--offload_model", "False", "--convert_model_dtype",
                "--sample_steps", str(job.get("talk_steps", 40)), "--base_seed", str(42 + shot["index"]),
                "--prompt", shot["motion_prompt"], "--image", str(src), "--audio", str(wav), "--save_file", str(dst)]
        # S2V writes beside the clip and replaces it only when it finishes, so a moving clip from an
        # earlier run survives a failure or the spending limit stopping the machine mid-shot (run 88).
        made = dst.with_suffix(".s2v.mp4")
        args[args.index("--save_file") + 1] = str(made)
        result = subprocess.run(args, cwd=repo, capture_output=True, text=True)
        if result.returncode != 0 or not made.exists():
            log("FAILED talk", shot["index"], (result.stderr + result.stdout)[-900:].replace("\n", " | "))
            continue
        made.replace(dst)
        (out / f"clip_{shot['index']:03d}.talk").write_text("Wan2.2-S2V")  # lip-synced, not a fallback
        log("talk", shot["index"], f"done in {time.time() - started:.0f} s")


# --- moving clips (Wan 2.2 I2V) -----------------------------------------------------------------
def move(job: dict, out: Path) -> None:
    import soundfile
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    from PIL import Image

    todo = [s for s in job["shots"] if not (out / f"clip_{s['index']:03d}.mp4").exists()
            and (out / f"shot_{s['index']:03d}.png").exists()]
    todo.sort(key=lambda s: (s["mode"] == "talk", s["index"]))  # talk shots here are fallbacks: last
    if not todo:
        return
    W, H = job.get("clip_size", [720, 1280])
    vae = AutoencoderKLWan.from_pretrained(job["clip_model"], subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(job["clip_model"], vae=vae, torch_dtype=torch.bfloat16)
    if torch.cuda.get_device_properties(0).total_memory > 90e9:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    log("clip model loaded")
    for shot in todo:
        src, dst = out / f"shot_{shot['index']:03d}.png", out / f"clip_{shot['index']:03d}.mp4"
        wav = out / f"line_{shot['index']:03d}.wav"
        seconds = float(shot.get("seconds", 3.0))
        if wav.exists():
            seconds = soundfile.info(str(wav)).duration
        frames = min(81, max(33, int(seconds * 16) // 4 * 4 + 1))
        started = time.time()
        try:
            image = Image.open(src).convert("RGB")
            scale = max(W / image.width, H / image.height)
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
            x, y = (image.width - W) // 2, (image.height - H) // 2
            image = image.crop((x, y, x + W, y + H))
            result = None
            for offload in (False, True):
                try:
                    result = pipe(image=image, prompt=shot["motion_prompt"], negative_prompt=NEGATIVE_CLIP,
                                  height=H, width=W, num_frames=frames, guidance_scale=3.5,
                                  num_inference_steps=int(job.get("clip_steps", 30)),
                                  generator=torch.Generator("cuda").manual_seed(7 + shot["index"])).frames[0]
                    break
                except torch.cuda.OutOfMemoryError:
                    if offload:
                        raise
                    log("out of GPU memory: moving the clip model to CPU offload")
                    free()
                    pipe.to("cpu")
                    pipe.enable_model_cpu_offload()
            export_to_video(result, str(dst), fps=16)
            log("move", shot["index"], f"done in {time.time() - started:.0f} s")
        except Exception:  # noqa: BLE001
            log("FAILED move", shot["index"], traceback.format_exc(limit=2).replace("\n", " | "))
            free()
    del pipe
    free()


# --- sound (MMAudio) ----------------------------------------------------------------------------
def sound(job: dict, out: Path) -> None:
    root = Path(job.get("mmaudio_dir", "/workspace/MMAudio"))
    for item in job.get("audio", []):
        target = out / f"{item['name']}.flac"
        if target.exists():
            continue
        folder = out / "mm" / item["name"]
        folder.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([sys.executable, "demo.py", "--prompt", item["prompt"], "--duration",
                                 str(item["duration"]), "--output", str(folder), "--negative_prompt",
                                 item.get("negative", "music, speech, voices, talking"), "--skip_video_composite"],
                                cwd=root, capture_output=True, text=True)
        made = sorted(folder.rglob("*.flac"), key=lambda p: p.stat().st_mtime)
        if result.returncode != 0 or not made:
            log("FAILED audio", item["name"], result.stderr[-300:].replace("\n", " | "))
            continue
        made[-1].replace(target)
        shutil.rmtree(folder, ignore_errors=True)
        log("audio", item["name"], "done")


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    stage = sys.argv[2]
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log("stage", stage, "starting")
    {"voices": voices, "pictures": pictures, "talk": talk, "move": move, "sound": sound}[stage](job, out)
    log("stage", stage, "finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
