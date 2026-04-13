import os
import time
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client

# ── ENV SETUP ─────────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

WORK_DIR = Path("/tmp/screensox")
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Cookies (optional — set YOUTUBE_COOKIES secret in GitHub)
COOKIE_PATH = Path("/tmp/yt_cookies.txt")
cookies_content = os.environ.get("YOUTUBE_COOKIES")
if cookies_content:
    COOKIE_PATH.write_text(cookies_content)

print(f"📦 Workspace: {WORK_DIR}")
print(f"🍪 Cookies  : {'READY' if COOKIE_PATH.exists() else 'SKIPPED'}")

# ── LOAD CLIP ─────────────────────────────────────────────────
print("🔄 Loading CLIP...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print(f"✅ CLIP ready on {device.upper()}")

# ── CLAIM JOB ─────────────────────────────────────────────────
def claim_job():
    res = sb.table("movies") \
            .select("*") \
            .eq("status", "pending") \
            .limit(1) \
            .execute()
    if not res.data:
        return None
    job = res.data[0]
    sb.table("movies").update({
        "status": "processing"
    }).eq("youtube_id", job["youtube_id"]).eq("status", "pending").execute()
    return job

# ── DOWNLOAD VIDEO ────────────────────────────────────────────
def download_video(vid, out_path):
    url = f"https://youtu.be/{vid}"

    base_cmd = [
        "yt-dlp",
        "--no-warnings",
        "--sleep-interval", "3",
        "--max-sleep-interval", "8",
        "--retries", "5",
        "--fragment-retries", "5",
        "--extractor-args", "youtube:player_client=android",
        "--user-agent", "com.google.android.youtube/17.31.35 (Linux; U; Android 11) gzip",
        "-f", "best[height<=360]/best",
        "-o", str(out_path),
        url
    ]

    # Try with cookies first if available
    if COOKIE_PATH.exists() and COOKIE_PATH.stat().st_size > 0:
        try:
            print("🍪 Trying with cookies...")
            cmd = ["yt-dlp", "--cookies", str(COOKIE_PATH)] + base_cmd[1:]
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=900)
            return
        except Exception as e:
            print(f"⚠️  Cookie download failed: {e}")

    print("🌐 Downloading without cookies...")
    subprocess.run(base_cmd, check=True, capture_output=True, text=True, timeout=900)

# ── GET DURATION ──────────────────────────────────────────────
def get_duration(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(float(r.stdout.strip()))
    except Exception:
        return 0

# ── EXTRACT FRAMES ────────────────────────────────────────────
def extract_frames(vid, video_path, duration):
    frames = []
    start  = 600                        # skip first 10 min (intro/credits)
    end    = max(duration - 120, start + 1)   # stop 2 min before end

    for idx, ts in enumerate(range(start, end, 10)):   # 1 frame every 10s
        fp = WORK_DIR / f"{vid}_{idx:04d}.jpg"
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(ts),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "3",
            "-vf", "scale=640:-1",
            str(fp),
            "-loglevel", "error"
        ], capture_output=True, timeout=30)

        if fp.exists() and fp.stat().st_size > 3000:
            frames.append((fp, ts, idx))

    return frames

# ── EMBED + STORE ─────────────────────────────────────────────
def flush_embeddings(images, meta, vid):
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)

    rows = []
    for i, (fp, ts, idx) in enumerate(meta):
        rows.append({
            "youtube_id"   : vid,
            "frame_index"  : idx,
            "timestamp_sec": ts,
            "embedding"    : feats[i].cpu().tolist(),
        })
        fp.unlink(missing_ok=True)

    sb.table("embeddings").insert(rows).execute()
    return len(rows)

# ── PROCESS ONE MOVIE ─────────────────────────────────────────
def process(job):
    vid        = job["youtube_id"]
    title      = job.get("title", "Unknown")
    video_path = WORK_DIR / f"{vid}.mp4"

    print(f"\n🎬 {title[:60]} ({vid})")

    try:
        # 1. Download at 360p (fast on GitHub's servers ~1-2 min)
        download_video(vid, video_path)
        size_mb = video_path.stat().st_size / 1024 / 1024
        print(f"   ✅ Downloaded: {size_mb:.1f} MB")

        # 2. Get duration
        duration = get_duration(video_path)
        print(f"   ⏱  Duration : {duration // 60}m {duration % 60}s")

        if duration < 660:
            raise Exception("Video too short (under 11 min)")

        # 3. Extract frames every 10s
        frames = extract_frames(vid, video_path, duration)
        print(f"   🖼  Frames   : {len(frames)}")

        if not frames:
            raise Exception("No frames extracted")

        # 4. Delete video immediately — frames are all we need
        video_path.unlink(missing_ok=True)

        # 5. Embed in batches of 16
        total_saved = 0
        for i in range(0, len(frames), 16):
            batch = frames[i:i + 16]
            imgs  = [Image.open(fp).convert("RGB") for fp, _, _ in batch]
            total_saved += flush_embeddings(imgs, batch, vid)
            for img in imgs:
                img.close()

        # 6. Mark done
        sb.table("movies").update({
            "status"     : "processed",
            "frame_count": total_saved,
            "indexed_at" : datetime.utcnow().isoformat(),
            "error_msg"  : None,
        }).eq("youtube_id", vid).execute()

        print(f"   ✅ {total_saved} embeddings saved")

    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        retries = job.get("retries", 0)
        sb.table("movies").update({
            "status"   : "failed" if retries >= 3 else "pending",
            "retries"  : retries + 1,
            "error_msg": str(e)[:200],
        }).eq("youtube_id", vid).execute()

    finally:
        # Always clean up disk
        video_path.unlink(missing_ok=True)

# ── MAIN LOOP ─────────────────────────────────────────────────
print("\n🚀 Worker started")
processed = 0

while True:
    job = claim_job()

    if not job:
        print(f"\n😴 No pending jobs. Total processed this run: {processed}")
        break   # exit cleanly — GitHub Actions will re-run on schedule

    process(job)
    processed += 1
    print(f"   📊 Run total: {processed}")
    time.sleep(3)   # brief pause between movies
