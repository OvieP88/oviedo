import os
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client

# -------------------------
# ENV SETUP & CLEANUP
# -------------------------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

WORK_DIR = Path("/tmp/screensox")
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Cookies (optional)
cookies_content = os.environ.get("YOUTUBE_COOKIES")
COOKIE_PATH = Path("/tmp/yt_cookies.txt")
if cookies_content:
    COOKIE_PATH.write_text(cookies_content)

print(f"📦 Workspace: {WORK_DIR}")
print(f"🍪 Cookie Setup: {'READY' if COOKIE_PATH.exists() else 'SKIPPED'}")

# -------------------------
# LOAD MODEL
# -------------------------
print("🔄 Loading CLIP...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.eval()
print(f"✅ CLIP ready on {device.upper()}")

# -------------------------
# JOB CLAIMING
# -------------------------
def claim_job():
    res = sb.table("movies").select("*").eq("status", "pending").limit(1).execute()
    if not res.data:
        return None

    job = res.data[0]  # ✅ FIXED
    vid = job["youtube_id"]

    sb.table("movies").update({
        "status": "processing"
    }).eq("youtube_id", vid).eq("status", "pending").execute()

    return job

# -------------------------
# DOWNLOAD VIDEO
# -------------------------
def get_stream_url(vid):
    url = f"https://youtu.be/{vid}"
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--extractor-args", "youtube:player_client=android",
        "--user-agent", "com.google.android.youtube/17.31.35 (Linux; U; Android 11) gzip",
        "-g",
        "-f", "best[height<=360]/best",
        url
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise Exception(f"stream URL failed: {r.stderr[:100]}")
    return r.stdout.strip().split("\n")[0]

def extract_frames(vid, stream_url, duration):
    frames = []
    start  = 600
    end    = max(duration - 120, start + 1)

    for idx, ts in enumerate(range(start, end, 10)):
        fp = WORK_DIR / f"{vid}_{idx:04d}.jpg"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ts),
            "-i", stream_url,
            "-vframes", "1",
            "-q:v", "3",
            "-vf", "scale=640:-1",
            str(fp),
            "-loglevel", "error"
        ]
        subprocess.run(cmd, capture_output=True, timeout=30)
        if fp.exists() and fp.stat().st_size > 3000:
            frames.append((fp, ts, idx))

    return frames
  
# -------------------------
# GET DURATION
# -------------------------
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
    except:
        return 0

# -------------------------
# EXTRACT FRAMES
# -------------------------
def extract_frames(video_path, vid, duration):
    frames = []
    start = 600
    end = max(duration - 120, start + 1)

    for idx, ts in enumerate(range(start, end, 15)):
        fp = WORK_DIR / f"{vid}_{idx}.jpg"

        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(ts),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "3",
            "-vf", "scale=640:-1",
            str(fp)
        ], capture_output=True)

        if fp.exists() and fp.stat().st_size > 3000:
            frames.append((fp, ts, idx))

    return frames

# -------------------------
# EMBEDDING
# -------------------------
def flush_embeddings(images, meta, vid):
    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)

    rows = []
    for i, (fp, ts, idx) in enumerate(meta):
        rows.append({
            "youtube_id": vid,
            "frame_index": idx,
            "timestamp_sec": ts,
            "embedding": feats[i].cpu().tolist()
        })
        fp.unlink(missing_ok=True)

    sb.table("embeddings").insert(rows).execute()
    return len(rows)

# -------------------------
# PROCESS JOB
# -------------------------
def process(job):
    vid   = job["youtube_id"]
    title = job.get("title", "Unknown")

    print(f"\n🎬 Processing: {title[:50]} ({vid})")

    try:
        stream_url = get_stream_url(vid)
        print(f"✅ Stream URL obtained")

        # Get duration via yt-dlp metadata (no download)
        info_cmd = [
            "yt-dlp", "--no-warnings",
            "--extractor-args", "youtube:player_client=android",
            "--dump-json", f"https://youtu.be/{vid}"
        ]
        info = subprocess.run(info_cmd, capture_output=True, text=True, timeout=60)
        import json
        duration = int(json.loads(info.stdout).get("duration", 5400))
        print(f"⏱ Duration: {duration}s")

        frames = extract_frames(vid, stream_url, duration)
        print(f"🖼 {len(frames)} frames extracted")

        if not frames:
            raise Exception("No frames extracted")

        total_saved = 0
        for i in range(0, len(frames), 16):
            batch = frames[i:i+16]
            imgs  = [Image.open(fp).convert("RGB") for fp, _, _ in batch]
            total_saved += flush_embeddings(imgs, batch, vid)
            for img in imgs:
                img.close()

        sb.table("movies").update({
            "status"     : "processed",
            "frame_count": total_saved,
            "indexed_at" : datetime.utcnow().isoformat(),
            "error_msg"  : None
        }).eq("youtube_id", vid).execute()

        print(f"✅ Success: {total_saved} embeddings")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        retries = job.get("retries", 0)
        sb.table("movies").update({
            "status"   : "failed" if retries >= 3 else "pending",
            "retries"  : retries + 1,
            "error_msg": str(e)[:200]
        }).eq("youtube_id", vid).execute()
    finally:
        if video_path.exists():
            video_path.unlink()

# -------------------------
# MAIN LOOP
# -------------------------
print("🚀 Worker started")

while True:
    job = claim_job()

    if not job:
        print("😴 No jobs. Sleeping 15s...")
        time.sleep(15)
        continue

    process(job)

    # ✅ safer rate limiting
    time.sleep(5)
