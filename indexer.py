import os
import time
import subprocess
from pathlib import Path
import pathlib
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client

# -------------------------
# ENV SETUP
# -------------------------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

WORK_DIR = Path("/tmp/screensox")
WORK_DIR.mkdir(exist_ok=True)

# Write cookies if provided
cookies_content = os.environ.get("YOUTUBE_COOKIES")
COOKIE_PATH = "/tmp/yt_cookies.txt"
if cookies_content:
    pathlib.Path(COOKIE_PATH).write_text(cookies_content)


print("COOKIE EXISTS:", os.path.exists(COOKIE_PATH))
print("COOKIE SIZE:", len(cookies_content or ""))

# -------------------------
# LOAD MODEL
# -------------------------
print("🔄 Loading CLIP...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.eval()
print(f"✅ CLIP ready on {device}")

# -------------------------
# JOB CLAIMING (SAFE)
# -------------------------
def claim_job():
    """Atomically claim 1 pending job"""
    res = sb.table("movies") \
        .select("*") \
        .eq("status", "pending") \
        .limit(1) \
        .execute()

    jobs = res.data
    if not jobs:
        return None

    job = jobs[0]
    vid = job["youtube_id"]

    # attempt to lock it
    sb.table("movies").update({
        "status": "processing"
    }).eq("youtube_id", vid).eq("status", "pending").execute()

    return job

# -------------------------
# DOWNLOAD VIDEO
# -------------------------
def download_video(vid, out_path):
    url = f"https://youtu.be/{vid}"

    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--cookies", COOKIE_PATH,
        "--sleep-interval", "2",
        "--max-sleep-interval", "5",
        "-f", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "-o", str(out_path),
        url
    ]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if r.returncode != 0:
        raise Exception(r.stderr[-300:])

# -------------------------
# GET DURATION
# -------------------------
def get_duration(path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
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

    for idx, ts in enumerate(range(600, max(duration - 120, 600), 15)):
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
def embed_and_store(frames, vid):
    batch_imgs = []
    meta = []
    saved = 0

    for fp, ts, idx in frames:
        try:
            img = Image.open(fp).convert("RGB")
            batch_imgs.append(img)
            meta.append((fp, ts, idx))

            if len(batch_imgs) >= 16:
                saved += flush_embeddings(batch_imgs, meta, vid)
                batch_imgs, meta = [], []

        except Exception as e:
            print(f"⚠️ Image error: {e}")

    if batch_imgs:
        saved += flush_embeddings(batch_imgs, meta, vid)

    return saved

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

    res = sb.table("embeddings").insert(rows).execute()

    if res.data is None:
        raise Exception("Embedding insert failed")

    return len(rows)

# -------------------------
# PROCESS JOB
# -------------------------
def process(job):
    vid = job["youtube_id"]
    title = job.get("title", "")

    print(f"\n🎬 Processing: {title[:50]}")

    video_path = WORK_DIR / f"{vid}.mp4"

    try:
        # download
        download_video(vid, video_path)
        print("✅ Downloaded")

        # duration
        duration = get_duration(video_path)
        print(f"⏱ Duration: {duration}s")

        if duration < 600:
            raise Exception("Video too short")

        # frames
        frames = extract_frames(video_path, vid, duration)
        print(f"🖼 Frames: {len(frames)}")

        if not frames:
            raise Exception("No frames extracted")

        # embeddings
        saved = embed_and_store(frames, vid)
        print(f"✅ Embeddings: {saved}")

        # mark success
        sb.table("movies").update({
            "status": "processed",
            "frame_count": saved,
            "indexed_at": "now()"
        }).eq("youtube_id", vid).execute()

    except Exception as e:
        print(f"❌ FAILED: {e}")

        sb.table("movies").update({
            "status": "failed",
            "error_msg": str(e)[:200]
        }).eq("youtube_id", vid).execute()

    finally:
        video_path.unlink(missing_ok=True)

# -------------------------
# MAIN WORKER LOOP
# -------------------------
print("🚀 Worker started")

while True:
    job = claim_job()

    if not job:
        print("😴 No jobs. Sleeping...")
        time.sleep(10)
        continue

    process(job)
    time.sleep(3)
