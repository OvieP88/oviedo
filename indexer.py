import os, subprocess, time, json, torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client
# Write cookies from secret
import os, pathlib
cookies_content = os.environ.get("YOUTUBE_COOKIES", "")
if cookies_content:
    pathlib.Path("/tmp/yt_cookies.txt").write_text(cookies_content)


sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
WORK_DIR = Path("/tmp/screensox")
WORK_DIR.mkdir(exist_ok=True)

print("Loading CLIP...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print("CLIP ready")

def get_batch():
    r = sb.table("movies").select("youtube_id,title").eq("status","pending").limit(10).execute()
    return r.data or []

def process(movie):
    vid = movie["youtube_id"]
    url = f"https://youtu.be/{vid}"
    print(f"\n🎬 {movie['title'][:50]}")
    out = WORK_DIR / f"{vid}.mp4"

    sb.table("movies").update({"status":"processing"}).eq("youtube_id",vid).execute()

    r = subprocess.run([
        "yt-dlp","--no-warnings","-f","worst[ext=mp4]/worst",
        "--merge-output-format","mp4","-o",str(out),url
    ], capture_output=True, text=True, timeout=600)

    if r.returncode != 0 or not out.exists():
        print(f"  ❌ Download failed: {r.stderr[-100:]}")
        sb.table("movies").update({"status":"failed"}).eq("youtube_id",vid).execute()
        return

    size = out.stat().st_size/1024/1024
    print(f"  ✅ {size:.1f}MB downloaded")

    # Get duration
    probe = subprocess.run(["ffmpeg","-i",str(out)],capture_output=True,text=True)
    duration = 6000
    for line in probe.stderr.split("\n"):
        if "Duration" in line:
            try:
                t = line.split("Duration:")[1].split(",")[0].strip()
                h,m,s = t.split(":")
                duration = int(h)*3600+int(m)*60+int(float(s))
            except: pass

    # Extract frames every 10s
    frames, i = [], 0
    for ts in range(600, duration-120, 10):
        fp = WORK_DIR/f"{vid}_{i:04d}.jpg"
        subprocess.run([
            "ffmpeg","-y","-ss",str(ts),"-i",str(out),
            "-vframes","1","-q:v","3","-vf","scale=640:-1",str(fp)
        ], capture_output=True, timeout=30)
        if fp.exists() and fp.stat().st_size > 3000:
            frames.append((fp, ts))
            i += 1

    out.unlink(missing_ok=True)
    print(f"  ✅ {len(frames)} frames")

    # Embed
    batch, saved = [], 0
    for fp, ts in frames:
        try:
            img = Image.open(fp).convert("RGB")
            inp = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                v = model.vision_model(pixel_values=inp["pixel_values"])
                feat = model.visual_projection(v.pooler_output)
                feat = feat/feat.norm(p=2,dim=-1,keepdim=True)
            batch.append({"youtube_id":vid,"frame_name":fp.name,
                          "timestamp_sec":ts,"vector":json.dumps(feat[0].tolist())})
            fp.unlink(missing_ok=True)
            if len(batch)>=100:
                sb.table("clip_frames").insert(batch).execute()
                saved+=len(batch); batch=[]
        except Exception as e:
            print(f"  ⚠️ {e}")
    if batch:
        sb.table("clip_frames").insert(batch).execute()
        saved+=len(batch)

    sb.table("movies").update({"status":"processed","indexed_at":"now()"}).eq("youtube_id",vid).execute()
    print(f"  ✅ {saved} embeddings saved → processed")

# Main
movies = get_batch()
print(f"📋 {len(movies)} pending movies")
for m in movies: process(m)
print("\n✅ Done")
