import os
import uuid
import numpy as np
import torch
import hashlib
import cv2
import ffmpeg
from PIL import Image
from flask import request, render_template
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import pyqrcode
from snowflake import SnowflakeGenerator

UPLOAD_DIR = "saved"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Setup devices and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
sf_gen = SnowflakeGenerator(0)

# Augmentation for embedding generation
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.8),
    transforms.RandomApply([transforms.GaussianBlur((3, 3), sigma=(0.1, 2.0))], p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(size=(160, 160), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
])

def get_upload_form():
    return render_template("pload_form.html", fields=["reference image", "video", "watermark image"])

def sha256_of_file(path, chunk_size=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def generate_qr_code(data, output_path):
    qr = pyqrcode.create(f"http://127.0.0.1:5000/partner/{data}")
    qr_path = output_path.replace(".mp4", "_qr.png")
    qr.png(qr_path, scale=6)
    return qr_path

def generate_video_keys(video_path):
    content_hash = sha256_of_file(video_path)
    unique_id = next(sf_gen)
    return {"video_id": unique_id, "content_hash": content_hash}

def build_prototype_embedding(image_path):
    orig = Image.open(image_path).convert('RGB')
    embs = []

    for _ in range(16):
        img_aug = augment(orig)
        face = mtcnn(img_aug)
        if face is None:
            face = mtcnn(orig)
        if face is not None:
            with torch.no_grad():
                emb = resnet(face.unsqueeze(0).to(device))
            embs.append(emb.cpu().numpy()[0])


    if not embs:
        raise RuntimeError("No face detected in any variant.")

    proto = np.mean(np.stack(embs), axis=0)
    proto /= np.linalg.norm(proto)
    return proto

def process_video(video_path, ref_emb, threshold=0.75, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    matches, frame_idx = [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face = mtcnn(img)
            if face is not None:
                with torch.no_grad():
                    emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]
                cos_sim = np.dot(ref_emb, emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb))
                matches.append(cos_sim)
        frame_idx += 1

    cap.release()
    if not matches:
        return False, None
    max_sim = float(np.max(matches))
    return max_sim >= threshold, max_sim

def watermark_video(input_path, watermark_path, output_path, scale_pct=0.5, opacity=0.8):
    video = ffmpeg.input(input_path)
    wm = ffmpeg.input(watermark_path)

    scaled_wm = wm.filter("scale", f"iw*{scale_pct}", -1)
    wm_rgba = scaled_wm.filter("format", "rgba").filter("colorchannelmixer", aa=opacity)
    overlaid = ffmpeg.overlay(video.video, wm_rgba, x="(W-w)/2", y="(H-h)/2")

    ffmpeg.output(overlaid, video.audio, output_path, vcodec="libx264", acodec="copy", format="mp4").overwrite_output().run()

def handle_input():
    if request.method == "POST":
        image_file = request.files.get("reference")
        video_file = request.files.get("video")
        watermark_file = request.files.get("watermark")

        if not all([image_file, video_file, watermark_file]):
            return render_template("error.html", message="All 3 files (reference, video, watermark) are required.")

        ref_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_ref.png")
        vid_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_vid.mp4")
        wm_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_wm.png")

        image_file.save(ref_path)
        video_file.save(vid_path)
        watermark_file.save(wm_path)

        try:
            ref_emb = build_prototype_embedding(ref_path)
            match, score = process_video(vid_path, ref_emb)

            if match:
                output_path = vid_path.replace(".mp4", "_watermarked.mp4")
                watermark_video(vid_path, wm_path, output_path)
                sha = sha256_of_file(output_path)
                keys = generate_video_keys(output_path)
                qr_path = generate_qr_code(keys['video_id'], output_path)

                return render_template("result.html", result={
                    "status": "Match Found ✅",
                    "score": f"{score:.4f}",
                    "sha256": sha,
                    "video_id": keys["video_id"],
                    "content_hash": keys["content_hash"],
                    "video_url": output_path,
                    "qr_code_path": qr_path
                })

            return render_template("result.html", result={"status": "No match ❌", "score": f"{score:.4f}"})
        except Exception as e:
            print("Processing error:", e)
            return render_template("error.html", message=f"Processing error: {str(e)}")

    else:
        return render_template("upload_form.html")
 
