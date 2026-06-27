"""Build degraded test set from easy50: 5 realistic transforms per sample."""
import json, os, sys, random
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2

DATASET_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
OUT_DIR = f"{DATASET_DIR}/data/test_degraded"
os.makedirs(OUT_DIR, exist_ok=True)

DEGRADATIONS = ['perspective', 'lighting', 'blur', 'noise', 'jpeg']

def apply_perspective(img):
    """Simulate camera tilt: slight perspective warp."""
    w, h = img.size
    margin = int(min(w, h) * 0.08)
    src = np.float32([[margin, margin], [w-margin, margin], [margin, h-margin], [w-margin, h-margin]])
    jitter = margin * 0.7
    dst = np.float32([
        [margin + random.uniform(-jitter, jitter), margin + random.uniform(-jitter, jitter)],
        [w-margin + random.uniform(-jitter*0.5, jitter*0.5), margin + random.uniform(-jitter, jitter)],
        [margin + random.uniform(-jitter, jitter), h-margin + random.uniform(-jitter*0.5, jitter*0.5)],
        [w-margin + random.uniform(-jitter*0.3, jitter*0.3), h-margin + random.uniform(-jitter*0.3, jitter*0.3)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    warped = cv2.warpPerspective(img_cv, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

def apply_lighting(img):
    """Simulate lighting variation: random brightness/contrast."""
    factor_b = random.uniform(0.5, 1.8)
    factor_c = random.uniform(0.6, 1.4)
    img = ImageEnhance.Brightness(img).enhance(factor_b)
    img = ImageEnhance.Contrast(img).enhance(factor_c)
    return img

def apply_blur(img):
    """Simulate focus/jitter: Gaussian blur."""
    sigma = random.uniform(0.8, 2.5)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))

def apply_noise(img):
    """Simulate sensor noise: salt-pepper + Gaussian."""
    arr = np.array(img).astype(np.float32)
    # Gaussian noise
    sigma = random.uniform(5, 20)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    # Salt & pepper
    sp_prob = random.uniform(0.005, 0.03)
    mask = np.random.random(arr.shape[:2]) < sp_prob
    arr[mask] = random.choice([0, 255])
    return Image.fromarray(arr.astype(np.uint8))

def apply_jpeg(img):
    """Simulate multiple re-encodes: heavy JPEG compression."""
    from io import BytesIO
    quality = random.randint(15, 50)
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf)

TRANSFORMS = {
    'perspective': apply_perspective,
    'lighting': apply_lighting,
    'blur': apply_blur,
    'noise': apply_noise,
    'jpeg': apply_jpeg,
}

def main():
    src_jsonl = f"{DATASET_DIR}/ocr_vl_sft-test-easy50.jsonl"
    with open(src_jsonl, encoding='utf-8') as f:
        samples = [json.loads(l) for l in f if l.strip()]

    degraded_samples = []
    total = 0

    for si, sample in enumerate(samples):
        img_rel = sample['images'][0].lstrip('./')
        img_path = f"{DATASET_DIR}/{img_rel}"
        if not os.path.exists(img_path):
            # Try jpeg variant
            alt = img_path.replace('.png','.jpg').replace('.JPG','.jpg').replace('.jpeg','.jpg')
            if os.path.exists(alt):
                img_path = alt
            else:
                print(f"  SKIP missing: {img_rel}")
                continue

        try:
            orig = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"  SKIP {img_rel}: {e}")
            continue

        base_name = Path(img_rel).stem
        ext = Path(img_rel).suffix or '.jpg'

        for deg in DEGRADATIONS:
            try:
                degraded = TRANSFORMS[deg](orig.copy())
                out_name = f"{base_name}_{deg}{ext}"
                out_path = f"{OUT_DIR}/{out_name}"
                degraded.save(out_path, quality=95)
                new_sample = {
                    'messages': sample['messages'],
                    'images': [f"./data/test_degraded/{out_name}"],
                    'degradation': deg,
                    'original': sample['images'][0],
                }
                degraded_samples.append(new_sample)
                total += 1
            except Exception as e:
                print(f"  FAIL {img_rel} {deg}: {e}")

        if (si + 1) % 20 == 0:
            print(f"  [{si+1}/{len(samples)}] {total} degraded samples generated")

    # Save
    out_jsonl = f"{DATASET_DIR}/ocr_vl_sft-test-easy50-degraded.jsonl"
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for s in degraded_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"\nDone: {len(samples)} originals → {total} degraded samples")
    print(f"Saved to: {out_jsonl}")
    # Print distribution
    from collections import Counter
    dist = Counter(s['degradation'] for s in degraded_samples)
    for k, v in sorted(dist.items()):
        print(f"  {k}: {v}")

if __name__ == '__main__':
    main()
