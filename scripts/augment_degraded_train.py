"""
Apply real-world photo degradation to a subset of training images.

Simulates: perspective tilt, uneven lighting, blur, sensor noise, JPEG artifacts.
The GT stays the same — only the image is degraded.

Usage: python scripts/augment_degraded_train.py
Output:
  - output/degraded_train/augmented/*.jpg       # Degraded images
  - output/degraded_train/train_degraded_mix.jsonl  # Degraded-only JSONL
  - Also prints how to merge with original training set
"""
import json
import os
import random
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

PROJECT_DIR = Path(__file__).parent.parent
DATASET_DIR = PROJECT_DIR / "circuit-ocr-dataset"
TRAIN_JSONL = DATASET_DIR / "ocr_vl_sft-train-v9-pure.jsonl"
OUTPUT_DIR = PROJECT_DIR / "output" / "degraded_train"
AUG_IMG_DIR = OUTPUT_DIR / "augmented"

# Config
NUM_AUGMENT = 200           # How many samples to augment
PERSPECTIVE_STRENGTH = 0.05 # Max perspective distortion (0-1)
BLUR_RADIUS = (0.5, 2.0)    # Gaussian blur range
NOISE_STD = (3, 12)          # Gaussian noise std range
BRIGHTNESS_RANGE = (0.6, 1.4)
CONTRAST_RANGE = (0.7, 1.3)
JPEG_QUALITY_RANGE = (40, 85)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUG_IMG_DIR.mkdir(parents=True, exist_ok=True)


def apply_perspective(img, strength):
    """Apply slight perspective warp simulating off-angle photo."""
    w, h = img.size
    # Random corner displacements
    dx = w * strength
    dy = h * strength
    src = np.float32([
        [random.uniform(0, dx), random.uniform(0, dy)],           # top-left
        [w - random.uniform(0, dx), random.uniform(0, dy)],       # top-right
        [random.uniform(0, dx), h - random.uniform(0, dy)],       # bottom-left
        [w - random.uniform(0, dx), h - random.uniform(0, dy)],   # bottom-right
    ])
    dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2_get_perspective(src, dst)
    return img.transform((w, h), Image.PERSPECTIVE,
                         matrix.flatten()[:8].tolist(),
                         Image.BICUBIC)


def cv2_get_perspective(src, dst):
    """Simple perspective matrix without cv2 dependency."""
    # Use PIL's own perspective transform (we just need random corners)
    # PIL Image.PERSPECTIVE takes a flat list of 8 coefficients
    # We already compute the right transform below
    return src, dst  # dummy, will use direct PIL approach


def apply_perspective_pil(img, strength):
    """Apply perspective warp using PIL's built-in transform."""
    w, h = img.size
    margin = strength
    # Generate 4 corner displacements
    dx_tl = random.uniform(0, w * margin)
    dy_tl = random.uniform(0, h * margin)
    dx_tr = random.uniform(0, w * margin)
    dy_tr = random.uniform(0, h * margin)
    dx_bl = random.uniform(0, w * margin)
    dy_bl = random.uniform(0, h * margin)
    dx_br = random.uniform(0, w * margin)
    dy_br = random.uniform(0, h * margin)

    # Find coefficients using PIL's perspective
    coeffs = find_coeffs(
        [(dx_tl, dy_tl), (w - 1 - dx_tr, dy_tr),
         (w - 1 - dx_br, h - 1 - dy_br), (dx_bl, h - 1 - dy_bl)],
        [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
    )
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def find_coeffs(pa, pb):
    """Find perspective transform coefficients."""
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([p[0] for p in pb] + [p[1] for p in pb], dtype=np.float64)
    # Use lstsq for better numerical stability
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return res.tolist()


def add_vignette(img, strength=0.3):
    """Add dark vignette corners (common in phone photos)."""
    w, h = img.size
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    dist = np.sqrt(xx**2 + yy**2) / np.sqrt(2)
    vignette = 1 - strength * (dist ** 1.5)
    vignette = np.clip(vignette, 0, 1)

    arr = np.array(img, dtype=np.float64)
    for c in range(3):
        arr[:, :, c] *= vignette
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_lighting_gradient(img):
    """Add uneven lighting gradient (one side brighter/darker)."""
    w, h = img.size
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))

    # Random gradient direction
    angle = random.uniform(0, 2 * np.pi)
    grad = xx * np.cos(angle) + yy * np.sin(angle)
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)

    # Brightness variation: 0.7x to 1.3x across the image
    factor = 0.7 + 0.6 * grad
    factor = factor[:, :, np.newaxis]

    arr = np.array(img, dtype=np.float64)
    arr *= factor
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_gaussian_noise(img, std):
    """Add Gaussian noise (sensor noise)."""
    arr = np.array(img, dtype=np.float64)
    noise = np.random.normal(0, std, arr.shape)
    arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_jpeg_artifacts(img, quality):
    """Apply JPEG compression artifacts."""
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def degrade_image(img, severity=None):
    """
    Apply a random combination of real-world degradations.
    Not all are applied every time — random selection mimics real variability.
    """
    if severity is None:
        severity = random.random()  # 0= mild, 1= heavy

    # 1. Perspective warp (50% chance, stronger with severity)
    if random.random() < 0.6:
        strength = PERSPECTIVE_STRENGTH * (0.3 + 0.7 * severity)
        img = apply_perspective_pil(img, strength)

    # 2. Lighting gradient (70% chance)
    if random.random() < 0.7:
        img = add_lighting_gradient(img)

    # 3. Vignette (50% chance)
    if random.random() < 0.5:
        img = add_vignette(img, 0.15 + 0.25 * severity)

    # 4. Slight rotation (30% chance)
    if random.random() < 0.3:
        angle = random.uniform(-2, 2)
        img = img.rotate(angle, expand=False, fillcolor='white')

    # 5. Gaussian blur (50% chance)
    if random.random() < 0.5:
        r = BLUR_RADIUS[0] + (BLUR_RADIUS[1] - BLUR_RADIUS[0]) * severity
        img = img.filter(ImageFilter.GaussianBlur(radius=r))

    # 6. Brightness/contrast variation (80% chance)
    if random.random() < 0.8:
        bf = BRIGHTNESS_RANGE[0] + (BRIGHTNESS_RANGE[1] - BRIGHTNESS_RANGE[0]) * random.random()
        img = ImageEnhance.Brightness(img).enhance(bf)
    if random.random() < 0.6:
        cf = CONTRAST_RANGE[0] + (CONTRAST_RANGE[1] - CONTRAST_RANGE[0]) * random.random()
        img = ImageEnhance.Contrast(img).enhance(cf)

    # 7. Gaussian noise (60% chance)
    if random.random() < 0.6:
        std = NOISE_STD[0] + (NOISE_STD[1] - NOISE_STD[0]) * severity
        img = add_gaussian_noise(img, round(std))

    # 8. JPEG compression (always, to simulate phone photo)
    quality = int(JPEG_QUALITY_RANGE[1] - (JPEG_QUALITY_RANGE[1] - JPEG_QUALITY_RANGE[0]) * severity)
    img = add_jpeg_artifacts(img, quality)

    return img


def main():
    # Load training data
    samples = []
    with open(TRAIN_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} training samples")

    # Select random subset for augmentation
    assert NUM_AUGMENT <= len(samples), f"NUM_AUGMENT ({NUM_AUGMENT}) > dataset size ({len(samples)})"
    selected = random.sample(samples, NUM_AUGMENT)

    # Generate augmented samples
    aug_samples = []
    for i, sample in enumerate(selected):
        img_rel = sample["images"][0]
        img_path = DATASET_DIR / img_rel.lstrip("./")

        if not img_path.exists():
            print(f"  SKIP: {img_path} not found")
            continue

        # Extract GT
        gt = ""
        for msg in sample.get("messages", []):
            if msg.get("role") == "assistant":
                gt = msg.get("content", "")
                break

        # Load image
        original = Image.open(img_path).convert("RGB")

        # Apply degradation with random severity
        severity = random.random()
        degraded = degrade_image(original, severity)

        # Save
        stem = img_path.stem
        aug_name = f"{stem}_deg{i:03d}.jpg"
        aug_path = AUG_IMG_DIR / aug_name
        degraded.save(aug_path, quality=90)
        rel_aug_path = f"./augmented/{aug_name}"

        # Build new sample with same GT
        aug_sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": rel_aug_path},
                        {"type": "text", "text": "Please identify all the components shown and their values in this circuit diagram picture. List it in the order of appearance with one component per line, using the format \"component_label\\nvalue\"."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": gt
                }
            ],
            "images": [rel_aug_path],
            "_meta": {"source": "degraded_aug", "original_image": os.path.basename(img_rel),
                       "severity": round(severity, 2)},
        }
        aug_samples.append(aug_sample)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{NUM_AUGMENT} done")

    print(f"Generated {len(aug_samples)} augmented samples")

    # Save degraded-only JSONL
    deg_jsonl = OUTPUT_DIR / "train_degraded_only.jsonl"
    with open(deg_jsonl, 'w', encoding='utf-8') as f:
        for s in aug_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f"Degraded JSONL: {deg_jsonl}")

    # Merge with original training set
    merged_jsonl = OUTPUT_DIR / "train_v9_pure_plus_degraded.jsonl"
    with open(merged_jsonl, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
        for s in aug_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f"Merged JSONL: {merged_jsonl} ({len(samples) + len(aug_samples)} samples = "
          f"{len(samples)} original + {len(aug_samples)} degraded)")

    print(f"""
========================================
  Training with degraded data:
    python scripts/train_llm_v14_degraded.py \\
      --train_jsonl {merged_jsonl}

  Or use only the degraded subset for a quick test:
    python scripts/train_llm_v14_degraded.py \\
      --train_jsonl {deg_jsonl}
========================================
""")


if __name__ == "__main__":
    main()
