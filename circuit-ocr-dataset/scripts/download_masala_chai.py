#!/usr/bin/env python3
"""
Masala-CHAI Downloader & Extractor
==================================
This script downloads the Masala-CHAI dataset repository as a zip archive
using multiple mirrors/fallbacks (to ensure high success rate in China),
and extracts it to ./data/raw/masala_chai/.

All paths are relative.
"""

import os
import io
import zipfile
import logging
import requests
import urllib3
from pathlib import Path

# Suppress SSL warnings for mirror sites with mismatched certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("download_masala_chai")

def download_and_extract(output_dir: str):
    urls = [
        # Fallback 1: kkgithub zip mirror
        "https://kkgithub.com/jitendra-bhandari/Masala-CHAI/archive/refs/heads/main.zip",
        # Fallback 2: gitclone zip mirror
        "https://gitclone.com/github.com/jitendra-bhandari/Masala-CHAI/archive/refs/heads/main.zip",
        # Fallback 3: Direct github zip link
        "https://github.com/jitendra-bhandari/Masala-CHAI/archive/refs/heads/main.zip"
    ]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    success = False
    for url in urls:
        logger.info(f"Attempting to download Masala-CHAI zip from: {url}")
        try:
            # We disable SSL verification because mirror sites often have certificate mismatch issues
            response = requests.get(url, verify=False, timeout=30)
            if response.status_code == 200 and len(response.content) > 10000:
                logger.info(f"Successfully downloaded zip archive ({len(response.content) / (1024*1024):.2f} MB)")
                
                # Extract zip file
                logger.info("Extracting zip archive...")
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    # Find root folder in zip (typically 'Masala-CHAI-main')
                    root_name = z.namelist()[0].split('/')[0]
                    
                    # Extract files
                    for member in z.infolist():
                        if member.is_dir():
                            continue
                        # Remove the root folder name from extraction path
                        rel_path = Path(member.filename).relative_to(root_name)
                        dest_file = out_path / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(member) as source, open(dest_file, "wb") as target:
                            target.write(source.read())
                            
                logger.info(f"Successfully extracted Masala-CHAI to {output_dir}")
                success = True
                break
            else:
                logger.warning(f"Download failed with status {response.status_code} or small file size ({len(response.content)} bytes)")
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            continue

    if not success:
        logger.error("All Masala-CHAI download attempts failed.")
        raise RuntimeError("Failed to download Masala-CHAI dataset.")

if __name__ == "__main__":
    download_and_extract("./data/raw/masala_chai")
