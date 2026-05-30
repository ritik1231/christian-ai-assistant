"""Download public-domain KJV Bible JSON and save to data/kjv.json."""
import json
import urllib.request
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KJV_URL = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"
OUT_PATH = DATA_DIR / "kjv.json"

def download():
    DATA_DIR.mkdir(exist_ok=True)
    if OUT_PATH.exists():
        print(f"KJV already present at {OUT_PATH} — skipping download.")
        return

    print(f"Downloading KJV from {KJV_URL} ...")
    import urllib.request as req
    raw = req.urlopen(KJV_URL).read().decode("utf-8-sig")
    data = json.loads(raw)
    # Normalise and save
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Downloaded {len(data)} books.")

if __name__ == "__main__":
    download()
