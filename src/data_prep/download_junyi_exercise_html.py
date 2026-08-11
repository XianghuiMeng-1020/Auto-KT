"""
download_junyi_exercise_html.py
================================
Download all exercise HTML files from junyiacademy/junyiexercise that match
interacted exercises in junyi_ProblemLog_original.csv.

Sources
-------
- Exercise list:   https://api.github.com/repos/junyiacademy/junyiexercise/contents/exercises
- Raw HTML files:  https://raw.githubusercontent.com/junyiacademy/junyiexercise/master/exercises/{slug}.html
- Licence:         Exercises: CC BY-NC-SA 3.0; Framework: MIT

Usage
-----
    python scripts/data/download_junyi_exercise_html.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
import urllib.request
import urllib.error
import sys

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]
HTML_DIR = ROOT / "data_raw" / "junyi" / "exercises_html"
LOG_CSV   = ROOT / "data_raw" / "junyi" / "extracted" / "junyi_ProblemLog_original.csv"
MANIFEST  = ROOT / "data_raw" / "junyi" / "junyi_exercise_html_manifest.json"

GITHUB_CONTENTS_URL = (
    "https://api.github.com/repos/junyiacademy/junyiexercise/contents/exercises"
)
RAW_BASE = "https://raw.githubusercontent.com/junyiacademy/junyiexercise/master/exercises"

HEADERS = {"User-Agent": "AutoKT-Research/1.0 (non-commercial academic)"}
REQUEST_DELAY = 0.3  # seconds between requests


def fetch_json(url: str) -> list | dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def get_interacted_slugs() -> set[str]:
    """Return the set of exercise slugs that appear in the problem log."""
    import csv

    slugs: set[str] = set()
    with open(LOG_CSV, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row.get("exercise", "").strip()
            if v:
                slugs.add(v)
    return slugs


def list_html_slugs() -> dict[str, str]:
    """Return {slug: download_url} for every HTML file in the GitHub repo."""
    files = fetch_json(GITHUB_CONTENTS_URL)
    return {
        f["name"].replace(".html", ""): f["download_url"]
        for f in files
        if isinstance(f, dict) and f.get("name", "").endswith(".html")
    }


def download_html(slug: str, url: str, dest: pathlib.Path) -> str | None:
    """Download one HTML file; return SHA-256 or None on error."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        dest.write_bytes(data)
        return hashlib.sha256(data).hexdigest()
    except urllib.error.HTTPError as exc:
        print(f"  WARN: HTTP {exc.code} for {slug}", file=sys.stderr)
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: {exc} for {slug}", file=sys.stderr)
        return None


def main() -> None:
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching interacted exercise slugs from problem log…")
    interacted = get_interacted_slugs()
    print(f"  Unique exercise slugs in log: {len(interacted)}")

    print("Fetching HTML file list from GitHub repo…")
    html_slugs = list_html_slugs()
    print(f"  HTML files available: {len(html_slugs)}")

    target = {s: u for s, u in html_slugs.items() if s in interacted}
    extra  = {s: u for s, u in html_slugs.items() if s not in interacted}
    missing = interacted - set(html_slugs)

    print(f"  Matching (interacted ∩ HTML repo): {len(target)}")
    print(f"  In log but NOT in repo:            {len(missing)}")
    print(f"  Coverage:                          {len(target)/len(interacted)*100:.1f}%")

    results: dict[str, dict] = {}
    n_ok = n_skip = n_fail = 0

    all_targets = {**target, **extra}   # download everything for completeness
    print(f"\nDownloading {len(all_targets)} HTML files (interacted + extras)…")

    for i, (slug, url) in enumerate(sorted(all_targets.items()), 1):
        dest = HTML_DIR / f"{slug}.html"
        if dest.exists():
            results[slug] = {
                "slug": slug,
                "url": url,
                "status": "cached",
                "sha256": sha256(dest),
                "interacted": slug in interacted,
            }
            n_skip += 1
            if i % 100 == 0:
                print(f"  [{i}/{len(all_targets)}] cached {slug}")
            continue

        checksum = download_html(slug, url, dest)
        time.sleep(REQUEST_DELAY)

        if checksum:
            results[slug] = {
                "slug": slug,
                "url": url,
                "status": "ok",
                "sha256": checksum,
                "interacted": slug in interacted,
            }
            n_ok += 1
        else:
            results[slug] = {
                "slug": slug,
                "url": url,
                "status": "error",
                "sha256": None,
                "interacted": slug in interacted,
            }
            n_fail += 1

        if i % 50 == 0:
            print(f"  [{i}/{len(all_targets)}] …")

    # Save manifest
    manifest = {
        "source_repo": "https://github.com/junyiacademy/junyiexercise",
        "raw_base": RAW_BASE,
        "licence": "Exercises: CC BY-NC-SA 3.0 (Khan Academy / Junyi Academy)",
        "retrieval_date_utc": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%d"),
        "total_html_in_repo": len(html_slugs),
        "interacted_exercises": len(interacted),
        "exercises_with_html": len(target),
        "exercises_missing_html": len(missing),
        "coverage_pct": round(len(target) / len(interacted) * 100, 2),
        "downloaded_ok": n_ok,
        "already_cached": n_skip,
        "download_errors": n_fail,
        "missing_slugs": sorted(missing),
        "files": results,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"\nDone. Downloaded={n_ok}, cached={n_skip}, errors={n_fail}. "
        f"Coverage={manifest['coverage_pct']}%"
    )
    print(f"Manifest saved → {MANIFEST}")


if __name__ == "__main__":
    main()
