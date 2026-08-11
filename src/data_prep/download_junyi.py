#!/usr/bin/env python3
"""Download Junyi Academy Math Practicing Log without login or API token."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Official public mirror used by EduData (bigdata-ustc/EduData), matching CMU DataShop release.
PRIMARY_URL = "http://base.ustc.edu.cn/data/JunyiAcademy_Math_Practicing_Log/junyi.rar"
EDUDATA_REPO = "https://github.com/bigdata-ustc/EduData"
CMU_DATASHOP_INFO = "https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198"
JUNYI_ACADEMY = "http://www.junyiacademy.org/"
DEFAULT_OUT = Path("data_raw/junyi")
CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download_resumable(url: str, dest: Path) -> tuple[int, list[str]]:
    """Return (final_size, redirect_chain)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    redirects: list[str] = []
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"User-Agent": "AutoKT-Research/1.0"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=120) as resp:
        if resp.url:
            redirects.append(resp.url)
        mode = "ab" if existing > 0 and resp.status == 206 else "wb"
        if mode == "wb":
            existing = 0
        with dest.open(mode) as out:
            while chunk := resp.read(CHUNK):
                out.write(chunk)
    return dest.stat().st_size, redirects


def extract_rar(archive: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # macOS bsdtar supports many formats including RAR.
    subprocess.run(
        ["bsdtar", "-xf", str(archive), "-C", str(out_dir)],
        check=True,
        capture_output=True,
    )
    return sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / "junyi.rar"
    extracted = out_dir / "extracted"
    retrieved_at = datetime.now(timezone.utc).isoformat()

    redirects: list[str] = []
    try:
        size, redirects = download_resumable(PRIMARY_URL, archive)
    except (HTTPError, URLError, TimeoutError) as e:
        manifest = {
            "dataset": "junyi_academy",
            "download_status": "FAILED",
            "error": str(e),
            "source_url": PRIMARY_URL,
            "retrieved_at_utc": retrieved_at,
        }
        (out_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Download failed: {e}")
        return 2

    archive_sha = sha256_file(archive)
    extracted_files: list[str] = []
    extract_error = None
    if not any(extracted.iterdir()) if extracted.exists() else True:
        try:
            extracted_files = extract_rar(archive, extracted)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            extract_error = str(e)
            if hasattr(e, "stderr") and e.stderr:
                extract_error += " " + e.stderr.decode("utf-8", "replace")[:500]

    manifest = {
        "dataset": "junyi_academy_math_practicing_log",
        "official_cmu_datashop": CMU_DATASHOP_INFO,
        "official_junyi_academy": JUNYI_ACADEMY,
        "edudata_reference": EDUDATA_REPO,
        "source_url": PRIMARY_URL,
        "final_url": redirects[-1] if redirects else PRIMARY_URL,
        "redirects": redirects,
        "retrieved_at_utc": retrieved_at,
        "archive_path": str(archive),
        "archive_size_bytes": size,
        "archive_sha256": archive_sha,
        "extracted_dir": str(extracted),
        "extracted_file_count": len(extracted_files),
        "extracted_filenames": extracted_files[:200],
        "extract_error": extract_error,
        "licence_note": (
            "Junyi Academy Math Practicing Log (to Jan 2015) distributed via CMU DataShop / "
            "EduData USTC mirror for research use; confirm terms on DataShop dataset page."
        ),
        "token_required": False,
        "login_required": False,
    }
    manifest_path = out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote {manifest_path}")
    print(f"SHA-256: {archive_sha}")
    print(f"Size: {size:,} bytes")
    if extract_error:
        print(f"Extract error: {extract_error}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
