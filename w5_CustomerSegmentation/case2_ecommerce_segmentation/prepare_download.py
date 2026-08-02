#!/usr/bin/env python3
"""Stream a Kaggle ZIP member into gzip without writing the multi-gigabyte plain CSV."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import zipfile
from pathlib import Path
from time import monotonic

import config
from resources import ResourceWatchdog, configured_connection, preflight_snapshot


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)
CHUNK_SIZE = 16 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_zip_to_gzip(
    archive_path: Path,
    member_name: str,
    output_path: Path,
) -> dict[str, object]:
    """Convert one verified ZIP member and atomically publish the gzip output."""
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    started = monotonic()
    uncompressed_digest = hashlib.sha256()
    uncompressed_bytes = 0
    next_progress = 512 * 1024**2
    connection = configured_connection()
    watchdog = ResourceWatchdog(connection, stage=f"prepare {member_name}")

    try:
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo(member_name)
            with watchdog:
                with archive.open(info, "r") as source, gzip.GzipFile(
                    filename=str(partial_path), mode="wb", compresslevel=1, mtime=0
                ) as destination:
                    while chunk := source.read(CHUNK_SIZE):
                        watchdog.raise_if_violated()
                        destination.write(chunk)
                        uncompressed_digest.update(chunk)
                        uncompressed_bytes += len(chunk)
                        if uncompressed_bytes >= next_progress:
                            LOGGER.info(
                                "Streamed %.1f / %.1f GiB",
                                uncompressed_bytes / 1024**3,
                                info.file_size / 1024**3,
                            )
                            next_progress += 512 * 1024**2
            assert uncompressed_bytes == info.file_size, "Uncompressed byte count mismatch."
            # ZipExtFile validates the member CRC when the stream reaches EOF.
            partial_path.replace(output_path)
    finally:
        connection.close()

    manifest: dict[str, object] = {
        "dataset_url": config.DATASET_URL,
        "archive_path": str(archive_path.resolve()),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "member_name": member_name,
        "member_uncompressed_bytes": uncompressed_bytes,
        "member_sha256": uncompressed_digest.hexdigest(),
        "gzip_path": str(output_path.resolve()),
        "gzip_bytes": output_path.stat().st_size,
        "gzip_sha256": sha256_file(output_path),
        "elapsed_seconds": round(monotonic() - started, 3),
        "resource_metrics": {
            "peak_rss_gb": round(watchdog.metrics.peak_rss_gb, 3),
            "minimum_free_disk_gb": round(watchdog.metrics.minimum_free_disk_gb, 3),
            "peak_temp_gb": round(watchdog.metrics.peak_temp_gb, 3),
        },
        "machine_preflight": preflight_snapshot(),
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=config.RAW_DIR / "2019-Oct.csv.zip"
    )
    parser.add_argument("--member", default="2019-Oct.csv")
    parser.add_argument(
        "--output", type=Path, default=config.RAW_DIR / "2019-Oct.csv.gz"
    )
    args = parser.parse_args()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Preflight: %s", json.dumps(preflight_snapshot(), sort_keys=True))
    LOGGER.info("Streaming %s from %s", args.member, args.archive)
    manifest = convert_zip_to_gzip(
        args.archive.resolve(), args.member, args.output.resolve()
    )
    output_stem = args.output.name.removesuffix(".gz").removesuffix(".csv")
    manifest_path = config.OUTPUT_DIR / f"{output_stem}_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "Prepared %.2f GiB gzip from %.2f GiB CSV in %.1f seconds",
        manifest["gzip_bytes"] / 1024**3,
        manifest["member_uncompressed_bytes"] / 1024**3,
        manifest["elapsed_seconds"],
    )
    LOGGER.info("Source manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
