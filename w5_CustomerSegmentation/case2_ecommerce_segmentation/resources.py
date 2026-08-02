#!/usr/bin/env python3
"""DuckDB configuration and resource watchdog shared by Case Study 2 stages."""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import duckdb
import psutil

import config


LOGGER = logging.getLogger(__name__)
GIB = 1024**3


class ResourceLimitExceeded(RuntimeError):
    """Raised after the watchdog interrupts a query at a hard resource limit."""


def directory_size(path: Path) -> int:
    """Return recursive file size without following directory symlinks."""
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def configured_connection() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection with the locked laptop-safe limits."""
    config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(database=":memory:")
    escaped_temp = str(config.TEMP_DIR).replace("'", "''")
    connection.execute(f"SET memory_limit = '{config.DUCKDB_MEMORY_LIMIT}'")
    connection.execute(f"SET threads = {config.DUCKDB_THREADS}")
    connection.execute(f"SET temp_directory = '{escaped_temp}'")
    connection.execute(f"SET max_temp_directory_size = '{config.DUCKDB_TEMP_LIMIT}'")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET enable_progress_bar = true")
    return connection


@dataclass
class ResourceMetrics:
    peak_rss_gb: float = 0.0
    minimum_free_disk_gb: float = float("inf")
    peak_temp_gb: float = 0.0
    elapsed_seconds: float = 0.0


class ResourceWatchdog:
    """Monitor process RAM, system headroom, free disk, and DuckDB spill usage."""

    def __init__(self, connection: duckdb.DuckDBPyConnection, stage: str):
        self.connection = connection
        self.stage = stage
        self.metrics = ResourceMetrics()
        self.violation: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._warned: set[str] = set()

    def __enter__(self) -> "ResourceWatchdog":
        self._started = monotonic()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=config.WATCHDOG_INTERVAL_SECONDS + 1)
        self.metrics.elapsed_seconds = monotonic() - self._started
        if self.metrics.minimum_free_disk_gb == float("inf"):
            self.metrics.peak_rss_gb = psutil.Process().memory_info().rss / GIB
            self.metrics.minimum_free_disk_gb = shutil.disk_usage(config.BASE_DIR).free / GIB
            self.metrics.peak_temp_gb = directory_size(config.TEMP_DIR) / GIB
        if self.violation and exc_type is None:
            raise ResourceLimitExceeded(self.violation)
        return False

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            LOGGER.warning(message)
            self._warned.add(key)

    def raise_if_violated(self) -> None:
        """Allow non-DuckDB batch loops to stop after the monitor detects a limit."""
        if self.violation:
            raise ResourceLimitExceeded(self.violation)

    def _interrupt(self, message: str) -> None:
        self.violation = f"{self.stage}: {message}"
        LOGGER.error(self.violation)
        self.connection.interrupt()
        self._stop.set()

    def _monitor(self) -> None:
        process = psutil.Process()
        while not self._stop.wait(config.WATCHDOG_INTERVAL_SECONDS):
            rss_gb = process.memory_info().rss / GIB
            available_gb = psutil.virtual_memory().available / GIB
            free_disk_gb = shutil.disk_usage(config.BASE_DIR).free / GIB
            temp_gb = directory_size(config.TEMP_DIR) / GIB
            self.metrics.peak_rss_gb = max(self.metrics.peak_rss_gb, rss_gb)
            self.metrics.minimum_free_disk_gb = min(
                self.metrics.minimum_free_disk_gb, free_disk_gb
            )
            self.metrics.peak_temp_gb = max(self.metrics.peak_temp_gb, temp_gb)

            if rss_gb >= config.RSS_ABORT_GB:
                self._interrupt(f"RSS {rss_gb:.1f} GiB reached the {config.RSS_ABORT_GB:.0f} GiB limit")
                return
            if available_gb <= config.SYSTEM_AVAILABLE_ABORT_GB:
                self._interrupt(f"system available RAM fell to {available_gb:.1f} GiB")
                return
            if free_disk_gb <= config.DISK_ABORT_GB:
                self._interrupt(f"free disk fell to {free_disk_gb:.1f} GiB")
                return
            if temp_gb >= 35.0:
                self._interrupt(f"DuckDB temp usage reached {temp_gb:.1f} GiB")
                return

            if rss_gb >= config.RSS_WARNING_GB:
                self._warn_once("rss", f"Process RSS warning: {rss_gb:.1f} GiB")
            if available_gb <= config.SYSTEM_AVAILABLE_WARNING_GB:
                self._warn_once("ram", f"System available RAM warning: {available_gb:.1f} GiB")
            if free_disk_gb <= config.DISK_WARNING_GB:
                self._warn_once("disk", f"Free disk warning: {free_disk_gb:.1f} GiB")


def preflight_snapshot() -> dict[str, float | int | str]:
    """Return and validate the current machine resource envelope."""
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(config.BASE_DIR)
    snapshot = {
        "physical_ram_gb": round(memory.total / GIB, 2),
        "available_ram_gb": round(memory.available / GIB, 2),
        "free_disk_gb": round(disk.free / GIB, 2),
        "logical_cpus": psutil.cpu_count(logical=True) or 1,
        "duckdb_memory_limit": config.DUCKDB_MEMORY_LIMIT,
        "duckdb_threads": config.DUCKDB_THREADS,
        "duckdb_temp_limit": config.DUCKDB_TEMP_LIMIT,
    }
    if disk.free / GIB <= config.DISK_WARNING_GB:
        raise ResourceLimitExceeded(
            f"Preflight requires more than {config.DISK_WARNING_GB:.0f} GiB free disk."
        )
    if memory.available / GIB <= config.SYSTEM_AVAILABLE_WARNING_GB:
        raise ResourceLimitExceeded(
            f"Preflight requires more than {config.SYSTEM_AVAILABLE_WARNING_GB:.0f} GiB available RAM."
        )
    return snapshot
