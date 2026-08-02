#!/usr/bin/env python3
"""Generate a small schema-compatible ecommerce event file for pipeline testing only."""

from __future__ import annotations

import argparse
import gzip
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

import config


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

CATEGORIES = (
    ("electronics.smartphone", "phone", 420.0),
    ("electronics.audio.headphone", "audio", 85.0),
    ("appliances.kitchen.kettle", "kitchen", 42.0),
    ("computers.notebook", "computer", 780.0),
    ("apparel.shoes", "shoes", 72.0),
)


def generate_smoke_events(session_count: int) -> pl.DataFrame:
    """Create ordered view/cart/purchase sessions with deterministic conversion behavior."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows: list[dict[str, object]] = []
    start = datetime(2019, 10, 1, tzinfo=timezone.utc)

    for session_number in range(1, session_count + 1):
        user_id = 500_000_000 + int(rng.integers(1, max(500, session_count // 2)))
        session_id = f"SMOKE-{session_number:08d}"
        # Reserve the final 30 minutes so later events cannot spill into November.
        session_start = start + timedelta(
            seconds=int(rng.integers(0, 31 * 24 * 3600 - 30 * 60))
        )
        category_code, brand, base_price = CATEGORIES[int(rng.integers(0, len(CATEGORIES)))]
        product_id = 1_000_000 + int(rng.integers(1, 50_000))
        category_id = 2_053_013_500_000_000_000 + int(rng.integers(1, 10_000))
        price = round(base_price * float(rng.uniform(0.75, 1.25)), 2)
        view_count = int(rng.integers(1, 5))
        converts_to_cart = rng.random() < 0.38
        converts_to_purchase = converts_to_cart and rng.random() < 0.42

        event_types = ["view"] * view_count
        if converts_to_cart:
            event_types.append("cart")
        if converts_to_purchase:
            event_types.append("purchase")
        elif converts_to_cart and rng.random() < 0.15:
            event_types.append("remove_from_cart")

        for event_index, event_type in enumerate(event_types):
            event_time = session_start + timedelta(minutes=event_index * int(rng.integers(1, 5)))
            rows.append(
                {
                    "event_time": event_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "event_type": event_type,
                    "product_id": product_id,
                    "category_id": category_id,
                    "category_code": category_code if rng.random() > 0.05 else None,
                    "brand": brand if rng.random() > 0.03 else None,
                    "price": price,
                    "user_id": user_id,
                    "user_session": session_id,
                }
            )

    events = pl.DataFrame(rows).sort("event_time")
    assert list(events.columns) == list(config.RAW_COLUMNS)
    assert set(events["event_type"].to_list()).issubset(config.ALLOWED_EVENT_TYPES)
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=20_000)
    args = parser.parse_args()
    if args.sessions < 100:
        parser.error("--sessions must be at least 100")

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RAW_DIR / "smoke-2019-Oct.csv.gz"
    events = generate_smoke_events(args.sessions)
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(events.write_csv())
    LOGGER.info("Wrote %d smoke events across %d sessions to %s", len(events), args.sessions, output_path)
    LOGGER.warning("Smoke data is synthetic validation input and must not appear in the final report.")


if __name__ == "__main__":
    main()
