#!/usr/bin/env python3
"""
Generate a deterministic cafe visit log for a loyalty-program segmentation lesson.

Business context: a cafe manager needs customer-level RFM segments for retention,
offer design, and menu strategy. Archetype labels are withheld from the transaction
file and used only after clustering as simulation ground truth.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

TRANSACTIONS_PATH = config.DATA_DIR / "cafe_transactions_example.csv"
GROUND_TRUTH_PATH = config.DATA_DIR / "cafe_customer_archetypes_validation.csv"


def _visit_datetimes(
    rng: np.random.Generator,
    visit_count: int,
    last_recency_days: tuple[int, int],
    force_latest_visit: bool = False,
) -> list[pd.Timestamp]:
    """Return unique visit timestamps ending inside the archetype's recency range."""
    start = pd.Timestamp(config.OBSERVATION_START)
    reference = pd.Timestamp(config.GENERATOR_REFERENCE_DATE)
    if force_latest_visit:
        recency_days = last_recency_days[0]
    else:
        recency_days = int(rng.integers(last_recency_days[0], last_recency_days[1] + 1))

    last_day = reference - pd.Timedelta(days=recency_days)
    last_visit = last_day + pd.Timedelta(
        hours=int(rng.integers(7, 19)), minutes=int(rng.integers(0, 60))
    )
    available_seconds = int((last_visit - start).total_seconds())
    prior_offsets = rng.choice(available_seconds, size=visit_count - 1, replace=False)
    visits = [start + pd.Timedelta(seconds=int(value)) for value in prior_offsets]
    visits.append(last_visit)
    return sorted(visits)


def _basket(
    rng: np.random.Generator,
    size_range: tuple[int, int],
    item_weights: tuple[float, ...],
) -> tuple[str, float]:
    """Sample a non-empty basket and calculate its price from the fixed menu."""
    menu_items = list(config.MENU_PRICES)
    basket_size = int(rng.integers(size_range[0], size_range[1] + 1))
    items = rng.choice(
        menu_items,
        size=basket_size,
        replace=False,
        p=np.asarray(item_weights, dtype=float),
    ).tolist()
    spend = round(sum(config.MENU_PRICES[item] for item in items), 2)
    return " | ".join(items), spend


def generate_transactions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate transactions and a separate customer-to-archetype validation table."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    transaction_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, str]] = []
    customer_number = 1

    for archetype, spec in config.ARCHETYPES.items():
        for archetype_index in range(spec["customer_count"]):
            customer_id = f"{config.CUSTOMER_PREFIX}-{customer_number:03d}"
            visit_count = int(rng.integers(spec["visit_range"][0], spec["visit_range"][1] + 1))
            visits = _visit_datetimes(
                rng,
                visit_count,
                spec["last_recency_days"],
                force_latest_visit=archetype == "VIP Daily Drinkers" and archetype_index == 0,
            )

            for visit_datetime in visits:
                items, total_spend = _basket(rng, spec["basket_size"], spec["item_weights"])
                transaction_rows.append(
                    {
                        "customer_id": customer_id,
                        "visit_datetime": visit_datetime,
                        "items": items,
                        "total_spend": total_spend,
                    }
                )

            truth_rows.append({"customer_id": customer_id, "archetype": archetype})
            customer_number += 1

    transactions = pd.DataFrame(transaction_rows).sort_values(
        ["visit_datetime", "customer_id"], ignore_index=True
    )
    transactions.insert(
        0,
        "transaction_id",
        [f"TX-{number:06d}" for number in range(1, len(transactions) + 1)],
    )
    transactions["visit_datetime"] = pd.to_datetime(transactions["visit_datetime"])
    ground_truth = pd.DataFrame(truth_rows).sort_values("customer_id", ignore_index=True)
    return transactions, ground_truth


def validate_transactions(transactions: pd.DataFrame, ground_truth: pd.DataFrame) -> None:
    """Fail loudly when generated data violates the locked simulation contract."""
    expected_columns = {
        "transaction_id",
        "customer_id",
        "visit_datetime",
        "items",
        "total_spend",
    }
    assert set(transactions.columns) == expected_columns
    assert len(ground_truth) == config.EXPECTED_CUSTOMERS == 150
    assert transactions["customer_id"].nunique() == config.EXPECTED_CUSTOMERS
    assert transactions["transaction_id"].is_unique
    assert not transactions.isna().any().any()
    assert not ground_truth.isna().any().any()
    assert set(ground_truth["archetype"]) == set(config.ARCHETYPES)
    assert set(transactions["customer_id"]) == set(ground_truth["customer_id"])
    assert transactions["total_spend"].gt(0).all()

    calculated_spend = transactions["items"].map(
        lambda basket: sum(config.MENU_PRICES[item] for item in basket.split(" | "))
    )
    assert np.allclose(calculated_spend, transactions["total_spend"])

    visits_per_customer = transactions.groupby("customer_id").size()
    latest_visit = transactions.groupby("customer_id")["visit_datetime"].max().dt.normalize()
    generator_reference = pd.Timestamp(config.GENERATOR_REFERENCE_DATE)
    recency = (generator_reference - latest_visit).dt.days

    truth_indexed = ground_truth.set_index("customer_id")
    for archetype, spec in config.ARCHETYPES.items():
        customer_ids = truth_indexed.index[truth_indexed["archetype"] == archetype]
        assert visits_per_customer.loc[customer_ids].between(*spec["visit_range"]).all()
        assert recency.loc[customer_ids].between(*spec["last_recency_days"]).all()


def write_outputs(transactions: pd.DataFrame, ground_truth: pd.DataFrame) -> None:
    """Write deterministic CSV outputs next to the case-study scripts."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_transactions = transactions.copy()
    output_transactions["visit_datetime"] = output_transactions["visit_datetime"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    output_transactions.to_csv(TRANSACTIONS_PATH, index=False)
    ground_truth.to_csv(GROUND_TRUTH_PATH, index=False)


def main() -> None:
    LOGGER.info("Generating cafe transactions with RANDOM_SEED=%d", config.RANDOM_SEED)
    transactions, ground_truth = generate_transactions()
    validate_transactions(transactions, ground_truth)
    write_outputs(transactions, ground_truth)
    LOGGER.info("Validation passed: %d customers, %d transactions, 0 nulls", config.EXPECTED_CUSTOMERS, len(transactions))
    LOGGER.info("Transactions: %s", TRANSACTIONS_PATH)
    LOGGER.info("Withheld validation labels: %s", GROUND_TRUTH_PATH)


if __name__ == "__main__":
    main()
