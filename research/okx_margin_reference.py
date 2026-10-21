"""Fail-closed, offline OKX SWAP margin-tier reference validation.

User-provided or separately captured OKX tier snapshots are NOT historical
rates and must not be used to revise old Fishnet backtests or claim actual
liquidation prices. This module places NO requests or orders.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

OKX_INSTRUMENTS = {"BTC-USDT-SWAP", "ETH-USDT-SWAP"}
REQUIRED_MODE = "isolated"


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"invalid {name}")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid {name}") from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"invalid {name}")
    return result


def validate_okx_tier_snapshot(snapshot: dict) -> dict:
    """Validate an explicitly sourced snapshot; never invent position tiers.

    Snapshot schema is a normalized research interchange, *not* a claim that
    OKX's account API returns this exact shape.
    """
    if not isinstance(snapshot, dict):
        raise ValueError("OKX snapshot must be an object")
    if snapshot.get("exchange") != "OKX" or snapshot.get("instrument") not in OKX_INSTRUMENTS:
        raise ValueError("wrong exchange or unverified SWAP instrument")
    if snapshot.get("margin_mode") != REQUIRED_MODE:
        raise ValueError("only isolated margin is supported for research reference")
    if snapshot.get("source") != "OKX_official_tier_reference":
        raise ValueError("explicit official tier reference required")
    captured = snapshot.get("captured_at_utc")
    try:
        parsed = datetime.fromisoformat(captured.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        raise ValueError("missing capture time") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("capture time must be UTC")
    if not isinstance(snapshot.get("source_url"), str) or not snapshot["source_url"].startswith(
        ("https://www.okx.com/", "https://app.okx.com/")
    ):
        raise ValueError("official OKX source URL required")
    tiers = snapshot.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise ValueError("explicit nonempty OKX tier table required")
    previous_cap = 0.
    previous_mmr = 0.
    checked = []
    for i, tier in enumerate(tiers, start=1):
        if not isinstance(tier, dict) or tier.get("tier") != i:
            raise ValueError("nonsequential tier identifier")
        cap = _finite_positive(tier.get("max_notional_usdt"), "notional cap")
        mmr = _finite_positive(tier.get("maintenance_margin_rate"), "maintenance ratio")
        max_leverage = _finite_positive(tier.get("max_leverage"), "max leverage")
        if cap <= previous_cap or mmr >= 1 or mmr < previous_mmr or max_leverage > 1 / mmr:
            raise ValueError("inconsistent tier limits")
        checked.append({"tier": i, "max_notional_usdt": cap,
                        "maintenance_margin_rate": mmr, "max_leverage": max_leverage})
        previous_cap, previous_mmr = cap, mmr
    return {
        "exchange": "OKX",
        "instrument": snapshot["instrument"],
        "margin_mode": REQUIRED_MODE,
        "captured_at_utc": parsed.isoformat(),
        "tiers": checked,
        "position_tier_sourced": True,
        "verified_historical_at_fishnet_event_time": False,
        "exchange_liquidation_engine_simulated": False,
        "can_approve_live_execution": False,
    }


def compare_current_okx_reference(snapshot: dict, *, notional_usdt: float,
                                  leverage: float) -> dict:
    """Current-reference tier comparison only; never alter historical fills."""
    validated = validate_okx_tier_snapshot(snapshot)
    notional = _finite_positive(notional_usdt, "position notional")
    lev = _finite_positive(leverage, "leverage")
    tier = next((t for t in validated["tiers"]
                 if notional <= t["max_notional_usdt"]), None)
    if tier is None:
        raise ValueError("position outside sourced tier coverage")
    return {
        "instrument": validated["instrument"],
        "selected_reference_tier": tier["tier"],
        "reference_max_leverage": tier["max_leverage"],
        "reference_maintenance_margin_rate": tier["maintenance_margin_rate"],
        "hypothetical_leverage_above_current_tier_limit": lev > tier["max_leverage"],
        "historical_fishnet_replay_unchanged": True,
        "historical_liquidation_result": None,
        "release_approved": False,
    }
