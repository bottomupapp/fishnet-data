"""Offline validation of captured OKX public SWAP position-tier responses.

GET /api/v5/public/position-tiers?instType=SWAP&tdMode=isolated&instFamily=BTC-USDT
(and ETH-USDT). minSz/maxSz are CONTRACT COUNTS, not USD notionals.
Never convert contract tiers to USDT without separately verified ctVal,
ctValCcy and contemporaneous mark price. No API calls or order placement.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from urllib.parse import urlsplit

VALID_FAMILIES = {"BTC-USDT", "ETH-USDT"}
SOURCE_ENDPOINT = "/api/v5/public/position-tiers"


def _number(value, field, *, allow_zero=False):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing OKX {field}")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ValueError(f"invalid OKX {field}") from None
    if not number.is_finite() or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"invalid OKX {field}")
    return number


def parse_public_swap_tiers(response: dict, *, inst_family: str,
                            captured_at_utc: str, source_url: str) -> dict:
    """Validate a captured public endpoint response, never certify history.

    This deliberately returns ranges in CONTRACTS. For SWAP, even the
    instrument family's price currency cannot substitute for contract size.
    """
    if inst_family not in VALID_FAMILIES:
        raise ValueError("unapproved OKX instrument family")
    if not isinstance(source_url, str):
        raise ValueError("official OKX public endpoint required")
    url = urlsplit(source_url)
    if (url.scheme != "https" or url.hostname not in ("www.okx.com", "app.okx.com")
        or url.port is not None or url.path != SOURCE_ENDPOINT
        or url.username or url.password):
        raise ValueError("official OKX public endpoint required")
    try:
        from urllib.parse import parse_qs
        params = parse_qs(url.query, strict_parsing=True)
    except ValueError:
        raise ValueError("invalid OKX endpoint parameters") from None
    if (params != {"instType": ["SWAP"], "tdMode": ["isolated"],
                   "instFamily": [inst_family]}):
        raise ValueError("unexpected OKX instrument or trade mode")
    try:
        t = datetime.fromisoformat(captured_at_utc.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        raise ValueError("invalid capture timestamp") from None
    if t.tzinfo is None or t.utcoffset().total_seconds() != 0:
        raise ValueError("capture timestamp must be UTC")
    if not isinstance(response, dict) or response.get("code") != "0":
        raise ValueError("OKX public tiers response not successful")
    data = response.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("missing OKX tiers")
    accepted = []
    last_max = Decimal(0)
    last_mmr = Decimal(0)
    for number, raw in enumerate(data, 1):
        if not isinstance(raw, dict) or raw.get("instFamily") != inst_family:
            raise ValueError("foreign OKX family or malformed tier")
        if raw.get("tier") != str(number):
            raise ValueError("out-of-order OKX tier")
        low = _number(raw.get("minSz"), "minSz", allow_zero=True)
        high = _number(raw.get("maxSz"), "maxSz")
        mmr = _number(raw.get("mmr"), "mmr")
        imr = _number(raw.get("imr"), "imr")
        leverage = _number(raw.get("maxLever"), "maxLever")
        if (low < last_max or low > high or high <= last_max
            or mmr >= 1 or imr >= 1 or imr < mmr
            or mmr < last_mmr or leverage > (1 / imr)):
            raise ValueError("inconsistent OKX contract tier bounds")
        accepted.append({
            "tier": number,
            "min_contracts": str(low),
            "max_contracts": str(high),
            "maintenance_margin_rate": str(mmr),
            "initial_margin_rate": str(imr),
            "max_leverage": str(leverage),
        })
        last_max, last_mmr = high, mmr
    return {
        "exchange": "OKX",
        "inst_family": inst_family,
        "margin_mode": "isolated",
        "captured_at_utc": t.isoformat(),
        "source_url": source_url,
        "tiers": accepted,
        "tier_size_unit": "contracts_NOT_USDT",
        "usdt_notional_comparison_available": False,
        "verified_at_historical_fishnet_event_time": False,
        "actual_exchange_liquidation_engine_simulated": False,
        "release_approved": False,
    }
