"""Fishnet research: position sizing from a fixed equity-risk budget.

Leverage affects margin and exposure, NOT the chosen per-trade stop budget.
This is NOT exchange liquidation/margin modelling and never places orders.
It runs on historical simulated gross fills; trading fees/slippage are charged
once on the notional. Historical funding is OPT-IN and never assumed present. Stops can gap, so 1% PLANNED risk is not a loss guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from research.execution import net_after_costs, net_after_costs_short
from research.portfolio import NET_30D_TARGET


@dataclass(frozen=True)
class RiskFill:
    symbol: str
    entry_at: str
    exit_at: str
    gross_return: float
    stop_fraction: float
    side: str = 'long'
    funding_fraction: float = 0.0
    funding_reconciled: bool = False


def risk_portfolio_audit(
    fills: list[RiskFill],
    *,
    starting_capital: float = 10000.0,
    risk_fraction: float = 0.01,
    max_leverage: float = 20.0,
    max_gross_exposure: float = 6.0,
    max_concurrent_positions: int = 2,
    fee_bps_per_side: float = 6.0,
    slippage_bps_per_side: float = 3.0,
    maintenance_margin_fraction: float = .005,
    liquidation_safety_buffer_fraction: float = .005,
    require_funding_reconciled: bool = False,
) -> dict:
    """Risk-budget sizing with hard caps on margin, gross exposure and overlap.

    The planned stop loss includes roundtrip trading costs, so a complete
    stop (with no adverse gap) targets at most risk_fraction of equity.

    When require_funding_reconciled=True, each event must have archived
    settlement cashflows attached. Funding losses may exceed planned stop risk.

    Effective leverage is capped by a simplistic liquidation-distance proxy;
    this proxy cannot certify actual Binance/OKX liquidation safety.
    """
    settings = (starting_capital, risk_fraction, max_leverage, max_gross_exposure,
                fee_bps_per_side, slippage_bps_per_side,
                maintenance_margin_fraction, liquidation_safety_buffer_fraction)
    if not all(math.isfinite(v) for v in settings):
        raise ValueError("non-finite risk settings")
    if (starting_capital <= 0 or not 0 < risk_fraction <= .1 or max_leverage < 1 or
        max_gross_exposure < 1 or not 0 <= fee_bps_per_side < 10000 or
        not 0 <= slippage_bps_per_side < 10000 or
        not 0 <= maintenance_margin_fraction < 1 or
        not 0 <= liquidation_safety_buffer_fraction < 1):
        raise ValueError("invalid risk or margin settings")
    if type(max_concurrent_positions) is not int or max_concurrent_positions < 1:
        raise ValueError("invalid max_concurrent_positions")

    ordered: list[tuple[pd.Timestamp, pd.Timestamp, RiskFill]] = []
    for fill in fills:
        entry, exit_ = pd.Timestamp(fill.entry_at), pd.Timestamp(fill.exit_at)
        if (entry.tzinfo is None or exit_.tzinfo is None or entry > exit_
            or not isinstance(fill.symbol, str) or not fill.symbol.strip()
            or not math.isfinite(fill.gross_return) or fill.gross_return <= -1
            or not math.isfinite(fill.stop_fraction) or
            not 0 < fill.stop_fraction < 1 or fill.side not in ('long', 'short')
            or not math.isfinite(fill.funding_fraction)):
            raise ValueError("invalid fill / stop / funding input")
        if fill.funding_fraction != 0 and fill.funding_reconciled is not True:
            raise ValueError("nonzero funding must be archive-reconciled")
        if require_funding_reconciled and fill.funding_reconciled is not True:
            raise ValueError("historical funding is required for every simulated fill")
        ordered.append((entry.tz_convert("UTC"), exit_.tz_convert("UTC"), fill))
    ordered.sort(key=lambda item: (item[0], item[2].symbol, item[1]))

    equity = float(starting_capital)
    peak = equity
    max_realized_drawdown = 0.0
    # release_at, symbol, notional, posted_margin, planned_risk, fee_net_return, funding_fraction
    positions: list[tuple[pd.Timestamp, str, float, float, float, float, float]] = []
    total_realized_funding_cash = 0.0
    skipped = {"symbol_overlap": 0, "capacity": 0, "insufficient_margin": 0,
               "liquidation_distance": 0, "equity_depleted": 0}
    accepted = 0
    traded_r: list[float] = []
    max_simultaneous_planned_risk = 0.0
    max_active_gross_to_equity = 0.0
    observed_max_leverage = 0.0
    ruination = False

    def settle(until: pd.Timestamp | None) -> None:
        nonlocal equity, peak, max_realized_drawdown, ruination, total_realized_funding_cash
        due = sorted((p for p in positions if until is None or p[0] <= until),
                     key=lambda p: (p[0], p[1]))
        for p in due:
            positions.remove(p)
            total_realized_funding_cash += p[2] * p[6]
            realized_pnl = p[2] * (p[5] + p[6])
            equity += realized_pnl
            traded_r.append(realized_pnl / p[4])
            peak = max(peak, equity)
            max_realized_drawdown = max(max_realized_drawdown,
                                        (peak - equity) / peak)
            if equity <= 0:
                ruination = True

    for entry, exit_, fill in ordered:
        settle(entry)
        if ruination:
            skipped["equity_depleted"] += 1
            continue
        if any(p[1] == fill.symbol for p in positions):
            skipped["symbol_overlap"] += 1
            continue
        if len(positions) >= max_concurrent_positions:
            skipped["capacity"] += 1
            continue
        cost_fn = net_after_costs_short if fill.side == 'short' else net_after_costs
        stop_net_loss = -cost_fn(
            -fill.stop_fraction, fee_bps_per_side, slippage_bps_per_side)
        if stop_net_loss <= 0:
            raise ValueError("planned stop must imply positive cost-inclusive loss")
        # Proxy only: exchange-specific maintenance tiers and liquidation engines
        # can liquidate substantially earlier, especially under cross-margin.
        denominator = (fill.stop_fraction + maintenance_margin_fraction
                       + liquidation_safety_buffer_fraction)
        if denominator >= 1:
            skipped["liquidation_distance"] += 1
            continue
        effective_leverage = min(max_leverage, .999 / denominator)
        if effective_leverage < 1:
            skipped["liquidation_distance"] += 1
            continue

        reserved_margin = sum(p[3] for p in positions)
        reserved_notional = sum(p[2] for p in positions)
        free_margin = max(0., equity - reserved_margin)
        # First solve for at most 1% stop risk; then apply liquidity/collateral
        # constraints WITHOUT rescaling beyond the risk budget.
        risk_limited_notional = equity * risk_fraction / stop_net_loss
        gross_capacity = max(0., equity * max_gross_exposure - reserved_notional)
        margin_capacity = free_margin * effective_leverage
        notional = min(risk_limited_notional, gross_capacity, margin_capacity)
        if notional <= 0 or not math.isfinite(notional):
            skipped["insufficient_margin"] += 1
            continue

        planned_loss = notional * stop_net_loss
        net_return = cost_fn(
            fill.gross_return, fee_bps_per_side, slippage_bps_per_side)
        posted_margin = notional / effective_leverage
        positions.append((
            exit_ + pd.Timedelta(minutes=1), fill.symbol, notional,
            posted_margin, planned_loss, net_return, fill.funding_fraction,
        ))
        accepted += 1
        observed_max_leverage = max(observed_max_leverage, effective_leverage)
        active_risk = sum(p[4] for p in positions)
        active_notional = sum(p[2] for p in positions)
        max_simultaneous_planned_risk = max(max_simultaneous_planned_risk,
                                            active_risk / equity)
        max_active_gross_to_equity = max(max_active_gross_to_equity,
                                         active_notional / equity)
    settle(None)

    ret = equity / starting_capital - 1
    return {
        "starting_capital": float(starting_capital),
        "final_equity": float(equity),
        "net_equity_return": float(ret),
        "net_30d_goal_met_in_development": bool(ret >= NET_30D_TARGET),
        "target_net_30d_return": NET_30D_TARGET,
        "executed_fills": accepted,
        "candidate_fills": len(fills),
        "skipped_fills": skipped,
        "risk_fraction_per_trade": risk_fraction,
        "max_leverage_allowed": max_leverage,
        "highest_effective_leverage_used": observed_max_leverage,
        "max_gross_exposure_multiple": max_gross_exposure,
        "max_concurrent_positions": max_concurrent_positions,
        "max_simultaneous_planned_risk_fraction_of_equity":
            max_simultaneous_planned_risk,
        "max_active_gross_exposure_to_equity": max_active_gross_to_equity,
        "mean_realized_r_on_planned_stop_risk": (
            sum(traded_r) / len(traded_r) if traded_r else None
        ),
        "max_realized_only_drawdown": max_realized_drawdown,
        "equity_depleted": ruination,
        "funding_reconciled_for_all_fills": bool(require_funding_reconciled),
        "realized_funding_pnl_cash": float(total_realized_funding_cash),
        "exchange_liquidation_engine_simulated": False,
        "release_approved": False,
        "limitations": [
            "One percent means estimated PLANNED risk to stop including fees, not a maximum loss.",
            "Gaps, funding, liquidation fees and bankruptcy can exceed planned risk.",
            "Archived funding is included ONLY when require_funding_reconciled is true; 1m opens approximate mark prices.",
            "Exchange-specific isolated/cross maintenance margin and fee tiers are not modelled.",
            "Realized-only drawdown can understate interim losses and liquidation risk.",
            "Equity/drawdown are settled-only, not intrabar marked-to-market; exposure under stress is understated.",
            "No market impact, partial fills or live order placement. Historical 30-day development windows only.",
        ],
    }
