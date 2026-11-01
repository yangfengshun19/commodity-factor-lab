"""Integer-contract, opening-execution and daily-settlement cash ledger.

Margin is collateral reserved from equity, never a second trading loss. Orders
must predate actual session openings. Failed product rolls are atomic. Source
certification, intraday liquidity and broker liquidation are outside this engine.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .curve_factors import utc_column


@dataclass
class ExecutionResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    rejected: pd.DataFrame
    holdings: pd.DataFrame
    capacity_events: pd.DataFrame = field(default_factory=pd.DataFrame)


def settle_portfolio(
    bars: pd.DataFrame,
    calendar: pd.DataFrame,
    orders: pd.DataFrame,
    initial_cash: float = 1_000_000,
    max_margin_fraction: float = 0.5,
    liquidate: bool = True,
    capacity_policy: str = "atomic",
) -> ExecutionResult:
    required = {
        "date",
        "code",
        "contract",
        "open",
        "settle",
        "multiplier",
        "margin_rate",
        "fee_rate",
        "fee_per_lot",
        "close_fee_rate",
        "close_fee_per_lot",
        "can_buy",
        "can_sell",
        "spec_available_at",
        "expiry",
    }
    if required - set(bars) or {"date", "open_at"} - set(calendar):
        raise ValueError(
            "Execution requires contract prices, dated specs, directional fill flags and actual openings"
        )
    if {"date", "contract", "target_lots", "decision_at"} - set(orders):
        raise ValueError(
            "Orders require execution date, target integer lots and decision timestamp"
        )
    if not np.isfinite(initial_cash) or initial_cash <= 0 or not 0 < max_margin_fraction <= 1:
        raise ValueError("Invalid starting capital or margin budget")
    data, sessions, plans = bars.copy(), calendar.copy(), orders.copy()
    capacity = "max_trade_lots" in data or "capacity_available_at" in data
    if capacity_policy not in {"atomic", "staged"} or (
        capacity_policy == "staged" and not capacity
    ):
        raise ValueError("Staged execution requires explicit capacity inputs")
    if capacity:
        if {"max_trade_lots", "capacity_available_at"} - set(data):
            raise ValueError("Capacity requires an integer lot limit and availability clock")
        data["capacity_available_at"] = utc_column(data.capacity_available_at)
        data["max_trade_lots"] = pd.to_numeric(data.max_trade_lots, errors="raise")
        if (
            not np.isfinite(data.max_trade_lots).all()
            or data.max_trade_lots.lt(0).any()
            or not data.max_trade_lots.eq(np.trunc(data.max_trade_lots)).all()
        ):
            raise ValueError("Capacity lot limits must be finite nonnegative integers")
    for frame in (data, sessions, plans):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    sessions["open_at"] = utc_column(sessions["open_at"])
    data["spec_available_at"] = utc_column(data["spec_available_at"])
    phased_margin = any(
        name in data for name in ("settlement_margin_rate", "settlement_spec_available_at")
    )
    if phased_margin:
        if {"settlement_margin_rate", "settlement_spec_available_at"} - set(
            data
        ) or "settlement_at" not in sessions:
            raise ValueError(
                "Phased margin requires a settlement rate, availability and calendar clock"
            )
        data["settlement_spec_available_at"] = utc_column(data["settlement_spec_available_at"])
    if "settlement_at" in sessions:
        sessions["settlement_at"] = utc_column(sessions["settlement_at"])
        if (sessions.settlement_at <= sessions.open_at).any() or (
            sessions.settlement_at.iloc[:-1].to_numpy() >= sessions.open_at.iloc[1:].to_numpy()
        ).any():
            raise ValueError(
                "Settlement clock must follow its opening and precede the next opening"
            )
    if phased_margin:
        data["settlement_margin_rate"] = pd.to_numeric(data.settlement_margin_rate, errors="raise")
        if (
            not np.isfinite(data.settlement_margin_rate).all()
            or not data.settlement_margin_rate.between(0, 1, inclusive="right").all()
        ):
            raise ValueError("Invalid settlement margin rate")
    plans["decision_at"] = utc_column(plans["decision_at"])
    data["expiry"] = pd.to_datetime(data["expiry"], errors="raise").dt.normalize()
    if data.expiry.isna().any() or any(
        frame.date.isna().any() for frame in (data, sessions, plans)
    ):
        raise ValueError("Execution dates and documented expiries must be complete")
    if (
        not sessions.date.is_unique
        or not sessions.date.is_monotonic_increasing
        or not sessions.open_at.is_unique
        or not sessions.open_at.is_monotonic_increasing
    ):
        raise ValueError("Execution calendar must be strictly chronological")
    if data.duplicated(["date", "contract"]).any() or plans.duplicated(["date", "contract"]).any():
        raise ValueError("Duplicate marks or target orders")
    if not data.date.isin(sessions.date).all() or not plans.date.isin(sessions.date).all():
        raise ValueError("Marks/orders outside declared calendar")
    for col in (
        "open",
        "settle",
        "multiplier",
        "margin_rate",
        "fee_rate",
        "fee_per_lot",
        "close_fee_rate",
        "close_fee_per_lot",
    ):
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col]).all():
            raise ValueError(f"Missing or nonfinite {col}")
    if (
        data.multiplier.le(0).any()
        or not data.margin_rate.between(0, 1, inclusive="right").all()
        or data[["fee_rate", "fee_per_lot", "close_fee_rate", "close_fee_per_lot"]]
        .lt(0)
        .any()
        .any()
    ):
        raise ValueError("Invalid multiplier, margin or fee specification")
    for col in ("can_buy", "can_sell"):
        if data[col].isna().any() or not data[col].isin([True, False]).all():
            raise ValueError(
                "Fill flags must be explicit booleans, never inferred from full-day volume"
            )
    if (
        data.groupby("contract").multiplier.nunique().gt(1).any()
        or data.groupby("contract").code.nunique().gt(1).any()
    ):
        raise ValueError(
            "Contract multiplier/identity changes require an explicit corporate-action model"
        )
    plans["target_lots"] = pd.to_numeric(plans["target_lots"], errors="raise")
    if (
        not np.isfinite(plans.target_lots).all()
        or not plans.target_lots.eq(np.trunc(plans.target_lots)).all()
    ):
        raise ValueError("Orders must use finite integer lots")
    openings = sessions.set_index("date").open_at
    if "order_cutoff_at" in sessions:
        if "settlement_at" not in sessions:
            raise ValueError("Order submission cutoffs require explicit settlement clocks")
        sessions["order_cutoff_at"] = utc_column(sessions.order_cutoff_at)
        if (
            not sessions.order_cutoff_at.is_unique
            or not sessions.order_cutoff_at.is_monotonic_increasing
            or (sessions.order_cutoff_at > sessions.open_at).any()
            or (
                sessions.settlement_at.iloc[:-1].to_numpy()
                >= sessions.order_cutoff_at.iloc[1:].to_numpy()
            ).any()
        ):
            raise ValueError(
                "Order cutoff must not follow its opening or precede the previous settlement"
            )
        deadlines = sessions.set_index("date").order_cutoff_at
        if (plans.decision_at >= plans.date.map(deadlines)).any():
            raise ValueError("Order decision must precede the order submission cutoff")
    elif (plans.decision_at >= plans.date.map(openings)).any():
        raise ValueError("Order decision must precede the actual opening; check night sessions")
    if (data.spec_available_at > data.date.map(openings)).any():
        raise ValueError("Contract specifications were unavailable before execution")
    if capacity:
        capacity_deadlines = sessions.set_index("date").get("order_cutoff_at", openings)
        if (data.capacity_available_at >= data.date.map(capacity_deadlines)).any():
            raise ValueError("Capacity inputs must precede the execution submission deadline")
    if (
        phased_margin
        and (
            data.settlement_spec_available_at
            > data.date.map(sessions.set_index("date").settlement_at)
        ).any()
    ):
        raise ValueError("Settlement margin specifications were unavailable at settlement")
    marks = data.set_index(["date", "contract"])
    contract_code = data.drop_duplicates("contract").set_index("contract").code.to_dict()
    if not plans.contract.isin(contract_code).all():
        raise ValueError("Unknown contract in target orders")
    holdings, last_settle, cash = {}, {}, float(initial_cash)
    trades, rejections, positions, daily = [], [], [], []
    outstanding, capacity_events = {}, []
    forced = False

    def row_for(date, contract):
        if (date, contract) not in marks.index:
            raise ValueError(f"Missing held/ordered contract mark: {contract} on {date.date()}")
        return marks.loc[(date, contract)]

    def margin_at(book, date, field):
        total = 0.0
        for contract, lots in book.items():
            if not lots:
                continue
            row = row_for(date, contract)
            floor = float(row.get("margin_floor_price", 0.0))
            if not np.isfinite(floor) or floor < 0:
                raise ValueError("Invalid margin price floor")
            rate = (
                row.settlement_margin_rate
                if phased_margin and field == "settle"
                else row.margin_rate
            )
            total += abs(lots) * max(abs(row[field]), floor) * row.multiplier * rate
        return float(total)

    for number, session in enumerate(sessions.itertuples()):
        date, opening_cash = session.date, cash
        overnight = 0.0
        for contract, lots in holdings.items():
            row = row_for(date, contract)
            if date > row.expiry:
                raise ValueError("A held contract has passed its documented expiry")
            overnight += lots * row.multiplier * (row.open - last_settle[contract])
        equity_open = cash + overnight
        if equity_open <= 0:
            raise ValueError("Portfolio insolvent at opening; no valid normal return series")
        desired = (
            plans.loc[plans.date.eq(date)].set_index("contract").target_lots.astype(int).to_dict()
        )
        if capacity_policy == "staged":
            new_codes = {contract_code[c] for c in desired}
            outstanding = {
                c: q for c, q in outstanding.items() if contract_code[c] not in new_codes
            }
            outstanding.update(desired)
            desired = outstanding.copy()
        terminal = liquidate and number == len(sessions) - 1
        if forced or terminal:
            desired = {contract: 0 for contract in set(holdings) | set(desired)}
            if capacity_policy == "staged":
                outstanding = desired.copy()
        # Each mentioned product is a complete target, including closure of an
        # old contract after a previously rejected roll. Never open both by accident.
        mentioned_codes = {contract_code[c] for c in desired}
        for contract in holdings:
            if contract_code[contract] in mentioned_codes:
                desired.setdefault(contract, 0)
        fees = 0.0
        # Deterministic product order is part of the execution policy.
        for code in sorted({contract_code[c] for c in desired}):
            requested = {c: int(q) for c, q in desired.items() if contract_code[c] == code}
            before = {c: holdings.get(c, 0) for c in requested}
            effective, capacity_reasons = requested.copy(), {}
            if capacity_policy == "staged":
                for contract, target in requested.items():
                    old, delta = before[contract], target - before[contract]
                    if not delta:
                        continue
                    row = row_for(date, contract)
                    step = int(np.clip(delta, -row.max_trade_lots, row.max_trade_lots))
                    if not bool(row.can_buy if delta > 0 else row.can_sell):
                        step = 0
                        capacity_reasons[contract] = "directional_fill_blocked"
                    elif step != delta:
                        capacity_reasons[contract] = "volume_limited"
                    effective[contract] = old + step
                # A delayed exit must not accidentally create a calendar spread.
                old_contract_remains = any(
                    before[c] != 0 and requested[c] == 0 and effective[c] != 0 for c in requested
                )
                if old_contract_remains:
                    for contract, target in requested.items():
                        if before[contract] == 0 and target != 0:
                            effective[contract] = 0
                            capacity_reasons[contract] = "deferred_until_old_contract_flat"
            changes = {
                c: int(q) - holdings.get(c, 0)
                for c, q in effective.items()
                if contract_code[c] == code and int(q) != holdings.get(c, 0)
            }
            if not changes and not capacity_reasons:
                continue
            proposed, proposed_fees, legs, blocked = holdings.copy(), 0.0, [], ""
            for contract, delta in sorted(changes.items()):
                row = row_for(date, contract)
                if capacity and abs(delta) > row.max_trade_lots:
                    blocked = f"capacity_limit:{contract}"
                    break
                if not bool(row.can_buy if delta > 0 else row.can_sell):
                    blocked = f"directional_fill_blocked:{contract}"
                    break
                target = effective[contract]
                if target != 0 and date >= row.expiry:
                    blocked = f"entry_on_expiry:{contract}"
                    break
                old = holdings.get(contract, 0)
                closed = min(abs(old), abs(delta)) if old * delta < 0 else 0
                opened = abs(delta) - closed
                fee = opened * (abs(row.open) * row.multiplier * row.fee_rate + row.fee_per_lot)
                fee += closed * (
                    abs(row.open) * row.multiplier * row.close_fee_rate + row.close_fee_per_lot
                )
                proposed[contract] = target
                proposed_fees += fee
                legs.append(
                    {
                        "date": date,
                        "code": code,
                        "contract": contract,
                        "old_lots": old,
                        "new_lots": target,
                        "delta_lots": delta,
                        "closed_lots": closed,
                        "opened_lots": opened,
                        "fill_price": row.open,
                        "fee": fee,
                        "reason": "margin_liquidation"
                        if forced
                        else ("terminal_liquidation" if terminal else "rebalance"),
                    }
                )
            if not blocked:
                budget = (equity_open - fees - proposed_fees) * max_margin_fraction
                new_margin = margin_at(proposed, date, "open")
                old_margin = margin_at(holdings, date, "open")
                if equity_open - fees - proposed_fees <= 0 or (
                    new_margin > budget and new_margin >= old_margin
                ):
                    blocked = "insufficient_margin_budget"
            if blocked:
                rejections.append(
                    {
                        "date": date,
                        "code": code,
                        "reason": blocked,
                        "contracts": ";".join(sorted(changes)),
                        "atomic_product_order": True,
                    }
                )
            else:
                holdings = {c: q for c, q in proposed.items() if q != 0}
                fees += proposed_fees
                trades.extend(legs)
            if capacity:
                for contract, target in requested.items():
                    if target == before[contract]:
                        continue
                    capacity_events.append(
                        {
                            "date": date,
                            "code": code,
                            "contract": contract,
                            "starting_lots": before[contract],
                            "requested_target_lots": target,
                            "attempted_target_lots": effective[contract],
                            "filled_delta_lots": holdings.get(contract, 0) - before[contract],
                            "remaining_lots": target - holdings.get(contract, 0),
                            "max_trade_lots": int(row_for(date, contract).max_trade_lots),
                            "limit_reason": capacity_reasons.get(contract, ""),
                            "rejection_reason": blocked,
                            "forced_or_terminal": forced or terminal,
                        }
                    )
        opening_margin = margin_at(holdings, date, "open")
        intraday = 0.0
        for contract, lots in holdings.items():
            row = row_for(date, contract)
            intraday += lots * row.multiplier * (row.settle - row.open)
            last_settle[contract] = row.settle
            positions.append(
                {
                    "date": date,
                    "contract": contract,
                    "code": row.code,
                    "lots": lots,
                    "settle": row.settle,
                    "multiplier": row.multiplier,
                }
            )
        cash = equity_open + intraday - fees
        if cash <= 0:
            raise ValueError("Portfolio insolvent after daily settlement")
        margin = margin_at(holdings, date, "settle")
        forced = margin > cash * max_margin_fraction
        daily.append(
            {
                "date": date,
                "previous_equity": opening_cash,
                "opening_equity": equity_open,
                "opening_reserved_margin": opening_margin,
                "overnight_pnl": overnight,
                "intraday_pnl": intraday,
                "fees": fees,
                "equity": cash,
                "reserved_margin": margin,
                "available_cash": cash - margin,
                "net_return": cash / opening_cash - 1,
                "margin_call": forced,
                "contracts_held": len(holdings),
                "terminal_unliquidated": terminal and bool(holdings),
                "reconciliation_error": cash - (opening_cash + overnight + intraday - fees),
            }
        )
    return ExecutionResult(
        pd.DataFrame(daily).set_index("date"),
        pd.DataFrame(
            trades,
            columns=[
                "date",
                "code",
                "contract",
                "old_lots",
                "new_lots",
                "delta_lots",
                "closed_lots",
                "opened_lots",
                "fill_price",
                "fee",
                "reason",
            ],
        ),
        pd.DataFrame(
            rejections, columns=["date", "code", "reason", "contracts", "atomic_product_order"]
        ),
        pd.DataFrame(
            positions, columns=["date", "contract", "code", "lots", "settle", "multiplier"]
        ),
        pd.DataFrame(
            capacity_events,
            columns=[
                "date",
                "code",
                "contract",
                "starting_lots",
                "requested_target_lots",
                "attempted_target_lots",
                "filled_delta_lots",
                "remaining_lots",
                "max_trade_lots",
                "limit_reason",
                "rejection_reason",
                "forced_or_terminal",
            ],
        ),
    )
