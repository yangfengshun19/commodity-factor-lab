"""Maintained factors have explicit roles; legacy indicator names stay archival."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    label: str
    role: str
    formula: str
    mechanism: str
    unit: str
    direction: int
    controls: tuple[str, ...] = ()


FACTORS = (
    FactorDefinition(
        "momentum_126d",
        "126 日截面动量",
        "baseline",
        "product(1 + lag-selected rank-1 same-contract returns, 126 sessions) - 1",
        "A price-information baseline; not fundamental value",
        "return",
        1,
    ),
    FactorDefinition(
        "carry",
        "期限结构斜率",
        "candidate",
        "log(F1/F2) / ((expiry2-expiry1).days / 365.25)",
        "Backwardation may reflect scarcity and risk compensation; hypothesis, not proof",
        "annual log price slope",
        1,
        ("momentum_126d",),
    ),
    FactorDefinition(
        "basis_momentum",
        "基差动量",
        "candidate",
        "252-session compounded rank-1 return minus compounded rank-2 return",
        "Historical near/far return divergence; this roll policy must be replicated explicitly",
        "return difference",
        1,
        ("momentum_126d", "carry"),
    ),
    FactorDefinition(
        "inventory_scarcity",
        "季节仓单稀缺分数",
        "candidate",
        "0.5 - percentile(current warrants within earlier seasonal anniversaries)",
        "Registered deliverable warrants are a limited scarcity proxy, not total inventories",
        "within-product percentile",
        1,
        ("momentum_126d", "carry"),
    ),
    FactorDefinition(
        "relative_basis",
        "相对基差代理",
        "exploratory",
        "annual log front/next slope minus annual log next/third slope",
        "A transparent curve-shape diagnostic; not an exact replication of every relative-basis paper",
        "annual log slope difference",
        1,
        ("momentum_126d", "carry", "basis_momentum"),
    ),
    FactorDefinition(
        "risk_volatility",
        "历史波动率",
        "risk_only",
        "std(rank-1 same-contract daily returns, 60 sessions) * sqrt(252)",
        "Risk measurement; no unconditional long/short prediction is asserted",
        "annualised volatility",
        0,
    ),
    FactorDefinition(
        "normalized_position_flow",
        "规范化 Top20 榜单净持仓变化",
        "exploratory",
        "(net_position(t)-net_position(t-20))/total_open_interest(t)",
        "Strict same-contract 21-session rank window, publication cutoffs and single-sided OI checks; changing Top20 is not a fixed investor cohort or COT category",
        "share of open interest",
        1,
        ("momentum_126d", "carry", "basis_momentum"),
    ),
    FactorDefinition(
        "spot_basis",
        "可核验现货基差",
        "future_candidate",
        "log(unit-converted, matched spot / signal-selected contract settlement)",
        "Requires verified grade, location, tax, units, contract mapping and publication history; no certified input panel yet",
        "log price gap",
        1,
        ("momentum_126d", "carry", "basis_momentum"),
    ),
    FactorDefinition(
        "long_horizon_value_proxy",
        "长期价值代理设计",
        "research_design",
        "not registered: requires long spot history and a separately documented economic definition",
        "Legacy positive 126/252-session returns are momentum; no value effect is asserted or computed here",
        "not specified",
        0,
    ),
)


def definitions() -> list[dict]:
    return [asdict(item) for item in FACTORS]
