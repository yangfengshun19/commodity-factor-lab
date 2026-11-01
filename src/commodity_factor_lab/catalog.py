"""Factor names and sector classifications inherited from the original project."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorFamily:
    label: str
    factors: tuple[str, ...]


FAMILIES = {
    "spread": FactorFamily("价差", ("C_frontnext", "C_mainsub", "C_nsnormal")),
    "warrant": FactorFamily("仓单", ("warrant_count", "W_pct", "W_ratio")),
    "basis": FactorFamily("期现", ("fs_basis", "fs_carry", "fs_exmom")),
    "trend": FactorFamily("量价趋势", ("t_macross", "t_mamom", "t_maratio", "t_mom")),
    "positioning": FactorFamily(
        "持仓",
        tuple(
            f"pr_{kind}_top{rank}" for rank in (5, 10, 15, 20) for kind in ("netmom", "lschgratio")
        ),
    ),
    "value": FactorFamily("价值", ("Val_halfyear", "Val_oneyear")),
    "volatility": FactorFamily("波动", ("vol_std", "vol_gk", "vol_rs")),
}

_SECTORS = {
    "financial": "if ih ic im ts tf t tl",
    "ferrous": "rb hc ss wr i j jm sf sm zc",
    "nonferrous": "cu al pb zn sn ni bc ao",
    "precious": "au ag",
    "energy_chemicals": "sc fu lu bu pg l v pp ta eg eb pf pr ma ur fg sa sh px ru nr br",
    "agriculture": "m rm a b rs pk y p oi c cs rr jr ri lr pm wh cf cy sr jd lh ap cj",
    "new_materials": "si lc ps",
    "other": "sp ec fb bb",
}
SECTOR_MAP = {code: sector for sector, codes in _SECTORS.items() for code in codes.split()}
SECTOR_LABELS = {
    "all_commodities": "全商品",
    "financial": "金融期货",
    "ferrous": "黑色金属",
    "nonferrous": "有色金属",
    "precious": "贵金属",
    "energy_chemicals": "能源化工",
    "agriculture": "农产品",
    "new_materials": "新能源与新材料",
    "other": "其他",
}


def code_from_filename(filename: str) -> str:
    parts = filename.split("_")
    if len(parts) < 3 or not parts[1].isascii() or not parts[1].isalpha():
        raise ValueError(f"Expected <name>_<code>_<suffix>.csv: {filename}")
    code = parts[1].lower()
    if code not in SECTOR_MAP:
        raise ValueError(f"Unknown product code: {code}")
    return code
