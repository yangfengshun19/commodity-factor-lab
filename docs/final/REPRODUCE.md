# 运行与复现边界

2026-11-01。本次公开交付以原项目根目录 README、`pyproject.toml`（v0.17.0）、本文及[结论](REPORT.md)为准。`FIn-claude` 是研究工作副本；它的新脚本、协议与测试不能假定已包含在公开包。

| 内容 | 公开源码包包含 | 额外条件 | 能验证什么 |
|---|---|---|---|
| 离线合成演示、原代码和测试 | 包含 | Python 3.11/3.12 及声明依赖 | 工程运行、合成资金账；不验证市场盈利 |
| v0.17 真实因子结果的说明、方法和配置 | 包含 | 无需数据即可阅读 | 审查方法与已报告数字；不能仅凭说明独立重算 |
| 官方行情、仓单、元数据原响应及规范化数据 | 不包含 | 合法持有本机相同输入及版本 | 重建真实特征；当前下载不保证与旧版本一致 |
| `reports/runs/`、`reports/protocols/`、审计回执 | 不包含 | 本机历史运行产物、冻结协议与哈希 | 复算和核对已报告真实结果 |
| `docs/collaboration/`、未冻结 draft 文档 | 不包含 | 本机协作档案 | 追溯交流和未决事项；不作为公开冻结协议 |
| Claude 新估计器、桥接、人工验证及数字核对脚本 | 本次不包含 | 研究副本代码、`work/` 产物和部分原项目输入 | 按该副本说明复核补充方法工作；不属公开 v0.17 安装包能力 |
| S1、E、S2 的新研究结果 | 不存在 | 尚待来源、评审或采集 | 不能声称已复现、已失败或已获确认 |

## 公开包：离线运行

解压源码包并进入根目录，或使用同样内容的源码目录：

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m unittest discover -s tests
python -m commodity_factor_lab contract-demo
```

安装依赖可能需要联网；装好后 `contract-demo` 不访问网络。依赖要求 `pandas>=2.2,<3`，不要将其他依赖环境的测试结果混作声明环境结果。合成报告位于 `reports/runs/contract_demo/<时间戳>/index.html`；其 `settlement_demo/summary.csv` 可检查现金对账误差及期末未平仓状态。全部演示数据为合成数据。

## 本机真实研究：必须保留原输入

以下命令在原项目根目录执行。公开包里的配置记录了原输入路径与哈希；它不提供缺失数据，也不跳过冻结核验。缺任何输入或协议回执时，不能以新抓取文件直接替换后声称原样复现。

| 运行 | 命令 | 需要的本机输入 / 对照 |
|---|---|---|
| 十品种特征与原诊断 | `cfl contract-research --config configs/curve_expanded.toml` | `data/market/shfe_all_registered_2020_2025/`、`shfe_metadata_v2_20261029/`、`shfe_warrants_expanded_2020_2025/` 等配置所列文件；原输出 `reports/runs/curve_expanded/20261029T045315046146Z/` |
| v0.17 共同样本 | `cfl factor-evidence --config configs/factor_evidence.toml` | 上述原目录的 `curves.csv`、`labels.csv`、`calendar_assumption.csv`、原诊断，以及配置所列登记/冻结材料；原输出 `reports/runs/factor_evidence/20261031T102024091795Z/` |
| v0.14 市场基准 | `cfl market-benchmark --config configs/market_benchmark.toml` | 配置列明的行情、manifest、原曲线、日历、登记及冻结材料；原输出 `reports/runs/market_benchmark/20261031T063013399593Z/` |

读取官方 Excel 年包另需 `python -m pip install -e ".[ingest]"`。具体协议与其他运行入口见[因子方法](../factor_evidence_methodology.md)、[基准方法](../market_benchmark_methodology.md)和根目录 README。重新运行会生成新的输出目录，不覆盖原证据。

## 本轮已核对与仍未执行

本包当前 **250 项测试通过**（本机 Python 3.12、pandas 2.3.3、statsmodels 0.15.0），包括打包边界测试；离线演示四本合成账现金对账误差均为 0，期末无未平仓。源码 ZIP 与安装源包均核验公开边界及 README、最终说明的本地相对链接；具体清单与哈希随本机核对回执保存，新增文档后重新生成核验记录。

Codex 已用本机原项目 Python 3.12 环境，对 Claude 当前副本跑完 269 项测试（全部通过），核对其报告索引 42/42 项，面板接口与 statsmodels 的系数及协方差差异均小于 1e-10。此为研究副本的核对记录，不能写成公开 v0.17 本身含 269 项测试。本轮未重跑容量、基准或新的市场收益实验，未重新运行完整人工模拟与 S0 桥接。

C004 新探针和两所官网人工核查仍未执行；不能据访问受阻断言数据不存在。S1、E、S2 均无新研究结果，作为后续工作保留。发布包为本地准备产物，尚未因此上传到远程仓库；许可证及数据再分发权利仍需作者确认。

完整因子范围见[覆盖表](FACTORS.md)，长短摘要与本地交付入口见[上传说明](RELEASE.md)。未认证的历史首发版本、无法完全核实的原论文样本使用史和未知搜索，不能通过重新运行同样代码消除；克隆或安装成功也不构成独立市场验证。
