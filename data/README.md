# 数据目录与接口

`raw/` 保留从原项目迁移的逐品种**因子快照**，不是交易所逐合约原始行情。原始文件不被清洗函数改写。`archive/` 是旧程序生成的合并面板；`demo/` 是程序生成的合成数据。上述目录默认不纳入 Git。

## 因子快照输入

```text
data/raw/
├── spread/        # C_frontnext, C_mainsub, C_nsnormal
├── warrant/       # warrant_count, W_pct, W_ratio
├── basis/         # fs_basis, fs_carry, fs_exmom
├── trend/         # t_macross, t_mamom, t_maratio, t_mom
├── positioning/   # pr_netmom_topN, pr_lschgratio_topN; N = 5,10,15,20
├── value/         # Val_halfyear, Val_oneyear
└── volatility/    # vol_std, vol_gk, vol_rs
```

每个 CSV 表示一个品种，文件名为 `<名称>_<品种代码>_<说明>.csv`，如 `螺纹钢_rb_factors.csv`。品种代码忽略大小写，必须存在于 `catalog.py` 的映射。UTF-8 优先，读取时支持 GB18030 回退。

| 字段 | 要求 |
|---|---|
| date | `YYYY-MM-DD`，同一品种不得重复；必须是真实交易日期 |
| open, high, low, close | 正数、有限值、OHLC 高低关系一致 |
| 该类因子列 | 使用上方名称，数值类型；缺失值保留为空并报告覆盖率 |

一个目录中不得有同一品种的重复文件。其他数据如 `volume`、`spot_price` 可以保留在源文件中；旧探索接口使用 OHLC 与注册因子；受控研究还必须使用成交量筛选。缺失因子列会记录并置为空，不会假造数值。异常 OHLC 保留日期，将当日信号置为空；非正数或非有限的价格字段置为空，防止跳过坏行后计算跨日收益。开盘价单独验证，不因为当日稍后的收盘或最高最低价异常而事后取消一个本可成交的开盘订单。

日期早于 2000 年的快照记录视为异常。没有 `date`、但存在明确 ISO 日期格式的 `Unnamed: 0` 时，允许在内存中重命名。其他异常不会推测日期。

## 重新计算价格因子

`build-price` 读取一个目录内的 OHLC CSV，生成 `trend/`、`value/` 和 `volatility/` 三个子目录。输入文件名仍需包含有效品种代码，至少提供 253 个交易日才能获得历史命名为一年价值的价格指标（实际应解释为过去收益）。函数不会截断预热历史；运行研究时再按配置日期筛选。

```bash
cfl build-price --input data/market --output data/features
```

在新的配置中将 `[paths].data` 指向 `data/features`，并将 `families` 设为三类价格因子。输入和输出必须分离；已有同名特征文件不会被覆盖。

## 历史采集接口

现有 AKShare 和 MySQL 程序保存在 `research/legacy/`，用来核对原研究实现。主研究只读本地文件；新增 fetch-sina 命令单独采集并保留来源版本。后续适配器应保存交易所、具体合约、真实到期日、来源、抓取时间、发布时点、复权/换月方法和字段单位，先形成可复核的快照，再接入因子计算。

## v0.2 受控研究输入

`study` 从一个数据根目录的 `trend/` 读取 date、OHLC 和 volume，自行计算三种固定信号，不读取旧因子列。缺失 volume 时不会假造流动性资格。预热要求、日期对齐和持仓缺价处理见[方法](../docs/methodology.md)。

`data/snapshots/` 存储新采集版本，每个新目录包含：

```text
snapshot.json        成功/失败、来源地址、抓取时间、内容哈希及认证边界
responses/*.jsonp    供应商原响应（未截断的返回历史）
trend/*.csv          截断至声明结束日的规范化价格
```

```bash
cfl fetch-sina --codes a c m y rb cu --through 2025-10-09 --output data/snapshots/example
```

命令只访问公开价格接口，不调用旧数据库脚本。保存目录必须不存在，不会覆盖或自动拼接旧数据；单个品种失败会记录并继续，不会假装成功。原响应可能含研究截止日之后的记录，但规范输入按截止日截断，研究不得使用原响应中的未来记录形成过去信号。

`study` 会检查已下载规范文件是否完整匹配清单哈希，包括删除文件的情况。它没有认证供应商历史修订、真实合约或历史可用时点；抓取时间只是本次下载的时间。

比较两份版本，输出独立审计而不修复价格：

```bash
python scripts/compare_price_vintages.py --old data/raw --new data/snapshots/example --output reports/audits/price_vintages
```

真实合约输入与交易时段要求另见[合约数据](../docs/contract_data.md)。


## v0.3 官方逐合约与注册仓单

当前主研究规范与字段见[合约输入](../docs/contract_data.md)，本机覆盖和已发现的来源问题见[数据就绪度](../docs/source_readiness.md)。数据仍默认只在本地保存。首次准备使用独立目录：

```bash
python -m pip install -e '.[ingest]'
cfl fetch-shfe --years 2024 2025 --output data/market/example_archives
cfl ingest-shfe --archives data/market/example_archives/2024.zip data/market/example_archives/2025.zip --codes cu al zn rb hc ru --output data/market/example_bars
cfl fetch-shfe-metadata --dates 20240130 20240229 --output data/market/example_metadata
cfl fetch-warrants --dates 20240105 20240112 --output data/market/example_warrants
```

以上短样本只演示采集格式，不能满足 252 日基差动量和三年季节仓单历史要求。正式研究需准备覆盖预热及整个评估期的输入，并在配置中明确路径。仓单最多一次 400 个预先固定的周采样日期，连续五次失败即停止；清单分别显示成功、失败和未尝试，不静默填缺。

字段结构升级时，可以读取已保存响应生成新派生版本，无须重新下载：

```bash
cfl reparse-shfe --source data/market/example_metadata --output data/market/example_metadata_v2
cfl reparse-warrants --source data/market/example_warrants --output data/market/example_warrants_v2
```

新目录须不存在，解析会校验原响应哈希并记录源清单。仓单只取仓库总计，厂库不混入。仅有日期的时钟无法进入严格时点流程。年度行情未认证发布时间时只能显式采用回溯模式；绝不能把源 report_date 等同于历史生效或可用时间。

离线端到端演示的合成行情、参数与仓单保存在 `data/contract_demo/<run-id>/`，不进入 Git 或公开包。它独立生成数据，不依赖上面的真实本机目录。

## v0.4 扩样接口

仓单采集和再解析可用 --codes 显式指定注册品种；默认六品种不变。例如在已保存周响应上用 cfl reparse-warrants --source data/market/shfe_warrants_weekly_2020_2025 --output data/market/example_expanded --codes cu al zn rb hc ru bu sp au ag。黄金未在此接口返回，不补零。新 fetch-shfe-metadata 输出规范文件哈希并拒绝 report_date 不匹配的响应；原响应与失败哈希仍保存。覆盖筛选脚本和本轮实际数据范围见[复核记录](../docs/validation_results.md)。
