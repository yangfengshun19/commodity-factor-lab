# 逐合约输入与现金账规范 v0.16

`contract-research` 研究特征与未来标签；`execution-ledger` 执行严格资金记账；旧 `contract-panel` 仅为同合约开盘收益的兼容组件。三者不可混称完整投资回测。

## 特征研究输入

| 文件 | 必需字段 |
|---|---|
| bars.csv | date、code、contract、open、high、low、close、settle、volume、open_interest；严格时点另需 available_at |
| metadata.csv | contract、expiry、metadata_available_at；可保留 listed、source 和时间来源 |
| warrants.csv | code、period_date、available_at、value、unit、source；建议另存 measure |

真实到期日必须来自权威日期，不能从合约月份猜每月 15 日。日期/合约键唯一，合约身份不能换品种。缺价不借用邻近合约或静默填补；无成交日的开盘缺失与结算价可用分别审计。当前特征要求正结算价，对非正价格的商品须另行定义特征；现金账可记非正价格盈亏。

每个规范输入旁须有 manifest.json，包含 kind、normalized_sha256 及来源溯源。研究读取前校验规范文件哈希；公开采集器还保存原响应、UTC 抓取时间、原响应哈希及失败清单。时间必须显式带 Z 或 UTC 偏移。抓取时间、报告经济日期、发布时间和生效时间是不同字段。

## 现金账执行输入

| 文件 | 字段与要求 |
|---|---|
| 行情与规格 | date、code、contract、open、settle、multiplier、margin_rate、fee_rate、fee_per_lot、close_fee_rate、close_fee_per_lot、can_buy、can_sell、spec_available_at、expiry |
| 实际时段日历 | date、open_at，日期和实际时刻严格递增；品种日历不兼容须分开运行 |
| 订单 | date、contract、target_lots、decision_at，手数为有限整数，同日合约不重复 |

手续费 = 新开手数×(绝对开盘价×乘数×fee_rate+fee_per_lot) + 平仓手数×(绝对开盘价×乘数×close_fee_rate+close_fee_per_lot)。同合约反转先平再开；双腿换月分别收费。费率为小数，不是百分数或万分数显示值。当前不区分平今优惠或额外罚费；有这些需求时须扩展库存年龄及费用模型。

margin_rate 为 (0,1] 内小数；可选 margin_floor_price 为非负绝对计价下限。保证金取绝对价格或下限较大者乘乘数、手数与比例。保证金只占用权益，不重复当费用扣除。当前多空同一合同使用同一输入比例，无跨品种抵扣和经纪商规则认证。

v0.12支持同日开盘/结算两阶段：`margin_rate`用于开盘预算；若结算比例不同，bars必须同时提供`settlement_margin_rate`和`settlement_spec_available_at`，calendar提供`settlement_at`，三个字段缺一即拒绝。结算参数须不晚于结算，结算须晚于本次开盘且早于下一开盘；可以是开盘后、结算前已公告的变更。未提供扩展时，明确沿用同日单比例假设。daily另输出`opening_reserved_margin`，原`reserved_margin`仍是结算占用。真实样本及合成手算见[两阶段核对](execution_readiness_results.md)。

v0.13另支持`calendar.order_cutoff_at`，用于集合竞价等存在更早申报截止的执行假设。提供时必须同时给出`settlement_at`：截止不得晚于本次执行时刻，且须严格晚于前一结算；订单决策必须严格早于截止。只给`settlement_at`可以配合不变保证金，不必填写新的结算比例。没有截止字段时仍按原实际执行时刻校验，不能把这种兼容模式自动宣称为竞价成交。计划复市竞价08:55—09:00与四分钟申报规则可给出08:59截止，实际开盘、报价来源及成交仍要外部证据；见[规则与时序核对](execution_rule_review_results.md)。

`spec_available_at` 必须不晚于实际开盘，`decision_at` 必须早于开盘。夜盘交易日标签通常不等于自然日，不能统一填交易日 09:00。can_buy/can_sell 必须是显式方向条件，不能由全日成交量推断；标志为真也仅表示输入允许模型在开盘完整成交，不等于已经验证排队、冲击、滑点和部分成交。

每个提及品种的订单表示其完整目标。旧合约自动补零，任一腿被拒则整组不成交；保证金检查按品种字母序。日终超预算触发下一开盘处置，终端最后开盘清仓；清仓受阻将标记，过期持仓或缺少持仓估值直接报错。不是交易所或经纪商日内强平过程的精确复制。

```bash
cfl execution-ledger --bars data/market/execution.csv --calendar data/market/calendar.csv --orders data/market/orders.csv --output reports/ledger/example --initial-cash 10000000 --max-margin-fraction 0.5
```

输出目录必须不存在，含 daily、trades、rejected、holdings 和输入清单。daily 同时记录隔夜/日内盈亏、费用、权益、保证金、可用现金、对账误差及终端未清仓标志；所有来源仍需外部核验。

完整方法见[逐合约方法](contract_methodology.md)，当前真实输入是否就绪见[来源核查](source_readiness.md)。

## 可选容量与分阶段执行

资金账可额外接收`max_trade_lots`与`capacity_available_at`，二者必须同时存在。上限为有限非负整数，容量时钟须严格早于申报截止；未提供截止时须严格早于执行时刻。默认`atomic`策略超量整组拒绝。

Python接口`settle_portfolio(..., capacity_policy="staged")`启用分日补执行，新信号替换同品种旧目标。先平旧约再开新约，开平/反转和终端清仓均受容量约束，期末未平显式标记。容量事件记录期望、尝试、实际成交、剩余量及限制/拒绝原因，保存在`ExecutionResult.capacity_events`。原未提供容量的默认账保持原行为。

CLI研究入口为`cfl capacity-study`，见[容量协议](capacity_methodology.md)。日量预算不等于真实开盘盘口容量，未完成账不能展示为已清算收益。
