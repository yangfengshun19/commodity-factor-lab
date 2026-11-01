# 文献与实现决策

检索截至 2026-10-30。优先采用作者、期刊、官方文档与交易所来源。阅读范围分别注明；未获得全文的项目不声称完成论文复现。“受到方法启发”与“复现了论文发现”严格区分。v0.11参考信息预测的最新方法核查与阅读范围见[该实验的方法说明](nbs_prediction_methodology.md)。

| 来源与阅读范围 | 对本项目的约束或启发 | 当前落地 |
|---|---|---|
| Qian, Jiang & Liu (2025), *Factor Momentum in Commodity Futures Markets*, Journal of Futures Markets 45(11), 1934–1969。[期刊摘要](https://onlinelibrary.wiley.com/doi/abs/10.1002/fut.70022) | 美英长历史中的因子择时收益会受到高换手成本侵蚀，不能从毛收益直接推投资价值 | 固定主成本并完整报告 0/5/10/20 bps；未声称复制其因子择时结果 |
| Kosch & Forsberg (2026-09-10), *Seasonal Trading in Commodity Futures*。[预印本摘要](https://arxiv.org/abs/2609.12227) | 统一交易约束、基准和多重校正后，季节性模型未建立稳健超额表现；摘要也说明其 MEB 的条件性 | 固定基准、滚动区间、完整检验族。这里采用配对移动分块区间，未实现或冒充 MEB |
| Jacquier, Muhle-Karbe & Mulligan (2025, v3 2025-12-01), *In-Sample and Out-of-Sample Sharpe Ratios for Linear Predictive Models*。[预印本摘要](https://arxiv.org/abs/2501.03938) | 复杂策略和大量弱信号会降低样本外复制比例；更多训练历史有帮助 | 主协议先保留少量经济定义明确的简单特征，不在短样本上添加神经网络或大规模参数搜索；尚未复制文中公式 |
| Harvey, Liu & Zhu (2016), *… and the Cross-Section of Expected Returns*。[作者工作论文记录](https://www.nber.org/papers/w20592) | 探索大量因子后，单项常规显著性不足以支撑发现 | 显式申报 v0.2 的 15 个及 v0.3 的三个主假设，并承认不能覆盖原研究未完整登记的全部搜索 |
| Benjamini & Yekutieli (2001), *The Control of the False Discovery Rate in Multiple Testing under Dependency*。[原文 DOI](https://doi.org/10.1214/aos/1013699998) | 任意依赖下校正需要调和系数，且仍以有效 p 值为前提 | BY 实现纳入所有预定项；HAC 是近似推断，未宣称有限样本保证 |
| statsmodels 官方 [cov_hac 文档](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html) | Bartlett 核、日历间距及有限样本修正需要明确 | 收益均值及多基准回归 HAC 与该实现交叉测试；缺失日期保留时隙 |
| Feng Yulin、Tang Ke、Kang Wenjin (2022), The Pricing Mechanism in Chinese Commodity Futures Markets。[《金融研究》原文页](https://www.jryj.org.cn/CN/abstract/abstract1131.shtml) | 国内商品需要市场、基差和基差动量等经济对照，不能靠均线排名直接宣称新 Alpha | 已增加 carry、基差动量及仓单的增量预测诊断；未声称完成该定价模型复制 |
| 上海国际能源交易中心 2026 年结算规则。[交易所原文](https://www.shfe.cn/regulation/ineregulation/businessmethods/settlement/202606/t20260622_832188.html) | 实际期货盈亏涉及结算、保证金和真实合约，连续价百分比不等于完整交易模型 | 已实现逐日结算与保证金现金账，并完成合成演示；真实执行输入仍待认证 |
| AKShare 官方 [新浪日线适配源码](https://github.com/akfamily/akshare/blob/main/akshare/futures/futures_zh_sina.py) | `代码0` 是连续价格接口，来源和快照要保留 | 独立公共读取器保存 JSONP 响应并只解析数据；不执行响应中的脚本，不将其认证为可交易合约 |

## 后续文献如何进入项目

每个新方向先写经济机制、信息可用时点、可反驳的基准和评价标准，再获得数据。复现论文时需要同口径的合约构造、滚动估计和成本，而不只是借用策略名称。预印本经过正式审稿或更新后应记录新版本，不能把“更新”自动当作“更正确”。

本项目当前没有复现上述论文的全部实证结果，也没有把海外长样本结论直接外推到中国短样本。

## 因子选用补充核查（2026-10-29）

- Boons & Porras Prado (2019), *Basis-Momentum*, Journal of Finance 74(1), 239–279。[期刊原文页](https://doi.org/10.1111/jofi.12738)。核验摘要及公开因子定义：近远月策略历史收益差与现货/期货单日收益差必须区分；文献结论不能替代本地复制。
- Asness, Moskowitz & Pedersen (2013), *Value and Momentum Everywhere*。[作者托管论文](https://spinup-000d1a-wp-offload-media.s3.amazonaws.com/faculty/wp-content/uploads/sites/3/2019/09/ValueandMomentumEverywhere.pdf)。核验商品价值构造段落：使用约五年历史现货价格相对当前价格的长期反转代理。价格也可形成价值代理；本项目半年/一年正收益的“价值”命名仍不恰当。
- Gu, Kang, Lou & Tang, *Relative Basis and the Expected Returns of Commodity Futures*。[2025 AEA 会议托管稿](https://www.aeaweb.org/conference/2025/program/paper/yzD5hdsG)，文件标注草稿日期为 2024-08。核验摘要和定义/机制部分：相对基差比较近端与远端期限结构斜率，试图区分短期库存稀缺与持久成本。未完整复现，不能称为本项目已实现的新因子。
- Luo & Xue (2026), *Spot-Based Basis and Basis Momentum in Commodity Futures Markets*。[SSRN 作者工作论文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6546878)。核验公开索引中的作者摘要与元数据，直接详情访问返回 403，未读取全文。作者提出使用真实现货价格构造信号；仅作为后续核查方向，不作为已确定有效的依据。

本轮纠正了此前中国市场研究条目中误写的作者姓名；采用期刊英文元数据列示。具体因子评价见[因子审查](factor_review.md)。

版本说明：上表 0/5/10/20 bps、滚动价格对照及分块区间属于 v0.2 复核。v0.3 主研究是同合约预测诊断，不输出真实收益；合成现金账逐笔使用明确开平仓费率和每手费用。基差动量与相对基差采用本地固定代理，尚未严格复制相应论文全部口径。

## v0.5 进一步核查

补充读取 [FMG Discussion Paper 942 全文](https://www.fmg.ac.uk/sites/default/files/2026-01/DP942.pdf)摘要与第 2 节：封面 2025-11、文内草稿 2024-07，相对基差为相邻年化对数斜率差。本地公式结构相符，但市场、频率、到期筛选和动量窗口不同；美国分类交易者套保压力也不能以经纪会员排名替代。本地登记检验未获得稳健增量证据，见[新增结果](other_factor_results.md)。

另核查 [AKShare 官方现期源码](https://raw.githubusercontent.com/akfamily/akshare/main/akshare/futures/futures_basis.py)中的生意社来源、单位与规格提醒；核查上期所[成交持仓公布标准](https://www.shfe.com.cn/reports/tradedata/dailyandweeklydata/dailyranking/decl/202401/t20240117_331641.html)以解释排名披露门槛，未把现行页面当成全历史有效规则。
