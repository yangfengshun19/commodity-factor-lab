# 目录迁移与第一阶段重构

2026 年 10 月完成第一阶段整理，原有 939 个文件均已记录原路径、目标路径、字节数与 SHA256；清单位于本地 `private/migration_manifest.csv`。原 CSV、图片、文本报告及论文材料不进行内容改写。

| 原内容 | 新位置 |
|---|---|
| 根目录七份采集脚本 | `research/legacy/collectors/` |
| 七类文件夹中的逐品种 CSV | `data/raw/<family>/` |
| 七类旧数据处理和分析代码 | `research/legacy/analysis/<family>/` |
| 旧全品种面板和分板块面板 | `data/archive/` |
| `00分项分析` 的分析代码 | `research/legacy/sector_analysis/` |
| 历史图表、回归报告和汇总结果 | `reports/archive/` |
| 数据库实验 | `research/legacy/database/` |
| 论文正文、答辩和签字等毕业材料 | `private/thesis/` |
| 原程序说明文档 | `private/notes/` |

归档数据库脚本为消除公开凭据而改写了连接参数，并移除了禁用 TLS 证书验证的语句；这些改动作为特例记录在迁移验证脚本中。其余原文件通过哈希核对内容完整性。

旧面板和旧回测保留作研究记录。新程序从逐品种快照重新计算，避免继承跨品种标签位移、组合敞口和时间口径问题。新的入口统一为 `cfl demo / audit / run / build-price`，不再依赖工作目录下散落脚本和旧电脑盘符。

可以运行 `python scripts/verify_migration.py` 校验保留文件。`scripts/reorganize_workspace.py` 是本次一次性迁移工具，正常研究无需再执行。
