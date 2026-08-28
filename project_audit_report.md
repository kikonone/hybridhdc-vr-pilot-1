# 项目全阶段完成度与合理性审计报告

审计日期：2026-08-21  
审计范围：项目计划、Phase 00-10 文档、数据合同、代码、Notebook、配置、结果、统计分析、冻结清单、测试、依赖、安全与 Git 交付状态。  
审计方法：计划交付物、实际 artifact、可执行验证三角校验；未重新训练模型，未修改冻结预测。

## 1. 总体结论

**当前项目不应标记为“全部完成”或“可最终提交”。**

- 科学实验主体（Phase 00-09）完成度约 **88%**：数据、双任务建模、HDC 变体、模态、融合、捷径、缺失模态和 LOSO 均有实质结果与较完整证据链。
- 完整项目（含 Phase 10、文档、版本控制、可复现交付）完成度约 **68%**。
- 当前验收结论：**CONDITIONAL / NOT READY FOR FINAL SUBMISSION**。
- 主要阻断项：Phase 10 未完成；项目几乎没有 Git 版本历史；状态文档严重过期；生命周期测试会重写冻结 artifact；Phase 06/09 的冻结验证与后续阶段共存逻辑存在缺陷。

项目的科学设计总体合理，尤其是 subject-wise GroupKFold、折内预处理、Primary/auxiliary 数据角色、OOF 保存、受试者级 bootstrap/Wilcoxon/Holm 与主张边界。但工程交付和冻结生命周期设计尚不足以支撑“可复现、可移交、可答辩演示”的最终状态。

## 2. 阶段审计矩阵

评分口径：交付物 40%、正确性/合理性 25%、可执行验证 20%、可复现与文档一致性 15%。评分为审计估计，不替代正式验收签字。

| 阶段 | 审计完成度 | 状态 | 核心证据与结论 |
|---|---:|---|---|
| Phase 00 项目设置 | 90% | 基本完成 | 目录、数据清单、安全规则、执行 Notebook 齐全；但根级项目状态未随实验推进更新，Git 未承载项目内容。 |
| Phase 01 原始数据/模态审计 | 90% | 基本完成 | 9,003 个原始文件、487 runs、35 subjects；0 unreadable；908 unknown files 已显式保留。缺少自动化单元测试，但有验证报告。 |
| Phase 02 特征提取 | 90% | 基本完成 | 487 runs、1,247 个非标识特征、599,569 long-table rows；失败文件为 0。control-input 不可用被诚实声明。 |
| Phase 03 双任务数据与折 | 95% | 完成 | 419 runs、35 subjects、1,176 Primary 特征；`target_class=0..3`、`target_score=1..4`；固定 5 折哈希与当前文件一致；泄漏审计通过。 |
| Phase 04A 传统分类 | 90% | 实验完成 | 6 个正式分类器加 Dummy OOF；Gradient Boosting Macro-F1=0.935608。主 Notebook 仅 14/34 代码单元带执行计数，执行证据不够整洁。 |
| Phase 04B 传统回归 | 84% | 实验完成/工程需清理 | 8 个回归变体；Gradient Boosting bounded MAE=0.107486。主 Notebook 20/24 单元执行；`logs/elastic_recovery_v1.py` 存在真实语法错误。 |
| Phase 05 基础双输出 HDC | 90% | 完成并冻结 | 4 dimensions × 5 seeds × 5 folds；20 个完整 OOF 配置；分类、similarity regression、Ridge readout 齐全。峰值内存和合规训练时间缺失，已明确不伪造。 |
| Phase 06 HDC 变体筛选 | 80% | 科学结果完成/选择证据有限 | 四种 HDC 与两类回归头齐全；选定 Hybrid d=5000 分类、Common Ridge d=10000 回归。选择规则在 final-confirmation artifact 已存在后补充，不能排除选择乐观偏差。初始化测试与最终态冲突。 |
| Phase 07 单模态贡献 | 90% | 完成并冻结 | 5 模态、250 runs、受试者级统计；flight parameter 为分类/回归最佳单模态。README 的中间状态段落未完全同步最终状态。 |
| Phase 08 融合与捷径 | 90% | 完成并冻结 | 370 runs、10,894 canonical OOF rows；统计与限制完整。飞行模态增量显著，但无法排除 difficulty-adjacent task structure；不能作因果或跨场景泛化解释。 |
| Phase 09 鲁棒性/泛化 | 85% | 结果完成，生命周期验证有缺陷 | 720 runs、30,168 raw rows、10,056 canonical rows、35-subject LOSO；内部重冻结恢复后所有 artifact/hash 检查通过。验证器却把 Phase 10 目录“存在”当作 Phase 09 已执行 Phase 10，导致后续阶段创建后永久 FAIL。 |
| Phase 10 最终汇总/UI | 25% | **未完成** | 仅完成 initialization/inventory preflight。最终预测库、统计包、论文表/图目录为空；无 final manifest、final report、UI。把 UI 标为“延期且不影响完成”与最新 Phase 10 修订版直接冲突。 |

## 3. 最高优先级问题

### P0-1：Phase 10 只做了初始化，不满足最新计划完成标准

证据：

- `final_prediction_library/`、`final_statistics_bundle/`、`final_paper_tables/`、`final_paper_figures/`、`reports/` 均为空。
- 不存在 `final_project_manifest.json`、`final_experiment_report.md`、`best_hdc_demo_ui/app.py`。
- 当前 `phase10_deliverables_plan.json` 明确把 final synthesis、artifact merging、UI 列为 deferred。
- Phase 10 UI 修订说明明确规定 Phase 10 应引用 UI 修订版，UI 是完成标准之一；现有 Phase 10 audit 却把 `UI_DEFERRED_BY_USER_NOT_EXECUTED` 判为 PASS。

影响：无法形成最终论文证据包、答辩演示、统一数字来源和最终可复现清单。

整改：完成只读最终汇总和 UI；若确需取消 UI，必须先正式修订最新计划和完成标准，不能仅在 Phase 10 内部 audit 自行降低标准。

### P0-2：项目没有可交付的版本控制历史

证据：Git 仅跟踪三个现已删除的 `data/salary*.csv/xtra_info.csv`，当前代码、配置、文档、结果均为 untracked；仅有 1 个无关提交。工作区约 17,427 文件、44.9 GiB。

影响：无法证明代码/配置版本、无法安全协作、无法回滚、无法从仓库复现项目。

整改：建立干净的 Git 交付边界；跟踪代码、配置、轻量 manifests、报告和必要小型结果；继续忽略受许可限制的原始数据、大型检查点和可再生中间产物；为大文件给出下载/校验说明，必要时使用 Git LFS 或独立 artifact release。

### P0-3：测试会破坏冻结态

证据：

- `test_phase09_contract.py` 在 `setUpClass` 直接调用会写文件的 `run_freeze()`。
- Phase 06 preflight 测试运行时也重写初始化 audit/config。
- 全仓测试后 Phase 09 已完成执行清单被退回 `AUTHORIZED_NOT_EXECUTED`，Phase 06 final manifest 出现初始化元数据哈希不一致。

影响：运行测试本身会改变受保护证据，破坏可复现性并产生伪失败。

整改：所有测试改为在 `tmp_path`/临时复制中运行；生产冻结目录必须只读；把初始化态、执行态、冻结态测试拆开；CI 首先校验工作树无修改，测试后再次校验 artifact hash 和 `git diff --exit-code`。

### P0-4：状态与权威计划不一致

根 README 和 `EXPERIMENT_STATUS.md` 仍称只完成到 Phase 03，并使用旧的单任务目录名称；`classification_regression_latest_plan.md` 仍写“立即下一步 Phase 04”，而实际 Phase 09 已冻结。Phase 04B/05/06/07/08 README 同时保留初始化状态与最终状态，机器和人员都难以判断唯一当前状态。

整改：建立单一权威 `PROJECT_STATUS.json/md`，每阶段只保留 `NOT_STARTED/INITIALIZED/EXECUTED/ANALYZED/FROZEN` 一个当前状态，并引用 freeze hash；根 README 由该状态自动生成或至少在每次冻结后更新。

## 4. 重要问题

### P1-1：Phase 06 的最终模型选择不是严格预注册

Amendment v2 在 final-confirmation artifacts 已存在后定义。虽然 selector 被限制为 inner-CV 和无标签效率证据、outer evidence 也提前封存，但流程仍不能证明制定规则时未受已见结果影响。报告已经正确承认 selection-induced optimism 无法排除。

建议：论文中使用“post-freeze, inner-evidence-only amendment”，不要写“fully preregistered selection”；以 Phase 09 LOSO 作为较独立的确认，并把选择偏差列入 Limitations。

### P1-2：飞行参数高性能可能含任务结构捷径

Phase 07/08/09 一致显示 flight parameters 贡献最大；去除 flight 后分类 Macro-F1 降幅约 0.51、回归 MAE 劣化约 0.59-0.72。当前只有 323 个 provenance 标为 behavioral response 的特征，缺少 scenario、route、task-template、configuration 标识，不能区分一般化飞行行为与难度邻近任务结构。

建议：论文只主张“predictive dependence”；不得主张生理因果、跨场景泛化或部署级 workload detection。未来采集显式 scenario/route metadata 并做跨场景留出。

### P1-3：可复现环境未锁定

`requirements.txt` 只有包名，无版本、Python ABI、操作系统和锁文件；`pip check` 只证明当前环境内部依赖没有已知冲突，不证明他人可复现。`pip-audit` 未安装，因此未完成依赖漏洞扫描。

建议：增加 `requirements-lock.txt` 或 `environment.yml`/`uv.lock`，记录 Python 3.12.7 和完整包版本；在隔离环境执行一次从零安装和 read-only verification。

### P1-4：Notebook 与脚本组织不一致

Phase 04A/04B 大量正式执行脚本放在 `logs/`，并混入 backup 和语法错误恢复脚本；主 Notebook 不是所有代码单元都执行。Phase 10 使用 `exec(compile(cell_source))` 执行本地 Notebook 单元，适合受信任仓库，但不应用于不可信 Notebook。

建议：正式代码移入 `scripts/`/`src/`，历史失败文件改为 `.txt` 或归档到非执行目录；为 canonical Notebook 提供一次 fresh-kernel 全量执行证据；声明 Notebook 必须受信任且哈希固定。

## 5. 科学合理性结论

### 做得合理的部分

- 标签边界清晰：分类与回归都只是 difficulty-induced workload proxy，不冒充直接心理负荷真值。
- 固定受试者级五折，分类/回归/HDC/传统模型共享 folds；当前数据和 fold SHA-256 与记录一致。
- 缺失填补、方差过滤、标准化、特征选择和调参均设计在训练折内；代码抽查与多数测试支持该结论。
- Primary 不含 performance；with-performance 和 performance-only 仅作辅助捷径分析。
- 保存 fold/seed/sample 级预测，统计单位主要为 35 名 subject，而不是误把 5 folds 当独立样本。
- 报告负结果和限制：HDC 没有击败传统模型；similarity regression 坍缩到中间等级；跨场景验证不可行；不把非显著解释为等效。

### 结果解释

- 最佳传统分类器 Gradient Boosting：OOF Macro-F1 0.935608；LOSO Macro-F1 约 0.9570。
- 选定 HDC 分类器 Hybrid d=5000：五种子 OOF Macro-F1 0.822309±0.032325；LOSO Macro-F1 约 0.8584。
- 最佳传统回归器 Gradient Boosting：bounded OOF MAE 0.107486。
- 选定 HDC Common Ridge d=10000：五种子 bounded MAE 0.276390±0.006419。
- 结论应是：HDC 提供可用且稳定的双任务表征，但本数据上预测性能明显不及 Gradient Boosting；若论文强调 HDC 价值，应依赖经完整测量的效率/内存优势，而不是性能领先。当前训练时间和峰值内存证据仍不完整，不能夸大效率优势。

## 6. 验证报告

```text
Dataset/Folds: PASS
  Primary rows=419, subjects=35, folds=5
  target_class={0,1,2,3}, target_score={1,2,3,4}
  Primary SHA-256=0a2aef89...
  Fold SHA-256=e4dc943a...

Tests: FAIL
  125 collected; 119 passed; 3 failed; 3 errors; 17 warnings
  Main causes: stale lifecycle assertions and tests mutating frozen artifacts

Python syntax: FAIL
  logs/elastic_recovery_v1.py: unterminated string literal at line 28
  Remaining 135 Python files parsed successfully after excluding the failed recovery pair

Notebook health: PARTIAL PASS
  All canonical notebooks contain no saved error output
  Phase 04A main notebook 14/34 code cells executed
  Phase 04B main notebook 20/24 code cells executed

Dependencies: PARTIAL PASS
  pip check: PASS
  versions unpinned; pip-audit unavailable

Secrets scan: PASS
  no hard-coded API key/password/token pattern found in project Python/JSON/YAML

Phase 09 re-freeze after audit-side-effect: PASS
  720 completed runs, 30,168 raw rows, 10,056 canonical rows
  manifest hashes and protected artifacts match
  standalone verifier still returns FAIL solely because Phase 10 directory exists

Git delivery: FAIL
  project source/artifacts untracked; only three deleted unrelated CSVs tracked

Overall: NOT READY FOR FINAL SUBMISSION
```

## 7. 建议整改顺序与验收门

1. 修复测试隔离：所有会写 artifact 的测试改到临时目录；修复 Phase 06 初始化态断言和 Phase 09 “Phase 10 目录存在即失败”逻辑。
2. 恢复并重新签署 Phase 06 初始化元数据清单，确认科学结果/OOF/checkpoint 哈希未变化；再跑全阶段只读 hash verification。
3. 清理 Phase 04B 语法错误脚本、backup 与 `logs/` 中正式代码；完成 canonical Notebook fresh-kernel 执行。
4. 完成 Phase 10 核心汇总、RQ 映射、最终论文表图、final manifest/report 和只读 UI；跨阶段数字核对必须 PASS。
5. 更新根 README、EXPERIMENT_STATUS 和权威计划指针，消除旧阶段名和旧“下一步”。
6. 建立 Git/LFS/artifact release 交付边界和锁定环境，在全新目录按 README 完成一次复现。
7. 最终验收必须同时满足：测试全绿、语法全绿、依赖/安全扫描完成、所有 freeze manifest 零哈希差异、Phase 10 全交付、Git 工作树可解释、论文主张不越界。

## 8. 审计过程中产生的变更说明

- 新增本报告、`task_plan.md` 和 `notes.md`。
- 全仓测试暴露出写入冻结目录的副作用，并实际重写 Phase 06/09 初始化元数据。
- Phase 09 已通过项目自带无重训流程重新冻结；预测、checkpoint 和统计 artifact 未重新生成，内部 manifest/hash 检查恢复 PASS。旧 freeze/manifest 备份保存在 `tmp/phase09_audit_restore_20260821_1438/`。
- Phase 06 的核心结果、OOF 和选模文件未变化，但 6 个初始化 audit/config 与既有 final manifest 的哈希仍不一致；必须按整改第 2 项处理，不能隐瞒或直接覆盖历史证据。
