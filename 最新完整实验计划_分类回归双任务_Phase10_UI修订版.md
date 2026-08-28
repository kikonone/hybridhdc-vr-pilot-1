# 最新完整实验计划：多模态 HDC 分类与回归双任务

## 1. 计划定位

### 1.1 建议论文题目

中文题目：**基于多模态生理与飞行数据的 VR 飞行工作负荷代理分类与回归：超维计算方法研究**

英文题目：**Hyperdimensional Computing-Based Classification and Regression of Workload Proxies Using Multimodal Physiological and Flight Data in VR Piloting Tasks**

### 1.2 核心任务

本研究使用同一套 run-level 多模态特征，完成两个并列主任务：

1. 分类任务：预测 Difficulty Level 1-4 对应的四个离散工作负荷代理等级。
2. 回归任务：预测范围为 1-4 的连续工作负荷代理分数。

两个任务共享数据、外层受试者划分、预处理和 HDC 编码，但使用不同的输出头和评价指标。

### 1.3 解释边界

Difficulty Level 1-4 是实验任务难度条件，只能解释为 task-difficulty-induced workload proxy。回归任务输出的是有界代理分数，不是直接测得的连续心理负荷、压力、临床状态或主观认知负荷。

## 2. 当前项目基础

### 2.1 已完成阶段

- Phase 00：项目结构、数据清单和安全规则，已完成。
- Phase 01：原始数据和模态审计，已完成。
- Phase 02：完整 run-level 多模态特征提取，已完成。
- Phase 03：四等级数据集构建、标签映射和泄漏审计，已完成。

### 2.2 当前数据规模

- 有效样本：419 个 run。
- 受试者：35 人。
- 四类样本数：104、106、104、105，类别基本均衡。
- 主数据集：1,176 个非 performance 特征。
- 辅助数据集：1,235 个包含 performance 的特征。
- Performance 特征：59 个。
- 当前明确可用模态：生理信号、眼动、头部运动、X-Plane 飞行状态、performance 和待确认的 torso accelerometer 特征。
- 明确 control-input 特征：当前未提取到，不能虚构该模态结果。

### 2.3 数据决策门

Phase 04 开始前审查 42 个 unknown/torso accelerometer 特征：

- 如果来源和物理含义得到确认，将其重命名为 body-movement 模态并纳入主实验。
- 如果无法确认，主结果排除这些特征，并将“包含 unknown 特征”作为敏感性分析。

## 3. 研究问题

### RQ1：双任务总体性能

多模态特征能否在未见受试者上同时支持可靠的四分类和有界分数回归？

### RQ2：HDC 与传统模型比较

HDC 在分类性能、回归误差、训练时间、推理时间和模型内存方面，与传统分类器和回归器相比表现如何？

### RQ3：HDC 变体比较

Vanilla Prototype HDC、OnlineHD-style HDC、Multi-centroid HDC 和 HDC+OnlineHD Hybrid 中，哪一种具有更好的性能、稳定性和效率平衡？

### RQ4：回归输出的增量价值

HDC 回归输出是否提供了超出四分类概率或原型相似度的有效等级距离信息？分类输出与回归输出的一致性如何？

### RQ5：模态贡献与融合

多模态是否优于单模态？生理、眼动、头部运动和飞行状态中，哪些模态对分类与回归贡献最大？

### RQ6：泄漏、缺失模态与泛化

Performance 特征会带来多大程度的捷径学习？传感器模态缺失时，不同模型的性能如何下降？最终模型在 LOSO 下是否稳定？

## 4. 研究假设

- H1：不含 performance 的完整多模态特征，在分类 Macro-F1 和回归 MAE 上优于最佳单模态设置。
- H2：HDC 在准确性上至少具有竞争力，并在训练、推理或内存中的至少一个维度表现出优势。
- H3：HDC 回归头能够保持 Difficulty Level 顺序，且不会简单坍缩到总体均值附近。
- H4：加入 performance 特征会明显提高结果，但这种提升部分来自与任务难度直接相关的捷径信息。
- H5：经过多模态训练的 HDC 在单一模态缺失时，性能下降不高于主要传统基线。

这些假设允许被否证。即使 HDC 没有全面获胜，只要验证严格、误差分析完整，负结果仍可形成合格论文结论。

## 5. 数据与标签合同

### 5.1 样本单位

每一行代表一个 `subject-session-run-difficulty` 样本。

### 5.2 分类标签

```text
target_class = difficulty_level - 1
Difficulty 1, 2, 3, 4 -> Class 0, 1, 2, 3
```

### 5.3 回归标签

```text
target_score = difficulty_level
Observed values = 1.0, 2.0, 3.0, 4.0
```

保存两种回归输出：

- `prediction_raw`：模型原始连续输出，用于诊断模型是否超出边界或向均值坍缩。
- `prediction_bounded`：裁剪到 `[1, 4]` 的输出，用于主要有界指标。

不得在训练前把连续预测四舍五入。只有在跨任务一致性分析时，才生成 `prediction_rounded`。

### 5.4 输入数据版本

1. Primary：without-performance，全论文主结果使用。
2. Auxiliary A：with-performance，用于估计上限和捷径风险。
3. Auxiliary B：performance-only，用于检验任务结果指标本身能达到多高预测性能。
4. Sensitivity：unknown/body-movement 特征包含与排除两个版本。

## 6. 验证与防泄漏设计

### 6.1 固定外层验证

- 主验证：固定 5 折 `GroupKFold`，group 为 `subject_id`。
- 同一受试者的所有 run 必须出现在同一折。
- 分类、回归、传统模型和 HDC 必须复用完全相同的外层折。
- 保存 `fold_assignments.csv`，后续不得因模型表现重新划分。

### 6.2 内层调参

- 每个外层训练集内部使用 3 折 subject-wise GroupKFold。
- 超参数选择、特征选择和阈值选择只能访问内层训练/验证数据。
- 测试折只在最终评估时使用一次。

### 6.3 LOSO 补充验证

LOSO 只对以下最终模型运行：

- 最佳传统分类器。
- 最佳传统回归器。
- 最佳双输出 HDC。

### 6.4 随机种子

- 数据划分固定。
- 涉及随机初始化、随机森林、HDC hypervector 和聚类的模型，最终使用 5 个预先定义的随机种子，例如 42-46。
- 报告均值、标准差和每个种子的原始结果。

## 7. 折内预处理

所有步骤均封装在训练折内部：

1. 删除标识列：`subject_id`、`session_id`、`run_id`、`run_key`、标签列。
2. 缺失值：训练折中位数填补，并保留缺失指示变量。
3. 零方差过滤：仅用训练折拟合。
4. 标准化：LR、SVM、KNN、SVR、线性回归器和 HDC 使用训练折参数。
5. 特征选择：在训练折内比较 `k = 50, 100, 200, all`；树模型可同时保留全特征方案。
6. 类别权重：只根据训练折确定。
7. 回归目标不做全局变换；如尝试标准化目标，必须只在训练折拟合并在输出时逆变换。

## 8. 传统模型实验矩阵

### 8.1 分类基线

- DummyClassifier：多数类和分层随机基线。
- Logistic Regression。
- SVM：Linear 和 RBF。
- Random Forest。
- K-Nearest Neighbors。
- Gradient Boosting；XGBoost 仅在环境可用且不增加部署依赖风险时加入。

### 8.2 回归基线

- DummyRegressor：均值和中位数基线。
- Ridge Regression。
- Elastic Net。
- SVR：Linear 和 RBF。
- Random Forest Regressor。
- Gradient Boosting Regressor。

### 8.3 紧凑调参原则

使用小而可解释的网格，避免在 419 个样本上进行大规模搜索：

- 线性模型：正则强度。
- SVM/SVR：`C`、核函数、`gamma` 和回归 `epsilon`。
- Random Forest：树数量、最大深度和每次分裂特征数。
- KNN：邻居数和距离权重。
- Gradient Boosting：学习率、树数量和最大深度。

## 9. HDC 双输出模型设计

### 9.1 共享编码器

1. 连续特征通过 level encoding 或分箱映射到 value hypervector。
2. 每个特征具有独立的 identity hypervector。
3. 使用 binding 组合特征身份与数值。
4. 使用 bundling 形成 sample hypervector。
5. 统一使用 bipolar 或 binary 表示，并在同一实验中保持一致。

### 9.2 分类头

训练每个等级的类别原型 `C1-C4`，通过余弦相似度或汉明相似度选择最近原型。

### 9.3 回归头 A：相似度加权解码

对四个类别原型的相似度进行温度缩放：

```text
p_k = softmax(similarity(h, C_k) / T)
y_hat = sum(k * p_k), k = 1, 2, 3, 4
```

温度 `T` 只能在内层训练中选择。

### 9.4 回归头 B：正则化 hypervector readout

以 sample hypervector 为输入，训练 Ridge/regularized linear readout 输出连续分数。正则参数在内层 subject-wise 验证中选择。

### 9.5 P1 承诺的四种 HDC

- Vanilla Prototype HDC：单类别原型。
- OnlineHD-style HDC：基于错误或低置信样本更新原型。
- Multi-centroid HDC：每类维护多个原型，处理受试者异质性。
- HDC+OnlineHD Hybrid：先构建稳定原型或多中心，再进行 OnlineHD-style 更新。

所有四种模型先在 Primary 数据版本上完成分类筛选。对兼容模型应用相似度回归解码；正则化 readout 至少在 Vanilla 和最终最佳 HDC 上完成。

### 9.6 HDC 参数筛选

采用两阶段筛选，控制实验规模：

1. 快速筛选：dimension 为 2,000/5,000，levels 为 21/51，单种子。
2. 最终确认：保留 Top 配置，比较 dimension 1,000/2,000/5,000/10,000，并使用 5 个随机种子。

Multi-centroid 的中心数优先比较 2 和 3，不允许根据测试折决定中心数。

## 10. 模态与融合实验

### 10.1 单模态设置

- Physiological：ECG、EDA、EMG、Respiration。
- Eye-tracking。
- Head movement。
- Flight-state/X-Plane。
- Body movement：仅在 unknown 特征来源确认后使用。
- Performance-only：仅用于捷径分析。

### 10.2 多模态设置

- Full multimodal without performance：主设置。
- Full multimodal with performance：辅助上限。
- Physiological + Eye。
- Physiological + Eye + Head。
- Physiological + Eye + Head + Flight-state。

### 10.3 融合策略

- Early Fusion：主结果，直接拼接折内处理后的特征。
- Late Fusion：可选，只对选定模型运行。
- HDC modality-aware binding：可选，在核心结果完成后运行。

不对所有模型与所有模态组合做完整笛卡尔积。先在 Primary 数据上筛选模型，再让最佳传统分类器、最佳传统回归器和最佳双输出 HDC 进入模态实验。

## 11. 鲁棒性实验

### 11.1 缺失模态

从完整多模态测试样本中分别移除：

- 生理模态。
- 眼动模态。
- 头部运动模态。
- 飞行状态模态。

比较分类 Macro-F1 下降量和回归 MAE 增加量。

### 11.2 噪声实验

仅在时间允许时，对标准化特征加入 5%、10%、20% 标准差的高斯噪声。噪声只能加入测试折或按照预先定义的训练增强协议实施，两种设置不得混淆。

### 11.3 数据量实验

可选比较 20%、40%、60%、80%、100% 训练受试者数据，评估小样本学习曲线。

## 12. 评价指标

### 12.1 分类主指标

- Primary endpoint：Macro-F1。
- Balanced Accuracy。
- Accuracy。
- Weighted-F1。
- Per-class Recall。
- Confusion Matrix。

### 12.2 回归主指标

- Primary endpoint：MAE。
- RMSE。
- Spearman rank correlation。
- `R²` 作为补充指标，不作为主结论，因为样本较少且标签只有四个观测值。

### 12.3 有序错误指标

- Adjacent Accuracy：`abs(round(y_hat) - y) <= 1`。
- Severe Error Rate：`abs(round(y_hat) - y) >= 2`。
- Quadratic Weighted Kappa。
- Rounded Regression Macro-F1：把有界回归结果四舍五入后与分类结果比较。

### 12.4 双任务一致性

- 分类预测与回归四舍五入等级的一致率。
- `abs((predicted_class + 1) - prediction_bounded)`。
- 分类正确但回归误差较大的样本。
- 回归接近真实值但分类错误的样本。
- 分类置信度、HDC similarity margin 与回归不确定性的关系。

### 12.5 工程指标

- 训练总时间。
- 单样本推理时间。
- 峰值内存和模型文件大小。
- HDC 原型数量和 hypervector 维度。

## 13. 统计分析

### 13.1 原始证据保存

每个模型必须保存：

- 外层折级结果。
- 每个样本的 out-of-fold 预测。
- `subject_id`、fold、真实标签、分类预测、分类分数、回归原始/有界预测。
- 训练时间、推理时间、种子和完整配置。

### 13.2 置信区间

- 对最终模型使用受试者级 bootstrap，建议 2,000 次重采样。
- 报告 Macro-F1、Balanced Accuracy、MAE、RMSE 和 Severe Error Rate 的 95% 置信区间。

### 13.3 模型比较

- 多模型整体比较：在受试者级指标上使用 Friedman 检验。
- 预先指定的两两比较：Wilcoxon signed-rank。
- 多重比较：Holm 校正。
- 同时报告效应量，如 rank-biserial correlation，不只报告 p 值。

避免用 5 个外层折直接声称强统计显著性；主要统计单位应为受试者或预先定义的成对 out-of-fold 误差。

## 14. 误差与解释分析

- 类别级：混淆矩阵、Per-class Recall、Level 1 与 Level 4 的严重混淆。
- 回归级：残差分布、均值坍缩、边界超出、各真实等级的 MAE。
- 受试者级：每位受试者的 Macro-F1、MAE 和 Severe Error Rate。
- 模态级：去除某模态后的性能变化。
- 捷径级：without-performance、with-performance 和 performance-only 的差异。
- HDC 级：原型相似度、margin、中心分布和错误样本距离。

解释性分析只解释模型行为，不能把 feature importance 或 prototype similarity 直接解释为生理因果关系。

## 15. 阶段实施计划与交付物

### Phase 00：项目和数据安全，已完成

交付物：数据清单、项目结构、配置和只读数据规则。

### Phase 01：原始数据与模态审计，已完成

交付物：文件清单、run-level 模态可用性、难度分布和异常文件清单。

### Phase 02：多模态特征提取，已完成

交付物：487 个 run 的完整特征表、特征组、失败日志和提取报告。

### Phase 03：双任务数据集准备，需小幅更新

在现有数据集基础上增加：

- `target_class`。
- `target_score`。
- 固定外层 `fold_assignments.csv`。
- unknown/body-movement 特征审计结论。

### Phase 04A：传统分类基线

交付物：

- `classification_baseline_summary.csv`。
- `classification_oof_predictions.csv`。
- 最佳分类模型的混淆矩阵。
- 分类配置和运行日志。

### Phase 04B：传统回归基线

交付物：

- `regression_baseline_summary.csv`。
- `regression_oof_predictions.csv`。
- 残差图、真实值与预测值图。
- 回归配置和运行日志。

### Phase 05：基础双输出 HDC

交付物：

- Vanilla HDC 分类结果。
- 相似度回归结果。
- Ridge readout 结果。
- HDC 参数搜索和相似度诊断。

### Phase 06：HDC 变体筛选

交付物：

- 四种 P1 HDC 变体比较表。
- HDC 分类与回归头比较表。
- 性能-时间-内存 Pareto 图。
- 最佳分类 HDC 和最佳回归 HDC 配置。

### Phase 07：单模态贡献

交付物：单模态分类/回归结果、模态排名图和每模态误差分析。

### Phase 08：融合与捷径分析

交付物：融合结果、with/without performance 对照、performance-only 结果和捷径风险结论。

### Phase 09：鲁棒性与泛化

交付物：缺失模态曲线、选定模型 LOSO 结果和受试者级稳定性分析。

### Phase 10：最终汇总、可复现性与最优双任务HDC演示界面

英文名称：**Phase 10: Final Synthesis, Reproducibility and Best Dual-Task HDC Demonstration UI**

建议目录：`experiments/phase_10_final_synthesis_and_demo_ui`

建议 Notebook：`Phase_10_Final_Synthesis_and_Demo_UI.ipynb`

#### Phase 10 定位与 OnlineHD 顺序回放状态

```text
ONLINEHD SEQUENTIAL REPLAY:
OPTIONAL_NOT_EXECUTED
```

- OnlineHD 顺序回放不再是 Phase 10 必需交付物。
- 不执行该可选实验不影响论文实验完整性。
- 本修订不修改 Phase 05-09 已经完成的 OnlineHD 相关结果。
- UI 不替代任何科学实验，只是 Phase 00-09 冻结结果的只读展示层。
- 传统模型与其他 HDC variants 不在 UI 中展示，但必须继续保留在论文正文、表格和统计分析中，作为基线、模型选择和负结果证据。

#### Phase 10 必需任务

1. **汇总最终预测库**

   只读汇总 Phase 04-09 已冻结的 classification OOF predictions、regression OOF predictions、HDC seed-level predictions、canonical OOF predictions、missing-modality predictions 和 LOSO predictions。禁止重新训练、修改或重新生成预测。

2. **汇总全部统计分析包**

   汇总 subject-level bootstrap confidence intervals、Friedman tests、Wilcoxon signed-rank tests、Holm corrections、effect sizes、HDC seed stability、subject stability、missing-modality robustness 和 shortcut analysis。统计单位、方向和结论必须延续各冻结阶段合同。

3. **统一论文表格和图表**

   必须整理传统模型基线表、HDC 与传统模型比较表、四种 HDC variant 比较表、最优 HDC 分类与回归表、单模态贡献表、融合与 shortcut 分析表、missing-modality 表、LOSO 稳定性表，以及最终论文图表和统计附录。传统模型和其他 HDC variants 即使不进入 UI，也不得从论文证据链中删除。

4. **建立 RQ-实验-证据-结论映射**

   每个 RQ 至少记录 research question、supporting phase、dataset、selected evidence、primary metric、statistical result、supported conclusion、unsupported claim、limitation 和 corresponding table/figure。

5. **整理可复现运行说明**

   至少记录 Python 环境、package versions、frozen checksums、fold checksum、Phase 00-09 运行顺序、Notebook 索引、configs 索引、manifests 索引、predictions 索引、reports 索引和一键只读验证方法。

6. **完成跨阶段一致性与数字核对**

   必须核对：rows = 419、subjects = 35、Primary features = 1176、frozen outer folds = 5、classification target 一致、regression target 一致、模型名称一致、指标名称和方向一致、表格/图/报告/Notebook 数字一致、Phase 00-09 freeze integrity，以及所有论文结论不超过实验支持范围。

7. **创建最优双任务 HDC 只读展示 UI**

   UI 对外统一名称为 **Best Dual-Task HDC System**，建议使用 Streamlit，本地离线运行，并且只能读取冻结 artifact。

#### UI 展示模型合同

UI 只展示以下两个冻结组件：

- 分类组件：`HDC+OnlineHD Hybrid`，dimension = 5000，来源为 Phase 06 frozen best classification HDC，主指标为 Macro-F1。
- 回归组件：`COMMON_ENCODER_READOUT_BASELINE`，dimension = 10000，来源为 Phase 06 frozen best regression HDC，主指标为 bounded MAE。

UI 不得展示 Dummy、Ridge、Elastic Net、SVR、Random Forest、Gradient Boosting 或任何其他传统模型；也不得展示 Vanilla HDC、OnlineHD-style 候选、Multi-centroid 候选、四种 HDC variant 比较、超参数搜索过程或模型选择过程。

此限制仅适用于演示 UI。论文正文仍必须保留传统模型基线、四种 HDC variant 比较、模型选择证据、负结果和不显著结果。

#### UI 页面

1. **Project Overview**
   展示 419 modeling runs、35 subjects、1176 Primary features、5 modalities、classification/regression 双任务和 subject-wise evaluation。

2. **Best HDC Classification**
   展示 Macro-F1、Balanced Accuracy、Accuracy、Severe Error Rate、Per-class Recall、Confusion Matrix 和 95% subject bootstrap CI。

3. **Best HDC Regression**
   展示 bounded MAE、bounded RMSE、bounded R²、bounded Spearman、clipping rate 和 95% subject bootstrap CI。

4. **Frozen OOF Prediction Explorer**
   允许选择 anonymized subject、anonymized run、outer-fold 或 LOSO 结果；展示 true difficulty、predicted class、bounded predicted score、classification correctness、regression absolute error，以及冻结 artifact 中实际可用的 HDC similarity、score 或 margin。

5. **HDC Modality Contribution**
   展示最优 HDC 的 physiological、eye tracking、head movement、flight parameters、body movement 和 multimodal reference。

6. **HDC Fusion and Shortcut Evidence**
   展示 PE、PEH、PEHF、without-performance、with-performance auxiliary 和 performance-only auxiliary。所有 performance 相关结果必须显著标记为辅助 shortcut 分析。

7. **HDC Missing-Modality Robustness**
   展示 Full Primary、Missing Physiological、Missing Eye Tracking、Missing Head Movement、Missing Flight Parameters 和 Missing Body Movement。

8. **HDC LOSO Stability**
   展示 35 名匿名 subject 的分类结果、回归结果、subject-level 分布、bootstrap CI 和稳定性结论。

9. **Reproducibility and Limitations**
   展示 Primary checksum、frozen fold checksum、selected HDC config hashes、manifest 状态、Notebook persistence 和泛化限制。

#### UI 安全与审计边界

UI 必须满足：

- 本地运行、只读、可离线使用；
- 不写回实验目录，不修改任何冻结 artifact；
- 使用 `Subject 01` 至 `Subject 35` 等匿名编号，不显示可识别 `subject_id`；
- 所有数字直接来自冻结 CSV/JSON，且与论文表格完全一致；
- 不提供训练、调参、模型选择、预测修改或新数据正式诊断功能；
- 不声称实时认知负荷测量或部署级航空安全系统；
- UI 失败不得影响冻结实验、论文证据或可复现性包。

固定免责声明：

> This interface is a read-only demonstration of the frozen best dual-task HDC system and its audited out-of-fold results. It is not a deployment or real-time cognitive workload diagnostic system.

#### 不包含实时新数据预测

Phase 10 必需范围不包括 full-data retraining、deployment model、live sensor inference、new participant prediction 或 real-time workload diagnosis。

如未来确需现场输入新数据，只能在 Phase 10 之外另建并明确标记：

```text
DEMO-ONLY FULL-DATA MODEL - NOT USED FOR REPORTED EVALUATION
```

该未来模型不得混入当前 Phase 10，也不得用于论文已报告性能。

#### Phase 10 交付物

- `final_prediction_library/`
- `final_statistics_bundle/`
- `final_paper_tables/`
- `final_paper_figures/`
- `rq_evidence_conclusion_matrix/`
- `reproducibility_package/`
- `cross_phase_consistency_audit/`
- `best_hdc_demo_ui/`
- `Phase_10_Final_Synthesis_and_Demo_UI.ipynb`
- `final_project_manifest.json`
- `final_experiment_report.md`

#### Phase 10 完成标准

以下条件必须全部满足：

- Phase 00-09 freeze integrity：`PASS`；
- final prediction library、final statistics bundle、final paper tables 和 final paper figures：完整；
- RQ mapping、reproducibility package：完整；
- cross-phase numerical consistency：`PASS`；
- Best Dual-Task HDC UI：可本地启动；
- UI read-only audit：`PASS`；
- UI 数字与冻结 artifact：一致；
- UI traditional models included：`NO`；
- UI other HDC variants included：`NO`；
- UI model training executed：`NO`；
- UI new-data prediction included：`NO`；
- final Notebook persistence：`PASS`；
- final project manifest：`PASS`。

#### 科学结论边界

Phase 10 必须保留传统模型对照的论文价值、四种 HDC variant 比较的模型选择证据、Phase 08 shortcut 结论、Phase 09 missing-modality 与 LOSO 结论、所有负结果与不显著结果，以及 unseen scenario/task-template/route metadata 限制。

回归任务必须使用 **bounded difficulty-induced workload proxy regression**，不得表述为 directly measured continuous cognitive workload。UI 聚焦最优双任务 HDC 不构成结果驱动重新设计，也不得缩窄或改写 Phase 00-09 的科学证据。

## 16. 结果表与图的最低集合

### 表格

1. 数据集和模态统计表。
2. 分类基线比较表。
3. 回归基线比较表。
4. 四种 HDC 变体及两个回归头比较表。
5. 单模态与多模态比较表。
6. With/without performance 捷径分析表。
7. 缺失模态与 LOSO 结果表。
8. 训练时间、推理时间和内存表。

### 图形

1. 四类分布和受试者分布。
2. 最佳分类器和最佳 HDC 的混淆矩阵。
3. 回归真实值-预测值图和残差图。
4. 分类-回归一致性图。
5. 模态贡献排名图。
6. 缺失模态性能下降图。
7. 性能-效率 Pareto 图。
8. 受试者级误差分布图。

## 17. 十六周时间表

- 第 1 周：完成 target_score、unknown 特征决策和固定 folds。
- 第 2-3 周：完成 Phase 04A 分类基线。
- 第 4 周：完成 Phase 04B 回归基线。
- 第 5-6 周：完成 Vanilla HDC 编码、分类头和两个回归头。
- 第 7-8 周：完成 OnlineHD-style、Multi-centroid 和 Hybrid 筛选。
- 第 9-10 周：完成单模态、多模态和 performance 捷径分析。
- 第 11 周：完成缺失模态实验。
- 第 12 周：完成最终 LOSO、时间和内存评估。
- 第 13 周：冻结所有 out-of-fold 预测，完成统计和误差分析。
- 第 14 周：完成 Methods 和 Results 初稿。
- 第 15 周：完成 Introduction、Related Work、Discussion 和 Limitations。
- 第 16 周：执行 Phase 10 最终汇总、跨阶段数字核对、可复现性打包、只读演示 UI 验证、主张审计、导师修改和答辩准备；OnlineHD 顺序回放保持 `OPTIONAL_NOT_EXECUTED`。

## 18. 风险与止损规则

### 回归标签只有四个值

风险：模型可能只是学习有序类别，而不是真正连续量。

处理：明确称为 bounded proxy score regression；同时报告连续指标、四舍五入指标和分类-回归一致性，不夸大连续含义。

### 特征数远大于样本数

风险：过拟合和调参不稳定。

处理：嵌套 subject-wise 验证、紧凑参数网格、折内特征选择和强正则化。

### HDC 回归坍缩

风险：预测集中在 2.5 附近。

处理：报告每个真实等级的预测分布；比较 similarity decoder 与 Ridge readout；若均无增量价值，保留负结果。

### 实验矩阵过大

风险：无法按期完成。

处理：所有模型只在 Primary 设置完成筛选；只有 Top 模型进入模态、融合、LOSO 和鲁棒性实验。

### Performance 捷径学习

风险：高结果来自任务执行误差而不是多模态代理模式。

处理：without-performance 为唯一主结果，with-performance 和 performance-only 只能放在辅助或敏感性分析中。

### 实时系统范围失控

风险：LSL/dashboard 工程消耗毕业时间。

处理：实时系统保持未来工作；Phase 10 只允许基于冻结 artifact 的本地离线只读展示，不包含训练、部署、实时传感器推理或新参与者预测。OnlineHD 顺序回放保持可选且不执行不影响论文完整性。

## 19. 毕业最低完成标准

满足以下条件即可形成完整硕士论文：

1. 分类与回归目标、固定 folds 和防泄漏流程可复现。
2. 完成传统分类和回归基线。
3. 完成 P1 承诺的四种 HDC 分类模型。
4. 完成至少两个 HDC 回归头。
5. 完成主多模态与至少四个模态组的比较。
6. 完成 with/without performance 捷径分析。
7. 完成最终模型的受试者级统计和误差分析。
8. 保存完整 OOF 预测、配置、种子、时间和日志。
9. 论文结论严格限定为 workload-proxy classification and regression。

## 20. 最终论文论证主线

论文不应写成模型排行榜，而应回答一个连续的问题链：

1. 在严格的未见受试者验证下，多模态数据能否支持分类和回归？
2. HDC 与传统方法相比，性能和效率分别如何？
3. 同一 HDC 表征能否同时产生可靠的离散等级和连续代理分数？
4. 哪些模态真正提供信息，哪些特征可能形成捷径？
5. 当受试者和传感器条件变化时，模型在哪里失效？

只要这五个问题得到可复现、边界清晰的回答，即使 HDC 并非所有指标第一，项目仍具有完整的硕士论文价值。
