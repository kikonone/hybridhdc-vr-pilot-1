# HDC VR Pilot

面向 VR 飞行任务多模态生理与行为数据的可复现实验仓库。项目使用受试者隔离的评估协议，对传统机器学习与 Hyperdimensional Computing（HDC）进行比较，并同时研究：

- 四级任务难度代理分类；
- 取值限制在 1–4 的任务难度代理回归；
- 单模态贡献、多模态融合、捷径敏感性、缺失模态鲁棒性与跨受试者泛化；
- 最终结果综合、统计汇总和本地演示 UI。

难度等级是 workload proxy，不应被解释为直接测量的心理工作负荷。

## 数据与共享限制

原始数据来自 PhysioNet VR Piloting 数据集。完整数据不包含在本仓库中：

- `vrdataset/dataPackage/` 为本地只读原始数据目录，已被 Git 忽略；
- 超大的 `vrdataset/referenceDocuments/DataQualityReport.pdf` 已被忽略；
- 超大的特征长表 `feature_extraction_long_table.csv` 已被忽略；
- 论文、Word、PDF 渲染和 PPT 制作材料不属于本代码仓库。

复现实验前，请按数据集许可自行获取原始数据，并放到 `vrdataset/dataPackage/`。

## 仓库结构

```text
.
├── README.md
├── EXPERIMENT_STATUS.md
├── requirements.txt
├── experiments/
│   ├── phase_00_project_setup/
│   ├── phase_01_raw_data_modality_audit/
│   ├── phase_02_full_multimodal_feature_extraction/
│   ├── phase_03_multimodal_dataset_labeling/
│   ├── phase_04a_traditional_classification_baselines/
│   ├── phase_04b_traditional_regression_baselines/
│   ├── phase_05_basic_dual_output_hdc/
│   ├── phase_06_hdc_variant_screening/
│   ├── phase_07_unimodal_contribution/
│   ├── phase_08_fusion_and_shortcut_analysis/
│   ├── phase_09_robustness_and_generalization/
│   └── phase_10_final_synthesis_and_demo_ui/
└── vrdataset/
    ├── referenceDocuments/
    └── starterCode/
```

每个实验阶段保留自己的 README、脚本、Notebook、配置、审计、结果和图表。阶段目录中的冻结记录与审计文件是对应阶段状态的权威来源。

## 核心数据协议

Phase 03 生成受试者级别冻结划分和三个建模数据集：

- 主数据集：`experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv`
- 含表现指标的辅助数据集：`experiments/phase_03_multimodal_dataset_labeling/data/auxiliary_with_performance.csv`
- 仅表现指标数据集：`experiments/phase_03_multimodal_dataset_labeling/data/performance_only.csv`
- 冻结外层折：`experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv`

主数据集包含 419 个运行样本、35 名受试者和 1,176 个预测特征。所有缺失值处理、标准化、特征选择和模型训练都应在训练折内部完成。

## 环境安装

建议使用 Python 3.10 或更新版本：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

部分阶段包含自己的附加依赖；运行前请同时查看对应阶段的 README 或 `requirements.txt`。

## 使用方式

1. 准备受许可的原始数据到 `vrdataset/dataPackage/`。
2. 阅读 `EXPERIMENT_STATUS.md` 和目标阶段的 README。
3. 优先使用阶段内的冻结配置、执行脚本和验证脚本。
4. 不要根据外层测试结果重新选择模型、维度或随机种子。
5. 生成物应保存在对应阶段目录，不要写回 `vrdataset/`。

最终综合入口位于 `experiments/phase_10_final_synthesis_and_demo_ui/`，本地 UI 启动说明位于其 `ui/README.md`。

## 关键原则

- 原始数据只读且不上传 GitHub；
- 主分析排除 performance metrics，辅助分析单独评估捷径风险；
- 使用受试者隔离的交叉验证或 LOSO；
- 模型选择仅基于训练侧证据；
- 保存折分、配置、预测、统计结果、审计和哈希，以支持复现；
- 分类与回归目标均为任务难度代理，结论不得越界为直接心理测量。

## 许可

代码和实验产物用于研究复现与方法开发。原始数据仍受其来源许可约束；使用者必须自行确认访问、使用和再分发条件。
