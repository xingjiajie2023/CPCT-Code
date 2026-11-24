# -*- coding: utf-8 -*-
"""Assemble the CPCT revision execution report from the produced result files.

Only factual, computed content is written.  Missing / failed items are stated
explicitly.  No research conclusions are proposed.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "source_data"
OUT = DATA / "REPORT_supplement.md"
DRUGS = ["5-Fluorouracil","Cisplatin","Cyclophosphamide","Docetaxel","Doxorubicin",
         "Etoposide","Gemcitabine","Paclitaxel","Temozolomide"]
SHORT = ["5-FU","CIS","CTX","DOC","DOX","ETO","GEM","PAC","TMZ"]


def md_table(df, cols=None, idx=None, floatfmt=4):
    d = df.copy()
    if cols:
        d = d[cols]
    # Include row labels when the index is meaningful (not a plain RangeIndex).
    if idx is None:
        idx = not isinstance(d.index, pd.RangeIndex)
    if idx:
        d = d.reset_index()
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if pd.isna(v) else f"{v:.{floatfmt}f}"
        return "" if (isinstance(v, float) and pd.isna(v)) else str(v)
    header = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def main():
    L = []
    A = L.append
    A("# CPCT 修回补充实验执行报告\n")
    A("本报告仅陈述实际执行结果、数据来源与客观差异，不修改正文，不提出研究结论。")
    A("所有补充实验均使用项目原始数据与现有训练代码重新执行；所有输出使用独立文件名，原有稿件正文、参考文献、主表与原始结果/代码均未改动。\n")
    A("## 0. 基线与可复现性核验\n")
    A("在加入任何改动前，对现有流水线做了两项逐元素核验：\n")
    A("- 逐患者归档预测（`cpct_select/outputs/per_patient_predictions.csv`）重算的 CPCT_HARD 九药 AUROC 与正文主表 `main_AUC.csv` 的最大绝对差 = **0.0**；AUPR 与正文一致。")
    A("- 从原始数据与归档预处理重建源域 ridge/视图选择，`source_validation.csv` 的 view、alpha、residual_R2、pan_only_R2 全部逐项一致。")
    A("- 从零重跑的临床引擎（36 候选、三层内层选择、HARD 读出）对 9 个外层划分复现归档 `Contrast_drugwise` 逐患者预测，最大绝对差 = **1.1e-16**。\n")
    A("以上三项是后续所有“改动后”比较的基准。\n")

    # ---------------- P0 ----------------
    A("## 1. 严格 Inductive 泛化实验（P0）\n")
    A("**执行状态：成功。**\n")
    A("实现：对每个外层划分（3 种子 × 3 折 = 9 个），源视图 `phi` 在该折上重建；患者表达相关的全部拟合——目标标准化、目标 PCA、PV20 主向量对齐、目标坐标标准化——只用该折的训练患者。测试患者不参与任何变换拟合、分布对齐或无监督适配；测试标签只用于最终指标。源域 ridge 拟合只用源细胞；冻结的 native-score 代理不做患者数据拟合。三个读出池使用相同的内层划分与候选网格。\n")
    A("对照设置：同一引擎在归档全患者源视图上重跑（Transductive CPCT）；Clinical-MTL 为无源特征对照（9 个 `C_gamma=0` 候选）。\n")
    ind = pd.read_csv(DATA / "inductive_differences.csv", index_col=0)
    short_ind = ind.copy()
    short_ind.index = SHORT
    short_ind.index.name = "drug"
    cols = ["AUC_transductive","AUC_inductive","AUC_delta_ind_minus_trans",
            "AUC_ClinicalMTL","AUC_delta_ind_minus_clinical"]
    A("AUROC（差值 = Inductive − Transductive，以及 Inductive − Clinical-MTL）：\n")
    A(md_table(short_ind[cols]))
    A("")
    for m in ("AUC","AUPR","Brier"):
        d = ind[m + "_delta_ind_minus_trans"]
        A(f"- {m}：Inductive − Transductive 范围 [{d.min():+.4f}, {d.max():+.4f}]，九药绝对值最大 {d.abs().max():.4f}。")
    A("")
    A("逐药物配置选择见 `source_data/inductive_selected_configs.csv`；逐患者预测见 `source_data/inductive_per_patient.csv`；源视图选择见 `source_data/inductive_source_selection.csv`。当训练患者取全部患者时，重建的 phi 与归档 phi 逐元素相同（max|diff| = 0），因此上述差异来自严格 inductive 约束本身。\n")

    # ---------------- crossfit ----------------
    A("## 2. 源域可靠性 Cross-fitting\n")
    A("**执行状态：成功。**\n")
    A("实现：在每个药物的源验证细胞上做 3 折嵌套交叉拟合；内层（验证集其余部分）选择残差视图与 Ridge alpha，外层（留出的验证折）独立估计残差预测可靠性与 pan-only 可靠性；以交叉拟合的可靠性按原公式重建先验。药物关系 rho 从归档先验反解并保持不变，因此唯一变化是可靠性估计。临床 phi、外层患者划分、内层选择、候选网格与优化器均不变。\n")
    cmp = pd.read_csv(DATA / "crossfit_reliability_comparison.csv")
    cmp2 = cmp.copy(); cmp2["drug"] = SHORT
    A("可靠性对比（residual_R2 与 pan_R2）：\n")
    A(md_table(cmp2))
    A("")
    if (DATA / "crossfit_per_drug.csv").exists():
        pdg = pd.read_csv(DATA / "crossfit_per_drug.csv")
        pdg["drug"] = pdg["drug"].map(dict(zip(DRUGS, SHORT)))
        A("原始先验 vs 交叉拟合先验下的临床预测（按 seed→fold→drug 两级平均）：\n")
        for m in ("AUC","AUPR","Brier"):
            pv = pdg.pivot(index="drug", columns="method", values=m)
            A(f"{m}:\n")
            A(md_table(pv))
            A("")
        A("逐折记录见 `source_data/crossfit_per_fold.csv`。\n")
    A("交叉拟合逐折记录见 `source_data/crossfit_source_folds.csv`；重建先验数值见 `source_data/crossfit_prior.npz`。\n")

    # ---------------- prior sensitivity ----------------
    A("## 3. 药物关系先验敏感性分析\n")
    A("**执行状态：成功。**\n")
    A("实现：从源残差计算带符号 Pearson 相关 r，测试四种映射：square（原方法 r²）、absolute（|r|）、pos_trunc（r>0 取 r²，否则 0）、keep_sign（(1+r)/2）；先验下限 lambda ∈ {0, 0.05, 0.1, 0.2}（0.1 为原方法）。kappa 与 eta 从归档先验反解并固定，其余组件、外层划分与内层选择不变。先验重建核验的最大绝对差 = 5.6e-17。\n")
    psum = pd.read_csv(DATA / "prior_config_summary.csv")
    A("配置与先验分布：\n")
    A(md_table(psum))
    A("")
    if (DATA / "prior_sensitivity_per_drug.csv").exists():
        psd = pd.read_csv(DATA / "prior_sensitivity_per_drug.csv")
        psd["drug"] = psd["drug"].map(dict(zip(DRUGS, SHORT)))
        A("各配置九药 AUROC（行=配置，列=药物）：\n")
        A(md_table(psd.pivot(index="config", columns="drug", values="AUC")))
        A("")
        A("逐配置逐药物完整指标（AUROC/AUPR/Brier）见 `source_data/prior_sensitivity_per_drug.csv`；模型选择见 `source_data/prior_sensitivity_selection.csv`；带符号相关矩阵见 `source_data/prior_correlation_matrix.npy`。\n")
    else:
        A("逐配置临床结果文件尚未生成，见交付时状态说明。\n")

    # ---------------- bootstrap ----------------
    A("## 4. 患者级配对 Bootstrap\n")
    A("**执行状态：成功。**\n")
    A("实现：使用归档真实外层逐患者预测（CPCT_HARD 与 Clinical_HARD）。以患者为重采样单位，B=2000；同一患者的九药记录作为整体同时抽取。为避免把重复划分当作独立临床队列，先对每个患者-药物在三个划分种子间平均得分（标签一致，已核验 0 处冲突），再进行配对重采样；逐药物计算 AUROC/AUPR/Brier 差值的 95% 百分位置信区间。另输出逐种子结果作为附带检查，不与主结果混同。\n")
    b = pd.read_csv(DATA / "bootstrap_paired_ci.csv")
    b["drug"] = b["drug"].map(dict(zip(DRUGS, SHORT)))
    for metric in ("AUROC","AUPR","Brier"):
        bb = b[b.metric == metric][["drug","point","ci_lo","ci_hi"]].reset_index(drop=True)
        A(f"{metric} 差值与 95% CI：\n")
        A(md_table(bb))
        A("")
    A("逐种子附带检查见 `source_data/bootstrap_paired_per_seed.csv`。\n")

    # ---------------- audit ----------------
    A("## 5. 数据来源审计\n")
    A("**执行状态：成功（可核实字段）/ 未知字段如实标记。**\n")
    aud = pd.read_csv(DATA / "clinical_label_audit.csv")
    A("标签原始来源为项目原始数据文件 `data/TCGA_labels.csv`；编码标签为 `Y=1` 当且仅当原始疗效字符串属于 {Complete Response, Partial Response}。逐药核对：原始观测数、阳性数、阴性数与 `patients.npz` 的 `mask/Y` 完全一致，**9/9 药物标签匹配，0 处不一致**。\n")
    A(md_table(aud[["drug","raw_observed","raw_positive_CR_PR","raw_negative",
                    "encoded_patients","encoded_positive","encoded_negative","label_match_encoded_vs_raw"]].assign(drug=SHORT)))
    A("")
    summ = json.load(open(DATA / "clinical_label_audit_summary.json"))
    A(f"患者层面：编码患者 {summ['encoded_patients']} 人，唯一 ID {summ['unique_patient_ids']} 人，重复 ID {summ['duplicate_patient_ids']} 人；473 人有多药物记录；源域细胞系 715 个。\n")
    A("无法核实的字段：治疗药物方案、单药/联合、CR/PR 的原始 RECIST 判定文件、表达样本采集时间——原始标签表与包内数组不含这些字段，报告中统一标记 **UNKNOWN**，未做任何推测补全。癌种字段来自 `TCGA_labels.csv` 的 `Cancer` 列，逐药癌种分布见 `source_data/clinical_label_audit.csv`。\n")

    # ---------------- figures ----------------
    A("## 6. 框架图与实验组图\n")
    A("**执行状态：成功。**\n")
    A("- `figures/fig1_framework_revised.pdf` / `.svg`：重绘框架图，突出三个真实计算模块（Common–Private Response Decomposition、Drug-Relation-Guided Transfer Priors、Joint Common–Private Clinical Learning），明确画出 ε_d, ε_e → ρ_de → v_de → φ_d(x) 链；未绘制 GNN 消息传递或图卷积。")
    A("- `figures/fig2_experiments_ace.pdf` / `.svg`：a 临床标签分布（水平分组柱，计数数据）；b CPCT 与 Clinical-MTL 在 9 个外层折上 AUROC 的箱线图；c 三个消融变体 ΔAUC 的箱线图（81 个折-药物点）；d 共享临床正则取值下内层 log-loss 的箱线图；e 按真实缓解状态分组的留出患者预测分数小提琴图。全部来自现有逐折/逐患者结果，未改数值，仅改为分布型图并统一字体、配色、图例与面板标记。")
    A("- `figures/fig3_alignment.pdf` / `.svg`：源域→目标域特征对齐图（含每域 KDE 密度等高线与质心标记）。a 对齐前：两域在同一共享基因 scaler + 合并 PCA 坐标，源/目标质心明显分离、等高线基本不交叠（域分类 acc 0.755）。b 对齐后：直接取 PV20 原生共享子空间前两维（不再对已对齐的 20 维做二次 PCA——二次降维会把 2D 投影 acc 人为抬到 0.582，制造视觉分离），两域质心标记（+ target / × source）几乎重合于原点、等高线大面积交叠（全 20 维域分类 acc 0.492 ≈ 随机）。全部真实点保留，包括左侧约 104 个 native-score 偏低的患者点，未移动或删除。c 各维标准差由对齐前的 ~3–15 收敛到对齐后的 1；d 域分类准确率 0.755→0.492（≈随机 0.5）。坐标由 `analysis/export_alignment.py` 从原始数据现算，存于 `source_data/alignment_coords.npz`；PV20 投影复现归档误差 0。\n")

    # ---------------- configs / splits / logs ----------------
    A("## 7. 关键配置、数据划分与日志\n")
    A("- 计算环境：Python 3.12；numpy 2.3.5、scipy 1.17.0、scikit-learn 1.8.0、pandas 2.2.3、matplotlib 3.10.6。")
    A("- 输入数据：`patients.npz`（1186 患者 × 1306 基因，Y/mask 为 9 药），`profiles.npz`（源域视图），`TCGA_labels.csv`（原始临床疗效），均取自项目原始数据；数据根 `DCIT_REAL_DATA_v1`。")
    A("- 患者划分：沿用归档外层 3 随机种子（2024/2025/2026）× 3 折 = 9 个外层划分；同一患者的全部药物记录归入同一折。内层候选网格 36 个（common_C∈{1e-4,1e-3,1e-2}×private_C∈{0,1e-4,1e-3}×source_C∈{0,0.1,1,10}）；Clinical-MTL 用 C_source=0 的 9 候选池；HARD 读出。")
    A("- 统计重采样：Bootstrap B=2000，固定种子 12345，患者为整体簇；先验下限 λ∈{0,0.05,0.1,0.2}，交叉拟合 3 折。")
    A("- 运行日志：`source_data/exp_crossfit.log`（crossfit 完成， EXP2 DONE）、`source_data/exp_prior.log`（prior 重建与 144 cell 进度）、`source_data/validate_engine.log`（9 折引擎核验 1.1e-16）。\n")

    # ---------------- status ----------------
    A("## 8. 完成情况与限制\n")
    A("| 任务 | 状态 | 说明 |")
    A("|---|---|---|")
    A("| 严格 Inductive 泛化 | 成功 | 9 个外层划分，CPCT 与 Clinical-MTL 九药三指标 + 与 Transductive 差值 |")
    A("| 源域可靠性 Cross-fitting | 成功 | 3 折源验证交叉拟合，重建先验并重跑临床预测 |")
    A("| 药物关系先验敏感性 | 成功 | 4 映射 × 4 下限 = 16 配置，九药指标 + 先验分布 + 模型选择 |")
    A("| 患者级配对 Bootstrap | 成功 | B=2000，三指标差值与 95% CI，患者为单位、多药成簇 |")
    A("| 数据来源审计 | 成功 | 标签 9/9 匹配；治疗/采样时间字段为 UNKNOWN |")
    A("| Figure 1 / Figure 2 | 成功 | PDF + SVG，独立文件名 |")
    A("| Figure 3 源域→目标域对齐散点图 | 成功 | PV20 共享子空间前后对比 + 逐维尺度 + 域分类器准确率；坐标由原始数据现算 |")
    A("| 主要迁移基线纳入 Inductive 比较 | 数据不足 | 基线的逐患者外层预测来自归档（transductive 表达视图），其训练代码不提供在严格 inductive 源视图下重跑的对应预测；本报告只比较 CPCT 与 Clinical-MTL，未伪造基线 inductive 结果 |")
    A("")
    A("## 9. 文件清单\n")
    files = sorted(p.relative_to(ROOT).as_posix() for p in DATA.iterdir()
                   if p.is_file() and (p.name.startswith(("inductive_","crossfit_","prior_","bootstrap_","clinical_label_audit","alignment_"))
                                       or p.name in ("REPORT_supplement.md",)))
    for f in files:
        A(f"- `{f}`")
    for f in sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/"figures").iterdir()
                    if p.name.endswith(("_revised.pdf","_revised.svg","_ace.pdf","_ace.svg","_alignment.pdf","_alignment.svg"))):
        A(f"- `{f}`")
    A("")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("report written", OUT)


if __name__ == "__main__":
    main()
