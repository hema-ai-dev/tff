# Stage 1B Dirichlet Non-IID FedAvg Technical Report v1.0

> **项目名称**：Vehicle-to-Cloud Federated Learning for Traffic Sign Recognition  
> **数据集**：GTSRB（German Traffic Sign Recognition Benchmark）  
> **框架**：TensorFlow Federated（TFF 0.87.0）  
> **任务类型**：43 类交通标志图像分类  
> **客户端数量**：5  
> **阶段定位**：在 Stage 1A IID FedAvg 基线基础上，引入可控的标签分布偏斜（Label Distribution Skew），系统研究 FedAvg 在不同 Non-IID 程度下的性能、收敛性和类别级退化现象。  
> **文档版本**：v1.0  
> **基准随机种子**：20260720  
> **最终正式训练轮数**：160 rounds

---

## 摘要

Stage 1A 已完成 GTSRB 数据集上的 5 客户端 IID 联邦学习基线，包括 Centralized、Local-only 和 Weighted FedAvg 三组实验，并验证了完整的 TFF 联邦训练、聚合、指标记录、checkpoint 保存和恢复流程。Stage 1B 在此基础上进一步引入 Dirichlet 标签分布偏斜，以构造可控且可复现的 Non-IID 客户端数据分布。

本阶段设置了四种 Dirichlet 强度：α=10、1.0、0.5 和 0.1。为保证实验可复现和公平比较，本阶段冻结 Stage 1A 的模型、优化器、学习率、batch size、客户端数量、每轮参与客户端数、聚合方式和评估方法，仅改变客户端训练数据划分。所有划分均固定保存，并输出客户端—类别分布统计、长表、宽表、manifest、元数据和热力图。

最终 160 轮结果表明：α=10 与 IID 基本一致；α=1.0 与 IID 几乎完全一致，说明 FedAvg 对中等标签偏斜具有较强鲁棒性；α=0.5 的总体准确率仅小幅下降，但 Macro Class Accuracy 出现更明显下降，说明类别公平性开始受损；α=0.1 下 FedAvg 出现严重退化，官方测试准确率降至 71.49%，多个类别准确率降为 0，且混淆矩阵显示明显的系统性类别偏置。

因此，Stage 1B 成功建立了一个可复现、可控、可扩展的 Non-IID 联邦学习实验基线，可直接作为后续 FedProx、SCAFFOLD、MOON、FedNova 或其他联邦优化算法的统一对照平台。

---

# 1. 研究背景与阶段目标

## 1.1 为什么需要 Stage 1B

Stage 1A 的主要任务是验证 GTSRB 数据加载、5 客户端组织、TFF 本地训练、Weighted FedAvg 聚合、全局模型更新、指标记录、通信量估算和 checkpoint 恢复是否能够完整运行。最终关系为：

```text
Local-only < FedAvg ≈ Centralized
```

但是 Stage 1A 使用按类别分层的近似 IID 划分，每个客户端样本数近似一致、覆盖全部 43 类，类别比例接近全局分布。该设置适合验证基础联邦流程，却不能代表真实车联网环境。

真实客户端可能位于不同道路、城市、天气、时段和交通场景，各自采集的交通标志类别及比例会明显不同。因此 Stage 1B 的核心问题是：

> 当客户端数据分布不一致时，FedAvg 是否仍然有效？

## 1.2 核心研究问题

1. 如何用统一参数连续控制 Non-IID 程度？
2. 如何保证划分可复现、可审计、可被后续算法复用？
3. FedAvg 在轻度、中度、明显和极端 Label Skew 下分别如何表现？
4. Overall Accuracy 和 Macro Class Accuracy 是否同步下降？
5. 极端异构是否会导致部分类别完全失效？
6. 100 轮是否足够，还是需要统一扩展到 160 轮？

---

# 2. Stage 1A 到 Stage 1B 的控制变量设计

## 2.1 “冻结 Stage 1A 配置”的含义

冻结并不是禁止修改代码，而是把已经验证成功的实验条件固定为后续基线。

允许修改：

- Dirichlet 数据划分脚本；
- 参数读取方式；
- CSV、metadata 和 heatmap 输出；
- 数据完整性验证；
- 客户端级评价；
- 日志和报告生成；
- 明确的软件错误。

暂时不修改：

- CNN 主干结构；
- 图像预处理；
- batch size；
- local epochs；
- 客户端与服务器优化器；
- 学习率；
- 客户端数量；
- 聚合权重；
- 官方测试集；
- 主要评价指标。

这样才能把性能变化主要归因于数据分布变化。

## 2.2 固定配置

```text
Dataset: GTSRB
Train samples: 39,209
Test samples: 12,630
Num classes: 43
Input: 32×32×3
Num clients: 5
Clients per round: 5
Local epochs: 1
Batch size: 64
Client optimizer: Adam
Client learning rate: 0.001
Server optimizer: stateless SGDM
Server learning rate: 1.0
Client weighting: NUM_EXAMPLES
Seed: 20260720
Formal rounds: 160
```

统一 CNN：

```python
def build_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(43),
    ])
```

损失与指标：

```text
SparseCategoricalCrossentropy(from_logits=True)
SparseCategoricalAccuracy
```

---

# 3. 为什么选择 Dirichlet Label Skew

固定类别子集划分虽然简单，但难以连续控制异构程度，也不利于与主流联邦学习实验比较。Dirichlet 划分对每个类别 k 采样：

\[
p_k \sim Dirichlet(\alpha,\alpha,\alpha,\alpha,\alpha)
\]

α 越大，类别分配越均匀；α 越小，类别越集中到少数客户端。

本项目设置：

```text
α=10   轻度 Non-IID，接近 IID
α=1.0  中等 Non-IID
α=0.5  明显 Non-IID
α=0.1  极端 Non-IID
```

α=10 不能直接等同 IID。严格对照仍然是 Stage 1A 的 stratified IID。

---

# 4. Dirichlet 数据划分实现

核心脚本：

```text
src/gtsrb_1b/prepare_gtsrb_dirichlet.py
```

核心流程：

1. 读取 `Train.csv`；
2. 检查 `Path`、`ClassId`、标签范围和重复路径；
3. 对每个类别打乱样本；
4. 采样 Dirichlet 比例；
5. 使用 multinomial 生成整数客户端样本数；
6. 分配全部样本；
7. 验证约束；
8. 不满足则重新采样；
9. 保存 CSV、manifest、metadata 和热力图。

为什么使用 multinomial：

```python
counts = rng.multinomial(len(class_indices), proportions)
```

它保证每个类别的整数分配总和严格等于该类别样本总数，避免四舍五入导致遗漏或重复。

正式约束：

```text
min_samples_per_client = 500
min_classes_per_client = 5
present_class_min_samples = 5
max_attempts = 5000
```

“有效类别”定义为该客户端至少拥有该类别 5 张图片。

---

# 5. 固定划分输出

```text
data/gtsrb/
├── federated_iid_5clients_seed20260720/
├── federated_dirichlet_alpha10_5clients_seed20260720/
├── federated_dirichlet_alpha1_5clients_seed20260720/
├── federated_dirichlet_alpha05_5clients_seed20260720/
├── federated_dirichlet_alpha01_5clients_seed20260720/
└── stage1b_partition_summary.csv
```

每个 Dirichlet 目录包含：

```text
clients/client_0_train.csv ... client_4_train.csv
client_distribution.csv
client_distribution_wide.csv
partition_manifest.csv
split_metadata.json
client_class_distribution_count.png
client_class_distribution_ratio.png
```

这些文件的意义：

- 客户端 CSV：后续所有算法直接读取；
- distribution long/wide：统计与绘图；
- manifest：检查重复、遗漏和归属；
- metadata：记录 α、seed、样本数、类别数和 JS divergence；
- heatmap：论文展示 Non-IID 程度。

保存划分的根本原因是保证 FedAvg、FedProx、SCAFFOLD 等算法使用完全相同的数据分布。

---

# 6. 数据划分结果

| α | Mean JS | Min Samples | Max Samples | Present Classes | Max Largest-Class Ratio |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.00846 | 7063 | 8558 | 43–43 | 8.49% |
| 1.0 | 0.08801 | 6630 | 10641 | 40–43 | 18.82% |
| 0.5 | 0.13379 | 7051 | 8711 | 36–40 | 23.03% |
| 0.1 | 0.31478 | 5748 | 10815 | 16–24 | 36.01% |

Mean JS 严格递增：

```text
0.00846 → 0.08801 → 0.13379 → 0.31478
```

说明四组数据确实形成从轻度到极端的连续 Non-IID 梯度。

---

# 7. Stage 1B FedAvg 训练实现

核心脚本：

```text
src/gtsrb_1b/train_fedavg_stage1b.py
```

关键改造：

1. 使用 `--partition-dir`；
2. 删除写死的 IID 路径；
3. 接入 `stage1b_partition_loader.py`；
4. 训练前验证客户端数、总样本数、重复和遗漏；
5. 允许单个客户端缺少类别；
6. 官方测试集固定读取 `data/gtsrb/Test.csv`；
7. 保存 α、Mean JS 和客户端分布；
8. 恢复训练时检查 partition 一致性；
9. 保存 `client_partition_summary.csv`；
10. 保留 Weighted FedAvg 和 NPZ checkpoint。

Non-IID 下不能要求每个客户端覆盖全部 43 类。客户端仅检查标签合法；global validation 和 official test 才要求完整类别覆盖。

---

# 8. 本地训练/验证与官方测试

每个客户端内部继续按：

```text
90% local train
10% local validation
```

五个本地验证集合并成 `global_val.csv`。官方 `Test.csv` 只在训练结束后评价。

因此：

- 验证准确率用于 checkpoint 选择；
- 官方 Test Accuracy 用于最终结论；
- Macro Class Accuracy 用于类别公平性分析。

由于 GTSRB 可能含同一真实标志的连续帧，训练内部验证集可能比官方测试集更容易。论文不能只报告接近 100% 的 validation accuracy。

---

# 9. checkpoint 与恢复

当前保存：

```text
TFF global ModelWeights → NumPy NPZ
run_state.json → completed round
```

恢复：

```text
load NPZ → set_model_weights → next round
```

当前服务器优化器是无动量 SGDM，聚合器是无状态 NUM_EXAMPLES weighted mean，因此该恢复方式对当前基线有效。

若以后使用 FedAdam、FedYogi、server momentum 或有状态聚合器，必须保存完整服务器状态。

---

# 10. 工程问题与修复

## 10.1 100 轮未充分收敛

最初 100 轮 best round：

```text
α=10: 97
α=1: 98
α=0.5: 99
α=0.1: 100
```

说明 100 轮不足，因此四组统一恢复到 160 轮，而不是只延长某一组。

## 10.2 TFF 完成后 Python 不退出

表现为结果全部保存，但 Shell 无法进入下一组。修复：

```python
if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
```

只在 `main()` 完整成功后强制退出，不吞掉真实错误。

## 10.3 Dirichlet 目录没有 global_test.csv

修复为始终读取：

```text
data/gtsrb/Test.csv
```

Dirichlet 目录只保存训练客户端划分。

---

# 11. 最终 160 轮结果

| Partition | α | Mean JS | Best Round | Best Val Acc | Test Acc | Macro Class Acc | Test Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| IID | — | 0.000000 | 152 | 99.5652% | 92.6049% | 89.4405% | 0.3921 |
| Dirichlet | 10 | 0.008460 | 157 | 99.4904% | 92.3911% | 89.3823% | 0.4500 |
| Dirichlet | 1.0 | 0.088009 | 160 | 99.7198% | 92.6049% | 89.4907% | 0.3962 |
| Dirichlet | 0.5 | 0.133792 | 158 | 99.1099% | 91.7736% | 87.2721% | 0.4189 |
| Dirichlet | 0.1 | 0.314775 | 156 | 80.5337% | 71.4885% | 69.4158% | 1.8679 |

---

# 12. 结果分析

## 12.1 α=10：近 IID

与 IID 相比：

```text
Test Accuracy 仅下降约 0.21 个百分点
Macro Accuracy 仅下降约 0.06 个百分点
```

结论：α=10 的异构程度很低，FedAvg 基本保持 IID 性能。

## 12.2 α=1：中等 Non-IID

```text
Test Accuracy: 92.60%
Macro Accuracy: 89.49%
```

几乎与 IID 完全一致。略高的 Macro Accuracy 不应解释为 α=1 优于 IID，更合理的是单随机种子、划分和训练波动。

核心结论：

> 在当前任务、5 客户端、全参与和 local epoch=1 条件下，FedAvg 对 α=1 的中等 Label Skew 具有较强鲁棒性。

## 12.3 α=0.5：明显 Non-IID

相对 IID：

```text
Test Accuracy 下降约 0.83 个百分点
Macro Accuracy 下降约 2.17 个百分点
```

总体准确率下降不大，但 Macro Accuracy 更敏感，说明类别级公平性先于总体性能恶化。

## 12.4 α=0.1：极端 Non-IID

```text
Test Accuracy: 71.49%
Macro Accuracy: 69.42%
Test Loss: 1.8679
```

相对 IID：

```text
Test Accuracy 下降约 21.12 个百分点
Macro Accuracy 下降约 20.02 个百分点
```

说明 FedAvg 在极端标签偏斜下明显失效。

---

# 13. 类别级分析

α=0.1 下多个类别准确率降至 0：

```text
class 0
class 8
class 19
class 38
class 39
```

部分近乎失效：

```text
class 33 ≈ 0.48%
class 17 ≈ 10.56%
class 5 ≈ 30.32%
class 6 ≈ 31.33%
```

从 α=10 到 α=0.1 的典型下降：

| Class | α=10 | α=0.1 | Drop |
|---:|---:|---:|---:|
| 19 | 100.00% | 0.00% | 100.00% |
| 33 | 99.52% | 0.48% | 99.05% |
| 38 | 98.41% | 0.00% | 98.41% |
| 8 | 92.67% | 0.00% | 92.67% |
| 0 | 70.00% | 0.00% | 70.00% |
| 17 | 80.28% | 10.56% | 69.72% |
| 39 | 68.89% | 0.00% | 68.89% |
| 5 | 95.08% | 30.32% | 64.76% |
| 6 | 91.33% | 31.33% | 60.00% |

这不是简单的整体下降，而是部分类别知识几乎完全失效。

更严谨的表述应为：

```text
class-level forgetting or class-level failure under extreme label skew
```

除非后续专门研究时间序列遗忘过程，否则不宜直接把它等同于持续学习中的经典 catastrophic forgetting。

---

# 14. 混淆矩阵分析

α=0.1 最常见错误：

```text
true 38 → pred 34 : 479
true 8  → pred 5  : 240
true 5  → pred 2  : 227
true 17 → pred 14 : 168
true 2  → pred 1  : 125
true 9  → pred 10 : 118
true 5  → pred 3  : 94
true 33 → pred 36 : 92
true 8  → pred 2  : 81
true 17 → pred 9  : 76
```

错误集中到特定类别对，说明不是均匀随机错误，而是形成稳定错误决策边界和系统性类别偏置，可称为：

```text
systematic inter-class confusion
```

---

# 15. 收敛与通信分析

最终 best round：

| Partition | Best Round |
|---|---:|
| IID | 152 |
| α=10 | 157 |
| α=1 | 160 |
| α=0.5 | 158 |
| α=0.1 | 156 |

说明 100 轮确实不足，160 轮是合理正式预算。

通信量：

```text
单个模型：395,180 bytes ≈ 0.377 MiB
每轮上下行：3,951,800 bytes ≈ 3.77 MiB
160 轮：632,288,000 bytes ≈ 603 MiB
```

不含协议头、序列化、TLS、重传等实际网络开销。

---

# 16. 核心研究结论

1. **FedAvg 对轻度和中度 Label Skew 具有较强鲁棒性。** α≥1 时官方测试性能与 IID 基本一致。
2. **Macro Accuracy 比 Overall Accuracy 更敏感。** α=0.5 下类别公平性已经下降，而整体准确率仍较稳定。
3. **极端 Label Skew 导致类别级严重失效。** α=0.1 下多个类别准确率为 0。
4. **错误具有系统性。** 混淆集中在特定类别对，而非均匀随机噪声。
5. **性能退化不是线性的。** Mean JS 从 0.008 增到 0.134 时性能变化较小，但增至 0.315 时出现断崖式下降。
6. **可能存在异构阈值。** 在一定范围内 FedAvg 可以抵消客户端差异，超过阈值后简单加权平均不足以恢复全局类别知识。

未来可细化 α：

```text
0.3, 0.2, 0.15
```

以定位性能断崖出现的位置。

---

# 17. 可直接用于论文的中文段落

## 17.1 实验设置

为研究标签分布异构对联邦交通标志识别的影响，本研究采用 Dirichlet 分布构造可控的 Label Skew。具体设置 α=10、1.0、0.5 和 0.1 四种异构程度，并保留按类别分层划分的 IID 结果作为对照。所有实验均使用相同的 5 客户端、CNN 模型、图像预处理、客户端 Adam 优化器、服务器 SGDM 优化器、本地训练轮数和通信轮数，唯一变化为客户端数据划分。所有 Dirichlet 划分均固定随机种子并保存为静态客户端清单，从而保证不同算法之间使用完全一致的数据分布。

## 17.2 实验结果

实验结果表明，FedAvg 对轻度和中度标签偏斜具有较强鲁棒性。在 α=10 和 α=1.0 条件下，模型官方测试准确率分别达到 92.39% 和 92.60%，与 IID 条件下的 92.60% 基本一致。当 α 降至 0.5 时，整体测试准确率仍保持在 91.77%，但 Macro Class Accuracy 由 IID 条件下的 89.44% 降至 87.27%，说明类别级性能不均衡已经开始出现。在极端异构的 α=0.1 条件下，模型测试准确率显著下降至 71.49%，Macro Class Accuracy 降至 69.42%，表明传统 FedAvg 难以处理高度集中的标签分布。

## 17.3 类别分析

进一步的类别级分析显示，极端标签偏斜并非仅导致整体性能均匀下降，而是造成部分类别的严重失效。在 α=0.1 条件下，类别 0、8、19、38 和 39 的测试准确率降至 0。混淆矩阵还显示出明显的系统性误分类模式，例如类别 38 大量被预测为类别 34，类别 8 大量被预测为类别 5。这说明极端 Non-IID 使全局模型偏向少数客户端中占优势的类别，并削弱了对低频或局部缺失类别的表征能力。

## 17.4 Discussion

值得注意的是，FedAvg 的性能退化并未随 Dirichlet 异构指标线性变化。在 α≥1 的条件下，即使客户端类别比例已经出现明显差异，全局模型仍保持了接近 IID 的测试性能；而当 α 降低至 0.1 时，性能出现断崖式下降。这说明 FedAvg 可能在一定异构范围内具有较强的聚合鲁棒性，但当本地类别缺失和局部优势类别过于严重时，客户端更新方向之间的冲突将超过简单加权平均的修正能力。

## 17.5 Conclusion

本阶段建立了一个可复现的 GTSRB Dirichlet Non-IID 联邦学习实验基线。结果表明，FedAvg 能够有效应对轻度和中度标签偏斜，但在极端 Label Skew 下会出现显著的整体性能下降、类别级失效和系统性类别混淆。该结果为后续研究 FedProx、SCAFFOLD、MOON 或其他异构优化算法提供了明确的困难场景和公平对照基线。

---

# 18. 可直接用于英文论文的核心句子

> Under mild and moderate label skew (α ≥ 1), FedAvg maintains test performance comparable to the IID baseline.

> When α decreases to 0.5, the overall accuracy remains relatively stable, whereas macro class accuracy starts to degrade, indicating increasing class-level unfairness.

> Under extreme label skew (α = 0.1), FedAvg suffers severe performance degradation, accompanied by complete failure on several classes.

> The confusion matrix reveals systematic inter-class bias rather than uniformly distributed random errors.

> These results suggest that FedAvg is robust within a moderate heterogeneity range, but its simple weighted averaging mechanism becomes insufficient once client label distributions become highly concentrated.

---

# 19. 后续研究建议

## 19.1 不继续修改当前 FedAvg baseline

当前 FedAvg 已完成 IID、四档 Dirichlet、160 轮、官方测试、Macro Accuracy、Per-class Accuracy、Confusion Matrix、checkpoint 和通信量估算。它应被保留为统一 baseline。

后续不应随意改变模型、学习率、local epochs、聚合权重或 partition。

## 19.2 Stage 1C 算法优先级

### 第一优先：FedProx

- 实现相对简单；
- 与 FedAvg 结构接近；
- 直接抑制客户端漂移；
- 适合 α=0.5 和 α=0.1。

### 第二优先：SCAFFOLD

- 使用控制变量修正客户端漂移；
- 对强 Non-IID 有理论意义；
- 状态管理更复杂。

### 第三优先：MOON

- 通过模型对比约束本地模型；
- 对 Non-IID 有研究价值；
- 需要额外计算和历史模型状态。

### FedBN

更适合 feature shift，如晴天/雨天、白天/夜间、模糊/清晰和摄像头风格差异。当前 Stage 1B 主要是 Label Skew，因此 FedBN 不应作为第一个对比算法。

## 19.3 推荐 Stage 1C 主矩阵

| Algorithm | IID | α=1 | α=0.5 | α=0.1 |
|---|---:|---:|---:|---:|
| FedAvg | ✓ | ✓ | ✓ | ✓ |
| FedProx | ✓ | ✓ | ✓ | ✓ |
| SCAFFOLD | ✓ | ✓ | ✓ | ✓ |
| MOON | 可选 | ✓ | ✓ | ✓ |

α=10 与 IID 过于接近，可保留在附录，但不必成为所有算法的主实验。

---

# 20. 后续建议增加的指标

## 20.1 客户端公平性

全局模型分别在各客户端本地验证集上评价：

```text
client_0_accuracy ... client_4_accuracy
mean client accuracy
worst client accuracy
client accuracy std
best-worst gap
```

## 20.2 多随机种子

正式论文建议至少 3 个训练 seed，报告 mean ± std。应区分 partition seed 与 training seed。

较严谨方案：固定 partition，只改变训练 seed，先测训练稳定性；之后再补不同 partition seed。

## 20.3 Track-level 划分

GTSRB 同一真实标志可能对应连续帧。未来可增加 track-level train/validation/client split，降低相似帧泄漏。

---

# 21. Decision Log

1. **选择 Dirichlet 而不是固定类别划分**：可调、可复现、文献常用、便于算法比较。
2. **选择 α=10、1、0.5、0.1**：覆盖近 IID、中等、明显和极端四档。
3. **增加客户端约束**：避免空客户端和不可训练划分。
4. **保存 partition 而不是训练时动态生成**：保证复现、审计和公平。
5. **100 轮统一延长到 160 轮**：best round 全部贴近 100，说明尚未收敛。
6. **不继续优化 FedAvg**：其作用是 baseline，后续算法必须与其公平比较。
7. **主要报告 Official Test Accuracy + Macro Accuracy**：validation 来自官方训练集内部，Macro Accuracy 更能反映类别公平性。

---

# 22. Troubleshooting 汇总

- 客户端缺少类别时报错：取消客户端必须覆盖 43 类的限制。
- Non-IID 目录缺少 global_test.csv：统一读取 `data/gtsrb/Test.csv`。
- `--split-root` 与 `--partition-dir` 冲突：统一使用 `--partition-dir`。
- 训练完成后 Shell 不继续：成功后使用 `os._exit(0)`。
- 100 轮结果偏低：统一恢复到 160 轮。
- α=0.1 性能差是否是代码错误：不是。数据完整、趋势连续、类别与混淆分析均支持这是合理实验现象。

---

# 23. 重要项目文件

```text
src/gtsrb_1b/
├── prepare_gtsrb_dirichlet.py
├── stage1b_partition_loader.py
├── test_stage1b_partition_loading.py
└── train_fedavg_stage1b.py
```

```text
scripts/
├── run_stage1b_fedavg_all.sh
├── resume_stage1b_fedavg_all.sh
└── summarize_stage1b_results.py
```

```text
outputs/stage1b/
├── fedavg_alpha10_seed20260720/
├── fedavg_alpha1_seed20260720/
├── fedavg_alpha05_seed20260720/
├── fedavg_alpha01_seed20260720/
└── comparison_160rounds/
```

长期保留：

```text
metrics.csv
test_metrics.json
config.json
run_state.json
client_partition_summary.csv
per_class_accuracy.csv
confusion_matrix.csv
global_models/best_global_model.h5
global_models/last_global_model.h5
checkpoints/
plots/
manifests/
```

---

# 24. AI 交接说明

## 已完成

### Stage 1A

- GTSRB 数据检查；
- 5 客户端 IID；
- Centralized；
- Local-only；
- Weighted FedAvg；
- 无 BN CNN；
- checkpoint 恢复；
- 160 轮；
- 官方测试；
- 类别准确率；
- 通信量估算。

### Stage 1B

- Dirichlet α=10、1、0.5、0.1；
- 固定 partition；
- CSV、manifest、metadata、heatmap；
- Mean JS divergence；
- Non-IID FedAvg；
- 100→160 轮统一恢复；
- 官方测试；
- Macro Accuracy；
- Per-class Accuracy；
- Confusion Matrix；
- 类别退化分析；
- 统一结果汇总。

## 不要重新做

- 不要重新随机生成现有 partition；
- 不要修改 Stage 1B FedAvg 基线超参数；
- 不要重新加入 BatchNorm 到当前 baseline；
- 不要只用 validation accuracy 得出结论；
- 不要改变官方 Test.csv；
- 不要将 α=10 写成 IID；
- 不要让不同算法使用不同 partition。

## 下一步

```text
Stage 1C-1: 实现 FedProx
Stage 1C-2: α=0.5 和 α=0.1 冒烟测试
Stage 1C-3: 统一 160 轮正式实验
Stage 1C-4: 与 FedAvg 比较 Test Accuracy、Macro Accuracy、
Worst Class、客户端公平性和通信轮数
```

---

# 25. 最终评价

Stage 1B 已经完成从“数据异构构造”到“全局性能、类别性能和系统性错误模式分析”的完整实验闭环。

其价值不仅是证明 Non-IID 会降低准确率，而是进一步揭示：

1. FedAvg 在轻度和中度 Label Skew 下具有较强鲁棒性；
2. 类别公平性比总体准确率更早恶化；
3. 性能下降具有明显非线性；
4. 极端异构会导致部分类别完全失效；
5. 错误呈系统性偏置；
6. α=0.1 是后续异构优化算法最有价值的困难场景。

因此 Stage 1B 已经具备工程复现价值、实验基线价值、论文结果价值、算法扩展价值和项目交接价值。

---

# 26. 一句话总结

> Stage 1B 在固定 Stage 1A FedAvg 基线的前提下，通过四组可复现的 Dirichlet Label Skew 数据划分，系统证明了 FedAvg 在轻中度 Non-IID 下具有较强鲁棒性，但在极端 α=0.1 异构下会出现显著整体性能下降、类别级失效和系统性类别混淆，从而为后续 FedProx、SCAFFOLD 等异构优化算法建立了严格、统一且具有论文价值的对照平台。
