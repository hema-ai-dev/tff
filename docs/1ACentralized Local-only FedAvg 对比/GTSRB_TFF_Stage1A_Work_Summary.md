# GTSRB 联邦学习阶段 1A 工作说明

> 项目目标：使用 TensorFlow Federated（TFF）在 GTSRB 交通标志分类数据集上完成一个可复现的 5 客户端联邦学习基线流程，并对比 Centralized、Local-only 和 FedAvg 三种训练方式。  
> 当前状态：阶段 1A 的主要流程与三组基线已经完成；最终有效模型采用**不含 Batch Normalization 的统一 CNN**。

---

## 1. 阶段 1A 的目标

阶段 1A 不追求复杂算法创新，主要用于验证完整联邦学习链路是否能够正常工作：

1. GTSRB 数据加载和完整性检查；
2. 5 个客户端的数据组织；
3. 相对均衡的 IID 数据划分；
4. TensorFlow 数据流水线；
5. TFF 客户端数据格式转换；
6. 客户端本地训练；
7. 服务器按客户端样本量加权聚合；
8. 全局模型更新；
9. 每轮指标记录；
10. checkpoint 保存与恢复；
11. Centralized、Local-only、FedAvg 三组基线对比。

本阶段是一个 **43 类图像分类任务**，不是目标检测任务。

---

## 2. 软件环境

当前实验环境：

```text
Python: 3.10
TensorFlow: 2.14.1
TensorFlow Federated: 0.87.0
```

项目虚拟环境名称：

```text
tff
```

项目根目录示例：

```text
/home/hema/projects/tff_mnist_fedavg
```

---

## 3. 数据集

使用数据集：

```text
GTSRB（German Traffic Sign Recognition Benchmark）
```

数据规模：

```text
官方训练样本：39,209
官方测试样本：12,630
类别数量：43
标签范围：0～42
```

原始数据根目录：

```text
data/gtsrb
```

主要文件：

```text
data/gtsrb/
├── Train.csv
├── Test.csv
├── Meta.csv
├── Train/
├── Test/
└── ...
```

数据读取以 CSV 中的 `Path` 字段为准。Linux 区分目录名大小写，因此不能随意合并 `Train/train` 或 `Test/test`。

---

## 4. 数据检查与清洗

数据划分前完成了非破坏性的完整性检查：

- `Train.csv` 和 `Test.csv` 能正常读取；
- 存在 `Path` 和 `ClassId` 字段；
- 类别完整覆盖 `0～42`；
- CSV 中没有重复图片路径；
- 图片路径对应文件真实存在；
- 训练集和测试集没有路径交叉；
- 官方测试集没有参与客户端训练数据划分；
- 所有客户端的并集等于完整训练集；
- 客户端之间没有重复样本。

图片在训练时动态执行以下预处理：

```text
读取图片
→ 解码为 RGB 三通道
→ resize 到 32×32
→ 转为 float32
→ 像素归一化到 [0, 1]
→ shuffle / batch / prefetch
```

当前没有使用水平翻转，因为部分交通标志在左右翻转后会改变类别语义。

---

## 5. 5 客户端 IID 数据划分

使用按类别分层的 IID 划分。

划分方法：

1. 对 `ClassId=0` 的全部图片随机打乱；
2. 近似平均分配给 5 个客户端；
3. 对 `ClassId=1～42` 重复相同操作；
4. 合并每个客户端获得的所有类别样本；
5. 固定随机种子保证划分可复现。

关键参数：

```text
num_clients = 5
num_classes = 43
seed = 20260720
split_type = stratified IID
```

客户端样本数：

```text
client_0: 7,842
client_1: 7,842
client_2: 7,841
client_3: 7,842
client_4: 7,842
```

客户端样本量最大差值仅为 1，每个客户端均包含全部 43 个类别。

划分输出目录：

```text
data/gtsrb/federated_iid_5clients_seed20260720/
├── centralized_train.csv
├── global_test.csv
├── split_summary_long.csv
├── split_summary_wide.csv
├── split_metadata.json
├── client_class_distribution.png
└── clients/
    ├── client_0_train.csv
    ├── client_1_train.csv
    ├── client_2_train.csv
    ├── client_3_train.csv
    └── client_4_train.csv
```

---

## 6. 最终统一分类模型

最初模型包含 Batch Normalization，但在当前 TFF FedAvg 流程中，客户端训练准确率很高，而全局验证和测试准确率接近随机水平。

诊断表明，当前 TFF FedAvg 实现中，全局推理时的 BatchNorm moving statistics 没有按照预期随客户端训练和服务器聚合更新。因此阶段 1A 最终采用**不含 Batch Normalization 的 CNN**，保证三组基线结构一致。

最终模型结构：

```python
def build_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),

        tf.keras.layers.Conv2D(
            32, kernel_size=3,
            padding="same",
            activation="relu",
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            64, kernel_size=3,
            padding="same",
            activation="relu",
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            128, kernel_size=3,
            padding="same",
            activation="relu",
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.25),

        # 43 类 logits
        tf.keras.layers.Dense(43),
    ])
```

分类配置：

```text
输入：32×32×3
输出：43 个 logits
损失：SparseCategoricalCrossentropy(from_logits=True)
指标：SparseCategoricalAccuracy
```

---

## 7. 项目主要脚本

```text
src/gtsrb_1a/
├── prepare_gtsrb_stage1a.py
├── gtsrb_data_pipeline.py
├── train_centralized.py
├── train_local_only.py
└── train_fedavg.py
```

脚本用途：

### `prepare_gtsrb_stage1a.py`

负责：

- 数据完整性检查；
- 43 类标签检查；
- 5 客户端 IID 划分；
- 客户端互斥性检查；
- 客户端并集完整性检查；
- 类别分布统计；
- 划分元数据保存。

### `gtsrb_data_pipeline.py`

负责：

- TensorFlow `tf.data.Dataset` 构建；
- TFF 所需的 `OrderedDict(x=image, y=label)` 格式；
- 图片预处理；
- CNN 模型定义。

### `train_centralized.py`

负责：

- 完整训练集集中训练；
- 分层训练/验证划分；
- 最佳模型和最新模型保存；
- 恢复训练；
- 测试集评价；
- 混淆矩阵和各类别准确率输出。

### `train_local_only.py`

负责：

- 5 个客户端分别独立训练；
- 客户端之间不交换参数；
- 五个模型使用相同随机初始化；
- 全部在同一官方测试集上评价；
- 输出均值、标准差、最优和最差客户端。

### `train_fedavg.py`

负责：

- TFF Weighted FedAvg；
- 5 个客户端每轮全部参与；
- 客户端本地训练；
- 按客户端实际样本数加权聚合；
- 全局模型更新；
- 每轮验证；
- 每轮通信量估算；
- 全局模型和轮次 checkpoint 保存；
- 从指定通信轮次恢复。

---

## 8. Centralized 基线

训练设置：

```text
epochs = 30
batch_size = 64
optimizer = Adam
learning_rate = 0.001
validation_ratio = 0.1
seed = 20260720
```

最终结果：

```text
Test Accuracy: 91.1639%
Macro Class Accuracy: 88.4368%
Test Loss: 0.4255
Test Samples: 12,630
```

模型目录：

```text
outputs/stage1a/centralized_nobn_seed20260720/
```

最佳模型：

```text
outputs/stage1a/centralized_nobn_seed20260720/checkpoints/best.h5
```

---

## 9. Local-only 基线

训练设置：

```text
每个客户端独立训练
epochs = 30
batch_size = 64
optimizer = Adam
learning_rate = 0.001
validation_ratio = 0.1
seed = 20260720
aggregation = none
```

每个客户端约有：

```text
本地训练样本：约 7,060
本地验证样本：约 782
官方测试样本：12,630
```

各客户端测试准确率：

| Client | Test Accuracy | Macro Class Accuracy |
|---:|---:|---:|
| 0 | 58.7648% | 47.7792% |
| 1 | 59.5249% | 49.0101% |
| 2 | 61.2114% | 50.8847% |
| 3 | 59.6200% | 49.1327% |
| 4 | 61.4093% | 51.5621% |

汇总结果：

```text
Mean Test Accuracy: 60.1061%
Population Std: 1.0289%
Min Test Accuracy: 58.7648%
Max Test Accuracy: 61.4093%

Mean Macro Class Accuracy: 49.6738%
Macro Population Std: 1.3678%
```

结果目录：

```text
outputs/stage1a/local_only_nobn_seed20260720/
```

---

## 10. FedAvg 基线

### 10.1 训练设置

```text
algorithm = Weighted FedAvg
num_clients = 5
clients_per_round = 5
client_fraction = 100%
local_epochs = 1
batch_size = 64

client_optimizer = Adam
client_learning_rate = 0.001

server_optimizer = stateless SGDM
server_learning_rate = 1.0

client_weighting = NUM_EXAMPLES
seed = 20260720
```

每轮服务器执行：

```text
下发全局模型
→ 5 个客户端各自训练 1 local epoch
→ 上传模型更新
→ 按客户端样本量加权聚合
→ 更新全局模型
```

### 10.2 checkpoint

TFF `FileProgramStateManager` 在当前 TensorFlow/TFF 版本组合中出现 `_DictWrapper` 序列化兼容问题，因此当前实现没有使用 SavedModel 保存整个服务器状态。

当前 checkpoint 方式：

```text
提取全局 ModelWeights
→ 保存为 NumPy NPZ
→ run_state.json 记录完成轮次
→ 恢复时通过 set_model_weights 写回服务器状态
```

由于当前服务器优化器是无动量 SGDM、服务器学习率为 1.0，聚合器是无状态加权平均，因此该方式能够精确恢复当前基线配置。

如果以后改为：

```text
服务器 momentum
FedAdam
FedYogi
有状态聚合器
```

则必须保存完整服务器优化器和聚合器状态，不能只恢复模型权重。

### 10.3 最终训练结果

总通信轮数：

```text
160 rounds
```

最佳验证轮次：

```text
best_round = 152
```

最终结果：

```text
Best Validation Accuracy: 99.5652%
Test Accuracy: 92.6049%
Macro Class Accuracy: 89.4405%
Test Loss: 0.3921
Test Samples: 12,630
```

最近 10 轮验证准确率没有继续增长：

```text
Round 151 val_accuracy: 99.4373%
Round 152 val_accuracy: 99.5652%
Round 153～160: 未超过 Round 152
```

因此模型在约 150 轮后进入平台期，继续增加通信轮数的收益很小。

结果目录：

```text
outputs/stage1a/fedavg_nobn_seed20260720/
```

最佳全局模型：

```text
outputs/stage1a/fedavg_nobn_seed20260720/global_models/best_global_model.h5
```

最佳轮次权重：

```text
outputs/stage1a/fedavg_nobn_seed20260720/checkpoints/model_weights_round_000152.npz
```

---

## 11. 三组最终结果对比

| 方法 | 训练数据方式 | Test Accuracy | Macro Class Accuracy |
|---|---|---:|---:|
| Centralized | 所有训练数据集中训练 | 91.1639% | 88.4368% |
| Local-only Mean | 每个模型仅使用约 20% 数据 | 60.1061% | 49.6738% |
| FedAvg | 5 客户端模型更新加权聚合 | **92.6049%** | **89.4405%** |

主要结论：

1. Local-only 明显弱于 Centralized，说明单客户端数据量不足会显著降低分类泛化能力；
2. FedAvg 明显优于 Local-only，证明参数聚合有效利用了五个客户端的信息；
3. FedAvg 最终达到与 Centralized 接近、略高的测试准确率；
4. FedAvg 略高于 Centralized 不能直接解释为联邦学习理论上优于集中训练，因为两者训练预算不同；
5. 当前关系符合阶段 1A 的预期：

```text
Local-only < FedAvg ≈ Centralized
```

---

## 12. 训练预算公平性说明

三组实验的训练预算并不完全相同：

```text
Centralized: 30 epochs
Local-only: 每个客户端 30 epochs
FedAvg: 160 rounds × 每客户端每轮 1 local epoch
```

FedAvg 实际遍历训练数据的次数高于 Centralized，因此主报告中应区分：

### 固定轮数比较

```text
Centralized: 30 epochs
Local-only: 30 epochs
FedAvg: 30 rounds
```

用于比较早期收敛速度。

### 充分收敛比较

```text
Centralized: 最佳验证 epoch
Local-only: 各客户端最佳验证 epoch
FedAvg: 最佳验证 round 152
```

用于比较各方法充分训练后的最好性能。

不应写成：

```text
FedAvg 在完全相同训练预算下超过 Centralized
```

更准确的表述是：

```text
在当前各自训练配置下，充分训练后的 FedAvg 测试准确率略高于 Centralized。
```

---

## 13. 通信量

无 BatchNorm CNN 的参数载荷估算：

```text
单个模型大小：395,180 bytes
约 0.377 MiB
```

每轮 5 客户端全部参与，一次下发和一次上传：

```text
estimated_round_communication_bytes = 3,951,800 bytes
约 3.77 MiB / round
```

160 轮累计：

```text
estimated_total_communication_bytes = 632,288,000 bytes
约 603 MiB
```

该通信量是参数载荷估算，不包含：

- 协议头；
- 网络重传；
- 序列化额外开销；
- TLS 或其他网络层开销。

Round 152 以后性能没有明显提升，因此后续轮次的通信收益很低。

---

## 14. 已解决的关键问题

### 问题 1：TensorFlow/TFF 环境未进入

现象：

```text
ModuleNotFoundError: No module named 'tensorflow'
```

原因：终端仍在 base 环境，没有进入 `tff` 虚拟环境。

解决：

```bash
conda activate tff
```

### 问题 2：Keras 原生 `.keras` checkpoint 保存报错

现象：

```text
ValueError: ... native Keras format: ['options']
```

解决：在 TensorFlow 2.14.1 环境中将完整模型 checkpoint 改为 HDF5：

```text
best.h5
last.h5
```

### 问题 3：TFF 构建模型时 Dropout 与 deterministic ops 冲突

现象：

```text
RuntimeError: Random ops require a seed to be set when determinism is enabled
```

解决：

- FedAvg 脚本不强制启用全局 `enable_op_determinism()`；
- 在 TFF `model_fn()` 内显式设置 `tf.random.set_seed(seed)`。

### 问题 4：TFF program state 保存 `_DictWrapper` 报错

现象：

```text
TypeError: this __dict__ descriptor does not support '_DictWrapper' objects
```

解决：

- 不再使用 `FileProgramStateManager`；
- 当前基线将全局 `ModelWeights` 保存为 NPZ；
- 使用 `run_state.json` 记录轮次；
- 恢复时通过 `set_model_weights()` 写回。

### 问题 5：重复调用 `ModelWeights.from_tff_result`

现象：

```text
TypeError: Expected tff.structure.Struct, found ModelWeights
```

原因：TFF 0.87.0 的 `get_model_weights(state)` 已直接返回 `ModelWeights`。

解决：

```text
如果返回值已经是 ModelWeights，直接使用；
仅当返回值确实为 TFF Struct 时才调用 from_tff_result。
```

### 问题 6：带 BatchNorm 的 FedAvg 全局模型几乎失效

现象：

```text
客户端训练准确率接近 99%
全局验证准确率约 6%
官方测试准确率约 6.7%
```

原因诊断：

```text
客户端本地训练时 BatchNorm 使用本地 batch statistics；
全局推理依赖 moving_mean / moving_variance；
当前 FedAvg 状态处理方式未正确维护这些 non-trainable statistics。
```

解决：阶段 1A 三组基线统一删除 `BatchNormalization`。

旧的带 BN 实验建议保留，后面用于 FedBN 研究或失败案例分析。

---

## 15. 需要注意的数据审计问题

GTSRB 中同一个真实交通标志可能对应一组连续帧（track）。当前训练/验证拆分和 IID 客户端划分以单张图片为最小单位，可能出现：

```text
同一 track 的部分帧进入训练集；
同一 track 的其他帧进入验证集；
同一 track 的不同帧被分给不同客户端。
```

这可能使验证集比官方测试集更容易，并造成：

```text
FedAvg Validation Accuracy: 99.5652%
FedAvg Test Accuracy: 92.6049%
```

因此：

1. 主结果应以官方测试集准确率为核心；
2. 验证准确率主要用于选择 checkpoint；
3. 后续更严格实验建议增加 track-level 划分；
4. 阶段 1A 当前图片级 IID 划分仍可用于验证联邦流程。

---

## 16. 阶段 1A 当前完成状态

| 内容 | 状态 |
|---|---|
| GTSRB 数据检查 | 完成 |
| 43 类分类数据加载 | 完成 |
| 5 客户端 IID 划分 | 完成 |
| TensorFlow 数据流水线 | 完成 |
| TFF 数据格式转换 | 完成 |
| Centralized | 完成 |
| Local-only | 完成 |
| Weighted FedAvg | 完成 |
| 按样本数加权聚合 | 完成 |
| 每轮指标记录 | 完成 |
| 全局模型更新 | 完成 |
| checkpoint 保存 | 完成 |
| checkpoint 恢复 | 完成 |
| 通信量估算 | 完成 |
| 三组基线结果汇总 | 完成 |
| Track-level 数据审计 | 建议补充 |
| 多随机种子重复实验 | 后续正式论文实验补充 |

---

## 17. 推荐归档文件

建议长期保留：

```text
data/gtsrb/federated_iid_5clients_seed20260720/
outputs/stage1a/centralized_nobn_seed20260720/
outputs/stage1a/local_only_nobn_seed20260720/
outputs/stage1a/fedavg_nobn_seed20260720/
src/gtsrb_1a/
```

最重要的模型：

```text
Centralized:
outputs/stage1a/centralized_nobn_seed20260720/checkpoints/best.h5

Local-only:
outputs/stage1a/local_only_nobn_seed20260720/client_*/checkpoints/best.h5

FedAvg:
outputs/stage1a/fedavg_nobn_seed20260720/global_models/best_global_model.h5
```

最重要的结果文件：

```text
Centralized:
test_metrics.json
metrics.csv
per_class_accuracy.csv
confusion_matrix.csv

Local-only:
local_only_summary.csv
local_only_aggregate.json
client_*/test_metrics.json

FedAvg:
test_metrics.json
metrics.csv
per_class_accuracy.csv
confusion_matrix.csv
config.json
```

---

## 18. 后续阶段建议

### 阶段 1B：更严格的数据与重复性验证

- Track-level IID 划分；
- 3 个随机种子重复实验；
- 报告 mean ± std；
- 固定训练预算与充分收敛结果同时报告。

### 阶段 2：Non-IID 与算法对比

可逐步加入：

```text
类别 Non-IID
场景/图像特征 Non-IID
FedAvg
FedProx
FedBN
SCAFFOLD
MOON
```

优先建议：

```text
IID FedAvg
→ 类别 Non-IID FedAvg
→ Non-IID FedProx
→ 带 BN 模型上的 FedAvg 与 FedBN
```

### 阶段 3：应用验证

将训练得到的全局模型下发给模拟客户端，完成：

```text
交通标志识别
→ 识别结果映射为规则
→ Stop：停车
→ Turn Left：左转
→ Turn Right：右转
→ Speed Limit：限速
```

---

## 19. 一句话总结

阶段 1A 已经完成了从 GTSRB 数据清洗、5 客户端 IID 划分、43 类 CNN 分类，到 TFF Weighted FedAvg 多轮训练、模型聚合、checkpoint 恢复、通信量统计和三组基线对比的完整流程。最终结果表明，Local-only 因单客户端数据不足表现明显较弱，而 FedAvg 在不共享原始图片的情况下，经过充分通信训练后达到与 Centralized 接近的官方测试性能。
