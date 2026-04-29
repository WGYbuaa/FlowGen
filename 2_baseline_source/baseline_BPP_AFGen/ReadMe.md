# 基于 Transformer 的 Non-Graph Baseline

本目录提供一个可复现的 Transformer baseline，用于论文中的两个模块：

- `BPP`：Branch Point Prediction
- `AFGen`：Alternative Flow Generation

这版 baseline 不再包含不同 `context scope` 的对比实验逻辑，AFGen 固定使用 **local / 1-hop context**。

## 文件说明

- `main.py`：训练与评估入口
- `seq_model.py`：Transformer 编码器、BPP 预测头、AFGen Transformer 解码器
- `run.sh`：运行单个 baseline 实验的脚本

## baseline 思路

本 baseline 不依赖图结构编码：

1. 仍然读取 `.pt` 中已有的节点特征 `x`
2. 仅取 basic flow 对应节点，按顺序送入 `Transformer Encoder`
3. 将编码后的 BF 节点表示散回原子图节点位置，保持索引体系不变
4. `BPP` 仍然做与原实现一致的多标签预测
5. `AFGen` 改为 `Transformer Decoder`
6. AFGen 固定使用 branch point 对应的 **local / 1-hop context**

可以把它理解为：

> 保留原实验数据、标签与指标，只把图编码器替换成 Transformer。

## 与原代码保持一致的部分

### 1. 输入文件

- `ERNIE_pub_withbp_20.pt`
- `Ernie_pub.json`
- `ERNIE_ncet_withbp_20.pt`
- `ERNIE_ncet_map.json`

### 2. 数据切分

- `pub`：train `0:30`，eval `30:33`，test `33:`
- `ncet`：train `0:338`，eval `338:341`，test `341:`

### 3. BPP 标签与输出

- 监督目标 `branch_target_full`
- 分支点二值目标 `is_branch_point_target`
- 每个 BF node 输出一个 `vocab_size` 维 logit 向量

### 4. AFGen 标签与输出

- 使用：
  - `af_sequences_input`
  - `af_sequences_target`
  - `af_start_nodes_for_decoder`
  - `af_contexts`
- 只对 `AF` 序列做生成与评估

### 5. BPP 评估方式

- 每个 BF node 的 branch score 取 `max(sigmoid(branch_logits), dim=1)`
- 输出：
  - micro `Precision / Recall / F1`
  - macro `Precision / Recall / F1`

### 6. AFGen 评估方式

- 对每一步输出的 vocab logits 展平
- 经过 `sigmoid`
- 搜索最优阈值
- 计算：
  - `Precision`
  - `Recall`
  - `F1`
  - `AUC`


## 如何运行

### 1. 直接运行

跑 `ncet`：

```bash
cd /root/autodl-tmp/ASSAM/program/new/baseline
python main.py --dataset ncet
```

跑 `pub`：

```bash
cd /root/autodl-tmp/ASSAM/program/new/baseline
python main.py --dataset pub
```

### 2. 使用脚本运行

跑 `ncet`：

```bash
cd /root/autodl-tmp/ASSAM/program/new/baseline
bash run.sh ncet
```

跑 `pub`：

```bash
cd /root/autodl-tmp/ASSAM/program/new/baseline
bash run.sh pub
```

## 参数说明

- `--dataset`
  - `pub` 或 `ncet`
- `--selection_metric`
  - `decoder_f1`：按 AFGen 的 F1 选 checkpoint
  - `branch_micro_f1`：按 BPP 的 micro-F1 选 checkpoint
  - `sum`：按 `BPP micro-F1 + AFGen F1` 选 checkpoint

## 输出文件

运行后会产生：

- checkpoint：
  - `baseline/outputs/best_transformer_*.pth`
- 测试结果汇总：
  - `baseline/outputs/summary_transformer_*.json`
- 日志：
  - `baseline/logs/transformer_*.log`

## 这版 baseline 的定位

这版模型的目的，是给 BPP 和 AFGen 提供一个：

- 非 prompt-only
- 可复现
- task-specific
- 不依赖图消息传递

的 Transformer baseline。

它的作用是帮助说明：如果去掉图结构传播能力，只保留序列建模能力，模型表现会如何变化。
