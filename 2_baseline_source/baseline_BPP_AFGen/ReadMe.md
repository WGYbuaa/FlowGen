# 基于 Transformer 的 Non-Graph Baseline

本目录提供一个可复现的 Transformer baseline，用于论文中的两个模块：

- `BPP`：Branch Point Prediction
- `AFGen`：Alternative Flow Generation

这版 baseline 不再包含不同 `context scope` 的对比实验逻辑，AFGen 固定使用 **local / 1-hop context**。你前面针对 RQ7 的上下文范围比较已经单独完成，因此这里保留单一设定即可。

## 文件说明

- `main.py`：训练与评估入口
- `seq_model.py`：Transformer 编码器、BPP 预测头、AFGen Transformer 解码器
- `run.sh`：运行单个 baseline 实验的脚本

## baseline 思路

这版 baseline 的目标是：**只替换模型骨架，不改任务接口和评估协议**。

原始方法依赖图结构编码；本 baseline 改为：

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

仍然使用以下文件：

- `ERNIE_pub_withbp_20.pt`
- `Ernie_pub.json`
- `ERNIE_ncet_withbp_20.pt`
- `ERNIE_ncet_map.json`

### 2. 数据切分

- `pub`：train `0:30`，eval `30:33`，test `33:`
- `ncet`：train `0:338`，eval `338:341`，test `341:`

### 3. BPP 标签与输出

- 监督目标仍是 `branch_target_full`
- 分支点二值目标仍是 `is_branch_point_target`
- 每个 BF node 仍输出一个 `vocab_size` 维 logit 向量

### 4. AFGen 标签与输出

- 仍使用：
  - `af_sequences_input`
  - `af_sequences_target`
  - `af_start_nodes_for_decoder`
  - `af_contexts`
- 仍然只对 `AF act` 序列做生成与评估

### 5. BPP 评估方式

与原逻辑一致：

- 每个 BF node 的 branch score 取 `max(sigmoid(branch_logits), dim=1)`
- 输出：
  - micro `Precision / Recall / F1`
  - macro `Precision / Recall / F1`

### 6. AFGen 评估方式

与原逻辑一致：

- 对每一步输出的 vocab logits 展平
- 经过 `sigmoid`
- 搜索最优阈值
- 计算：
  - `Precision`
  - `Recall`
  - `F1`
  - `AUC`

## 环境与依赖

根据 `../environment.yaml`，这版 baseline 需要的核心依赖已经在环境里：

- `python=3.10`
- `torch`
- `torch-geometric`
- `torchmetrics`
- `numpy`
- `tqdm`

如果服务器上已经按 `environment.yaml` 建好环境，一般不需要额外补装新包。

你当前给出的服务器路径是：

- `main.py`：`/root/autodl-tmp/ASSAM/program/new/baseline/main.py`
- 数据目录：`/root/autodl-tmp`

因此这版代码已经把 `--data_dir` 的默认值设成了：

```text
/root/autodl-tmp
```

## 如何创建或激活环境

如果服务器上还没有环境：

```bash
cd /path/to/4_baseline_BPP_AFGen
conda env create -f environment.yaml
conda activate bertTest
```

如果环境已经存在：

```bash
cd /path/to/4_baseline_BPP_AFGen
conda activate bertTest
```

## 运行前注意事项

### 1. `batch_size` 必须为 1

这套数据管线默认“一个 batch = 一个 subgraph”，因此：

- 必须保持 `--batch_size 1`
- 不要改成大于 1

### 2. `hidden_dim` 必须能被 `num_heads` 整除

默认参数是：

- `hidden_dim=128`
- `num_heads=8`

需要满足：

```text
hidden_dim % num_heads == 0
```

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

## 常用命令行参数

```bash
python main.py \
  --dataset ncet \
  --data_dir /root/autodl-tmp \
  --output_dir /root/autodl-tmp/ASSAM/program/new/baseline/outputs \
  --num_epochs 2000 \
  --learning_rate 0.001 \
  --batch_size 1 \
  --hidden_dim 128 \
  --num_heads 8 \
  --num_encoder_layers 2 \
  --num_decoder_layers 2 \
  --dropout 0.1 \
  --hard_negative_ratio 50 \
  --early_stop_epoch 10 \
  --selection_metric decoder_f1
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
