# 反诈 | Fanzha Qwen3.5-4B GRPO v2

中文反诈宣传模型的 GRPO 实验代码、规则奖励和本地监督页面。代码按 MIT 许可。LoRA 模型单独发布于 [Shokou1/fanzha-qwen3.5-4b-grpo-v2](https://huggingface.co/Shokou1/fanzha-qwen3.5-4b-grpo-v2)，基座许可、修改说明和限制见模型卡与 NOTICE。

本轮 40 步 GRPO 没有超过 SFT v2 基线；最终 adapter 作为未选中的实验权重发布。结果见 [实验记录](EXPERIMENT.md)。

## 文件与数据

本仓库提供训练与奖励实现、最小 SFT 特征处理工具、依赖清单和本地监控。原始训练数据、验证题、生成回答、训练日志和模型权重不随代码仓库发布。复现实验需要自行准备经许可审核的数据，并取得符合 PEFT 格式的初始化 SFT adapter。

训练 prompt JSONL 支持 `system`、`instruction`、`task` 和 `meta.required`，或包含 user 的 `messages`。固定生成评测 prompt 应与训练 prompt 完全分离。SFT 验证行支持 `system`、`instruction`、`output`，或包含 assistant 答案的 `messages`。请自行确认数据来源许可、署名、隐私与再分发权。

## 环境

实验环境为 Python 3.12、CUDA 12.8、单张 24 GB GPU。核心依赖版本见 `requirements.txt`。PyTorch CUDA wheel 请按本机驱动和 CUDA 环境从 PyTorch 官方渠道安装，再安装其余依赖。

## 训练

```bash
python -m pip install -r requirements.txt
python scripts/train_grpo_fanzha_v2.py \
  --base Qwen/Qwen3.5-4B \
  --policy-adapter /path/to/sft-adapter \
  --prompts-file /path/to/train.jsonl \
  --val-file /path/to/sft-validation.jsonl \
  --eval-prompts /path/to/eval-prompts.jsonl \
  --out /path/to/grpo-output \
  --steps 40 --group 4 --seq 1024 --max-new 256 \
  --lr 1e-6 --kl-beta 0.05 --save-every 10 --eval-every 10
```

GRPO 使用固定初始化 reference、逐 token clipped objective 和 k3 KL，每条 completion 按实际长度平均，temperature=1、top-p=1、top-k=0，关闭 dropout。训练 prompt 与固定生成评测 prompt 会自动做完整消息精确去重。每 10 步保存并评估；按固定评测奖励选优，前提是 SFT 验证 CE 不高于初始化值的 105%。

## 本地监督页面

```bash
python scripts/serve_monitor.py --out /path/to/grpo-output --port 8787
```

打开 http://127.0.0.1:8787/grpo_v2/。服务只监听 localhost，只读取训练输出目录中的摘要和指标。

## 结果解释

奖励函数主要依赖词面规则，可能误判安全性、漏掉错误建议或鼓励堆叠关键词。有限题集上的奖励、自动 `valid` 和 SFT CE 都不能证明实际场景中的质量、安全、泛化或法律合规。该实验的 16 题固定验证集只是探索性验证，且部分题目曾用于旧版 GRPO；请另做独立评测和人工安全审查。

训练输出可能包含输入、回答和检查点；公开前应审核敏感内容并确认相应授权。
