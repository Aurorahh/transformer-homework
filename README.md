# 从零开始构建 Transformer (Seq2Seq 机器翻译)

本项目根据作业的要求，使用 PyTorch 从零开始实现了一个完整的 Encoder-Decoder Transformer 模型。该模型在 IWT2017 英德机器翻译任务上 进行了训练和评估。

## 📍 项目结构

```
transformer-from-scratch/
|
├── src/
|   ├── model.py           \# Transformer (Encoder, Decoder, Attention) 核心实现
|   ├── dataset.py         \# Hugging Face `datasets` 数据加载与预处理 (spaCy 分词)
|   └── utils.py           \# 辅助函数 (绘图, BLEU评估, 翻译, V随机种子设置)
|
├── scripts/
|   ├── run.sh             \# (主实验) 运行基线和位置编码消融实验
|   └── run\_sensitivity.sh \# (挑战任务) 运行超参数敏感性分析
|
├── results/
|   ├── training\_curves/   \# 保存训练/验证损失曲线图
|   └── tables/            \# 保存所有实验的量化结果 (Loss, PPL, BLEU)
|
├── train.py                \# 主训练/评估脚本
├── requirements.txt        \# Python 依赖
└── README.md               \# 本文档
```

## 🚀 硬件要求与环境设置

本项目已在以下环境中成功测试：

  * **GPU**: NVIDIA RTX 3090 (24GB VRAM)
  * **CPU**: (Linux 系统)
  * **Python**: 3.11
  * **核心依赖**: `torch`, `torchtext`, `datasets`, `spacy`

**部署步骤:**

1.  **创建 Conda 环境 (推荐使用 `pip` 进行安装)**

    ```bash
    # 我们的调试证明，使用纯 pip 和指定 python 版本是最佳实践
    conda create -n transformer_env python=3.11 -y
    conda activate transformer_env
    ```

2.  **安装 PyTorch (cu121 对应 3090/4090 系列)**

    ```bash
    pip3 install torch torchtext --index-url https://download.pytorch.org/whl/cu121
    ```

3.  **安装受控版本的依赖（一键命令）**
    为了确保 100% 兼容性 (避免 NumPy 2.x 和 `huggingface_hub` 冲突)，请运行：

    ```bash
    pip install "numpy<2.0" "datasets<2.16.0" "huggingface_hub<0.20.0" spacy tqdm matplotlib
    ```

4.  **下载 `spaCy` 语言模型**

    ```bash
    python -m spacy download de_core_news_sm
    python -m spacy download en_core_web_sm
    ```

## 🏃 运行与复现实验

PDF 明确要求提供**包含随机种子**的精确复现命令行。我们通过 shell 脚本提供。

#### 1\. 主实验 (基线 vs. 消融)

此脚本将运行基线模型和移除了位置编码 的消融模型。

```bash
# 赋予权限
chmod +x scripts/run.sh

# 运行 (预计在 RTX 3090 上耗时约 80-90 分钟)
./scripts/run.sh
```

#### 2\. 挑战任务 (超参数敏感性分析)

此脚本将运行不同注意力头数（4, 16）的模型，用于与基线（8头）进行对比。

```bash
# 赋予权限
chmod +x scripts/run_sensitivity.sh

# 运行
./scripts/run_sensitivity.sh
```

## 📊 预期结果

所有实验完成后，请检查 `results/` 目录：

1.  **`results/tables/experiment_summary.md`**: 此文件将包含一个 Markdown 表格，量化对比所有运行（基线、消融、敏感性）的最终 PPL 和 BLEU 分数。
2.  **`results/training_curves/`**: 此目录将包含所有实验的 `_loss_curve.png` 文件，可视化地展示训练过程。
