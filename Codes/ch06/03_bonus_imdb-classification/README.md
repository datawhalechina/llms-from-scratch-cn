# 对 50k IMDB 电影评论的情感进行分类的附加实验

&nbsp;
## Step 1: 安装依赖

通过下列命令安装额外的依赖项

```bash
pip install -r requirements-extra.txt
```

&nbsp;
## Step 2: 下载数据集

这些代码使用 IMDb 中的 50k 电影评论来预测电影评论是正面还是负面。 ([数据集](https://ai.stanford.edu/~amaas/data/sentiment/)) 

运行以下代码来创建`train.csv`, `validation.csv`, 和 `test.csv`数据集:

```bash
python download-prepare-dataset.py
```


&nbsp;
## Step 3: 运行模型

主要章节中使用的 124M GPT-2 模型，从预训练权重开始，仅训练最后一个 Transformer 块加上输出层：

```bash
python train-gpt.py
```

```
Ep 1 (Step 000000): Train loss 2.829, Val loss 3.433
Ep 1 (Step 000050): Train loss 1.440, Val loss 1.669
Ep 1 (Step 000100): Train loss 0.879, Val loss 1.037
Ep 1 (Step 000150): Train loss 0.838, Val loss 0.866
...
Ep 1 (Step 004300): Train loss 0.174, Val loss 0.202
Ep 1 (Step 004350): Train loss 0.309, Val loss 0.190
Training accuracy: 88.75% | Validation accuracy: 91.25%
Ep 2 (Step 004400): Train loss 0.263, Val loss 0.205
Ep 2 (Step 004450): Train loss 0.226, Val loss 0.188
...
Ep 2 (Step 008650): Train loss 0.189, Val loss 0.171
Ep 2 (Step 008700): Train loss 0.225, Val loss 0.179
Training accuracy: 85.00% | Validation accuracy: 90.62%
Ep 3 (Step 008750): Train loss 0.206, Val loss 0.187
Ep 3 (Step 008800): Train loss 0.198, Val loss 0.172
...
Training accuracy: 96.88% | Validation accuracy: 90.62%
Training completed in 18.62 minutes.

Evaluating on the full datasets ...

Training accuracy: 93.66%
Validation accuracy: 90.02%
Test accuracy: 89.96%
```

---

一个 66M 参数的编码器模型 [DistilBERT](https://arxiv.org/abs/1910.01108)（从 340M 参数 BERT 模型蒸馏而来），从预训练权重开始，仅训练最后一个 Transformer 块和输出层：


```bash
python train-bert-hf.py
```

```
Ep 1 (Step 000000): Train loss 0.693, Val loss 0.697
Ep 1 (Step 000050): Train loss 0.532, Val loss 0.596
Ep 1 (Step 000100): Train loss 0.431, Val loss 0.446
...
Ep 1 (Step 004300): Train loss 0.234, Val loss 0.351
Ep 1 (Step 004350): Train loss 0.190, Val loss 0.222
Training accuracy: 88.75% | Validation accuracy: 88.12%
Ep 2 (Step 004400): Train loss 0.258, Val loss 0.270
Ep 2 (Step 004450): Train loss 0.204, Val loss 0.295
...
Ep 2 (Step 008650): Train loss 0.088, Val loss 0.246
Ep 2 (Step 008700): Train loss 0.084, Val loss 0.247
Training accuracy: 98.75% | Validation accuracy: 90.62%
Ep 3 (Step 008750): Train loss 0.067, Val loss 0.209
Ep 3 (Step 008800): Train loss 0.059, Val loss 0.256
...
Ep 3 (Step 013050): Train loss 0.068, Val loss 0.280
Ep 3 (Step 013100): Train loss 0.064, Val loss 0.306
Training accuracy: 99.38% | Validation accuracy: 87.50%
Training completed in 16.70 minutes.

Evaluating on the full datasets ...

Training accuracy: 98.87%
Validation accuracy: 90.98%
Test accuracy: 90.81%
```

---

一个355M 参数量的编码器模型 [RoBERTa](https://arxiv.org/abs/1907.11692) ，从预训练权重开始，仅训练最后一个 Transformer 块和输出层：


```bash
python train-bert-hf.py --bert_model roberta
```

---

一个scikit-learn Logistic 回归模型作为基线。

```bash
python train-sklearn-logreg.py
```

```
Dummy classifier:
Training Accuracy: 50.01%
Validation Accuracy: 50.14%
Test Accuracy: 49.91%


Logistic regression classifier:
Training Accuracy: 99.80%
Validation Accuracy: 88.60%
Test Accuracy: 88.84%
```
