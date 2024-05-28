# 额外实验

下表添加了一些实验来回答有关各种设计选择的其他问题。 第一行使用与主要章节相同的设置并用作参考。
例如，

- 比较第 1 行和第 2 行回答了以下问题：“当我们训练最后一个或第一个标记时，性能差异是什么？”；
- 比较第 1 行和第 3 行回答了以下问题：“当我们只训练最后一层而不是最后一个块时，性能差异是什么？”；
- 等等。

&nbsp;

|      | Model              | Weights    | Trainable token | Trainable layers | Context length          | Training acc | Validation acc | Test acc | Training time | CPU/GPU |
| ---- | ------------------ | ---------- | --------------- | ---------------- | ----------------------- | ------------ | -------------- | -------- | ------------- | ------- |
| 1    | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120)                                | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |
| 2    | gpt2-small (124M)  | pretrained | first                    | last_block       | longest train ex. (120)                                | 78.46%       | 80.54%         | 75.00%   | 0.28 min      | A100    |
| 3    | gpt2-small (124M)  | pretrained | last                     | last_layer       | longest train ex. (120)                                | 78.65%       | 79.87%         | 72.00%   | 0.25 min      | A100    |
| 4    | gpt2-small (124M)  | pretrained | last                     | last_two_blocks  | longest train ex. (120)                                | 98.85%       | 98.66%         | 98.33%   | 0.33 min      | A100    |
| 5    | gpt2-small (124M)  | pretrained | last                     | all              | longest train ex. (120)                                | 99.62%       | 96.64%         | 96.67%   | 0.69 min      | A100    |
| 6    | gpt2-medium (355M) | pretrained | last                     | last_block       | longest train ex. (120)                                | 87.50%       | 91.28%         | 84.67%   | 0.75 min      | A100    |
| 7    | gpt2-large (774M)  | pretrained | last                     | last_block       | longest train ex. (120)                                | 99.52%       | 98.66%         | 96.67%   | 1.50 min      | A100    |
| 8    | gpt2-xl (1558M)    | pretrained | last                     | last_block       | longest train ex. (120)                                | 99.81%       | 99.33%         | 98.33%   | 2.83 min      | A100    |
| 9    | gpt2-small (124M)  | random     | last                     | all              | longest train ex. (120)                                | 100%         | 96.64%         | 93.67%   | 0.69 min      | A100    |
| 10   | gpt2-small (124M)  | pretrained | last                     | LoRA             | longest train ex. (120)                                | 100.00%      | 97.32%         | 96.67%   | 0.75 min      | A100    |
| 11   | gpt2-small (124M)  | pretrained | last                     | last_block       | context length (1024)                                  | 83.08%       | 87.92%         | 78.33%   | 2.46 min      | A100    |
| 12   | gpt2-small (124M)  | pretrained | last                     | last_block       | variable: no padding (batch size 1)                    | 100.00%      | 98.66%         | 98.00%   | 1.75 min      | A100    |
| 13   | gpt2-small (124M)  | pretrained | last                     | last_block       | variable: no padding (batch size 8)                    | 99.33%       | 98.66%         | 98.33%   | 1.70 min      | A100    |
| 14   | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120); but no causal mask            | 99.23%       | 98.66%         | 95.33%   | 0.29 min      | A100    |
| 15   | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120) and `ignore_index` for padding | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |


&nbsp;

## 使用方法

您可以使用以下代码来重现实验：

- Row 1: `python additional-experiments.py`
- Row 2: `python additional-experiments.py --trainable_token_pos first`
- Row 3: `python additional-experiments.py --trainable_layers last_layer`
- Row 4: `python additional-experiments.py --trainable_layers last_two_blocks`
- Row 5: `python additional-experiments.py --trainable_layers all`
- Row 6: `python additional-experiments.py --model_size "gpt2-medium (355M)"`
- Row 7: `python additional-experiments.py --model_size "gpt2-large (774M)"`
- Row 8: `python additional-experiments.py --model_size "gpt2-xl (1558M)"`
- Row 9: `python additional-experiments.py --weights random --trainable_layers all`
- Row 10: `python additional-experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 16`
- Row 11: `python additional-experiments.py --context_length "model_context_length"`
- Row 12: `python additional-experiments.py --no_padding --batch_size 1`
- Row 13: `python additional-experiments.py --no_padding --batch_size 1 --accumulation_steps 8`
- Row 14: `python additional-experiments.py --disable_causal_mask`
- Row 15: `python additional-experiments.py --ignore_index 50256`

我特意将 LLM 和数据集保持得较小，因此，如果您无法使用 GPU，您可以在 MacBook Air M3 等普通笔记本电脑上运行大约 15 分钟的训练。

&nbsp;

## 解释
1. **训练最后一个输出标记与第一个输出标记（第 1 行与第 2 行）**：与第一个输出标记相比，训练最后一个输出标记会带来更好的性能。由于因果自注意力掩模，这种改进是可以预期的。
2. **训练最后一个 Transformer 块与最后一层（第 1 行与第 3 行）**：训练整个最后一个 Transformer 块也比仅训练最后一层获得更好的结果。
3. **训练所有层与最后一个 Transformer 块（第 1 行与第 4 行）**：训练所有层比仅训练最后一个 Transformer 块显示出约 2% 的适度改进，但它需要的时间几乎是三倍的训练时间。
4. **训练最后一个 Transformer 块与所有层（第 1 行与第 5 行）**：训练所有层比仅训练最后一个 Transformer 块显示出约 2% 的适度改进，但就时间而言，它需要几乎三倍的时间训练持续时间。 此外，它仅训练 12 个变压器块中的最后两个，其性能也不佳。
5. **使用更大的预训练模型（第 1 行与第 5 行，以及第 1 行与第 7 行和第 8 行）**：采用 3 倍大的预训练模型会导致更差的结果。然而，正如预期的那样，与初始模型相比，使用大 5 倍的模型可以提高性能。同样，12 倍大的模型进一步提高了预测性能。（中等模型可能没有经过很好的预训练，或者特定的微调配置对该模型效果不佳。）
6. **使用具有随机权重的模型与预训练权重（第 1 行与第 9 行）**：使用具有随机权重的模型产生的结果仅比使用预训练权重稍差 1.3%。
7. **使用 LoRA（低阶适应）与训练所有层（第 10 行与第 5 行）**：保持模型冻结并添加可训练的 LoRA 层是训练所有模型参数的可行替代方案(请参阅[附录 E](../../appendix-E/01_main-chapter-code/appendix-E.ipynb))，甚至可以将性能提高 1%。 从使用 LoRA 时训练和验证准确率之间的差距降低约 1% 可以看出，这可能是由于过度拟合较少。此外，使用 LoRA 的速度也稍快一些，因为需要更新的参数较少。
8. **将输入填充到完整上下文长度与最长训练示例（第 1 行与第 11 行）**：将输入填充到完整支持的上下文长度结果明显更差。
9. **填充与无填充（第 1 行与第 12 行和第 13 行）**：`--no_padding` 选项禁用数据集中的填充，这需要使用批量大小 1 来训练模型，因为输入具有变量 长度。 这会带来更好的测试精度，但需要更长的训练时间。 在第 12 行中，我们另外启用了 8 个步骤的梯度累积，以实现与其他实验相同的批量大小，这有助于减少过度拟合并略微提高测试集的准确性。
10. **禁用因果注意掩码（第 1 行与第 14 行）**：禁用多头注意模块中使用的因果注意掩码。这意味着所有Token都可以参加所有其他Token。 与带有因果掩模的 GPT 模型相比，模型精度略有提高。
11. **忽略损失和反向传播中的填充索引（第 1 行与第 15 行）**：设置 `--ignore_index 50256` 会排除 PyTorch 中 `cross_entropy` 损失函数中的 `|endoftext|` 填充标记。 在这种情况下，它没有任何效果，因为我们替换了输出层，以便二元分类示例的标记 ID 为 0 或 1。 然而，当第 7 章中的指令微调模型时，此设置很有用。