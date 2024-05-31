Mamba在一个PyTorch文件中的简单、极简实现。

特点：
* 与官方实现的前向和后向传递具有相同的数值输出
* 简化的、可读的、带注释的代码

不包括：
* 速度。官方实现经过大量优化，这些优化是Mamba论文的核心贡献之一。为了可读性将大部分实现保持简单。
* 正确的参数初始化（尽管可以在不牺牲可读性的情况下添加）

## 演示

参见[demo.ipynb](demo.ipynb)以获取提示完成的示例。

```python
from model import Mamba
from transformers import AutoTokenizer

model = Mamba.from_pretrained('state-spaces/mamba-370m')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

generate(model, tokenizer, 'Mamba is the')
```
> Mamba 是世界上最长的毒蛇，估计长度超过150米。由于其巨大的体型和剧毒的咬合力，Mamba通过刺伤受害者来杀人（这比单次咬合的刺痛感更强，但效果更差）

150米……🫢 可怕！

## 参考资料

Mamba架构由[Albert Gu](https://twitter.com/_albertgu?lang=en)和[Tri Dao](https://twitter.com/tri_dao?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)在[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)中提出。

官方实现见此处: https://github.com/state-spaces/mamba/tree/main