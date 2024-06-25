---
language:
- en
- zh
tags:
- MiniCPM
- ModelBest
- THUNLP
---


<div align="center">
<h1>
  MiniCPM
</h1>
</div>

<p align="center">
<a href="https://shengdinghu.notion.site/MiniCPM-c805a17c5c8046398914e47f0542095a?pvs=4" target="_blank">MiniCPM 技术报告</a><a href="https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20?pvs=4" target="_blank"> Technical Report</a> |
<a href="https://github.com/OpenBMB/OmniLMM/" target="_blank">OmniLMM 多模态模型 Multi-modal Model</a> |
<a href="https://luca.cn/" target="_blank">CPM-C 千亿模型试用 ~100B Model Trial </a> 
</p>

MiniCPM 是面壁与清华大学自然语言处理实验室共同开源的系列端侧语言大模型，主体语言模型 MiniCPM-2B 仅有 24亿（2.4B）的非词嵌入参数量。
- 经过 SFT 后，MiniCPM 在公开综合性评测集上，MiniCPM 与 Mistral-7B相近（中文、数学、代码能力更优），整体性能超越 Llama2-13B、MPT-30B、Falcon-40B 等模型。
- 经过 DPO 后，MiniCPM 在当前最接近用户体感的评测集 MTBench上，MiniCPM-2B 也超越了 Llama2-70B-Chat、Vicuna-33B、Mistral-7B-Instruct-v0.1、Zephyr-7B-alpha 等众多代表性开源大模型。
- 以 MiniCPM-2B 为基础构建端侧多模态大模型 MiniCPM-V，整体性能在同规模模型中实现最佳，超越基于 Phi-2 构建的现有多模态大模型，在部分评测集上达到与 9.6B Qwen-VL-Chat 相当甚至更好的性能。
- 经过 Int4 量化后，MiniCPM 可在手机上进行部署推理，流式输出速度略高于人类说话速度。MiniCPM-V 也首次跑通了多模态大模型在手机上的部署。
- 一张1080/2080可高效参数微调，一张3090/4090可全参数微调，一台机器可持续训练 MiniCPM，二次开发成本较低。

我们将完全开源MiniCPM-2B的模型参数供学术研究和有限商用，以及训练过程中的所有Checkpoint和大部分非专有数据供模型机理研究。

- 基于MiniCPM-2B的指令微调与人类偏好对**MiniCPM-2B-SFT/DPO。**
- 基于MiniCPM-2B的多模态模型**MiniCPM-V**，能力超越基于Phi-2的同参数级别多模态模型**。**
- MiniCPM-2B-SFT/DPO的Int4量化版**MiniCPM-2B-SFT/DPO-Int4。**
- 基于MLC-LLM、LLMFarm开发的MiniCPM手机端程序，**文本及多模态模型均可在手机端进行推理。**


MiniCPM is an End-Size LLM developed by ModelBest Inc. and TsinghuaNLP, with only 2.4B parameters excluding embeddings.

- MiniCPM has very close performance compared with Mistral-7B on open-sourced general benchmarks with better ability on Chinese, Mathmetics and Coding after SFT. The overall performance exceeds Llama2-13B, MPT-30B, Falcon-40B, etc.
- After DPO, MiniCPM outperforms Llama2-70B-Chat, Vicuna-33B, Mistral-7B-Instruct-v0.1, Zephyr-7B-alpha, etc. on MTBench.
- MiniCPM-V, based on MiniCPM-2B, achieves the best overall performance among multimodel models of the same scale, surpassing existing multimodal large models built on Phi-2 and achieving performance comparable to or even better than 9.6B Qwen-VL-Chat on some tasks.
- MiniCPM can be deployed and infer on smartphones, and the speed of streaming output is relatively higher than the verbal speed of human. MiniCPM-V is the first multi-modal models that can be deployed on smartphones.
- The cost of developing based on MiniCPM is low. Parameter efficient finetuning can be conducted with a single 1080/2080 GPU and full parameter finetuning can be conducted with a 3090/4090 GPU.

We release all model parameters for research and limited commercial use. We also release all the checkpoint during training and most public training data for research on model mechanism.

- SFT and DPO version based on MiniCPM-2B and human preference: **MiniCPM-2B-SFT/DPO**
- The multi-modal model **MiniCPM-V** based on MiniCPM-2B, which outperforms models with similar size, i.e., Phi-2
- The INT4 quantized version **MiniCPM-2B-SFT/DPO-Int4** based on MiniCPM-2B-SFT/DPO
- Mobile phone application based on MLC-LLM and LLMFarm. Both language model and multimodel model can conduct inference on smartphones.

### 评测结果 Evaluation Results

  详细的评测结果位于[github仓库](https://github.com/OpenBMB/MiniCPM?tab=readme-ov-file#%E8%AF%84%E6%B5%8B%E7%BB%93%E6%9E%9C)

  Detailed evaluation results are in [github repo](https://github.com/OpenBMB/MiniCPM/blob/main/README-en.md#evaluation-results)

  注意：我们发现使用Huggingface生成质量略差于vLLM，因此推荐使用vLLM进行测试。我们正在排查原因。

  Notice: We discovered that the quality of Huggingface generation is slightly lower than vLLM, thus benchmarking using vLLM is recommended. 
  We are investigating the cause now.

### 局限性 Limitations

- 受限于模型规模，模型可能出现幻觉性问题。其中由于DPO模型生成的回复内容更长，更容易出现幻觉。我们也将持续进行MiniCPM模型的迭代改进；
- 为了保证在学术研究用途上模型的通用性，我们未对模型进行任何身份认同训练。同时由于我们用ShareGPT开源语料作为部分训练数据，模型可能会输出类似GPT系列模型的身份认同信息；
- 受限于模型规模，模型的输出受到提示词（prompt）的影响较大，可能多次尝试产生不一致的结果；
- 受限于模型容量，模型的知识记忆较不准确，后续我们将结合RAG方法来增强模型的知识记忆能力。

- Due to limitations in model size, the model may experience hallucinatory issues. As DPO model tend to generate longer response, hallucinations are more likely to occur. We will also continue to iterate and improve the MiniCPM model.
- To ensure the universality of the model for academic research purposes, we did not conduct any identity training on the model. Meanwhile, as we use ShareGPT open-source corpus as part of the training data, the model may output identity information similar to the GPT series models.
- Due to the limitation of model size, the output of the model is greatly influenced by prompt words, which may result in inconsistent results from multiple attempts.
- Due to limited model capacity, the model's knowledge memory is not accurate. In the future, we will combine the RAG method to enhance the model's knowledge memory ability.

## 模型下载 Download
 
  | HuggingFace | ModelScope | WiseModel |
  |-------------|------------|-----------|
  |[sft-bf16](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)|[sft-bf16](https://modelscope.cn/models/OpenBMB/miniCPM-bf16)|[sft-bf16](https://wisemodel.cn/models/OpenBMB/miniCPM-bf16)
  |[sft-fp32](https://huggingface.co/openbmb/MiniCPM-2B-sft-fp32)|[sft-fp32](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-sft-fp32)|[sft-fp32](https://wisemodel.cn/models/OpenBMB/miniCPM-dpo-fp32)
  |[dpo-bf16](https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16)|[dpo-bf16](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-dpo-bf16/summary)|[dpo-bf16](https://wisemodel.cn/models/OpenBMB/MiniCPM-2B-dpo-bf16)
  |[dpo-fp16](https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp16)|[dpo-fp16](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-dpo-fp16/)|[dpo-fp16](https://wisemodel.cn/models/OpenBMB/MiniCPM-2B-dpo-fp16)
  |[dpo-fp32](https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp32)|[dpo-fp32](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-dpo-fp32)|[dpo-fp32](https://wisemodel.cn/models/OpenBMB/miniCPM-dpo-fp32)

## 模型使用 Usage

* 安装`transformers>=4.36.0`以及`accelerate`后，运行以下代码
* 注意：需要在`from_pretrained`中明确指明模型的数据类型，否则会引起较大计算误差
* Run the following code after install `transformers>=4.36.0` and `accelerate`
* Warning: It is necessary to specify the data type of the model clearly in 'from_pretrained', otherwise large calculation errors will be caused 
```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

path = 'OpenBMB/MiniCPM-2B-dpo-bf16'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)

responds, history = model.chat(tokenizer, "山东省最高的山是哪座山, 它比黄山高还是矮？差距多少？", temperature=0.8, top_p=0.8)
print(responds)
```

* 期望输出 Expected Output
```shell
山东省最高的山是泰山，海拔1545米。

相对于黄山（海拔1864米），泰山海拔较低，相差约319米。
```

## 开源协议 LICENSE

#### 模型协议 Model LICENSE

* 本仓库中代码依照 [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) 协议开源
* MiniCPM 模型权重的使用则需要遵循 [“通用模型许可协议-来源说明-宣传限制-商业授权”](https://github.com/OpenBMB/General-Model-License/blob/main/%E9%80%9A%E7%94%A8%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE-%E6%9D%A5%E6%BA%90%E8%AF%B4%E6%98%8E-%E5%AE%A3%E4%BC%A0%E9%99%90%E5%88%B6-%E5%95%86%E4%B8%9A%E6%8E%88%E6%9D%83.md)。
* MiniCPM 模型权重对学术研究完全开放。
* 如需将模型用于商业用途，请联系cpm@modelbest.cn来获取书面授权，在登记后亦允许免费商业使用。

* This repository is released under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) License. 
* The usage of MiniCPM model weights must strictly follow [the General Model License (GML)](https://github.com/OpenBMB/General-Model-License/blob/main/%E9%80%9A%E7%94%A8%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE-%E6%9D%A5%E6%BA%90%E8%AF%B4%E6%98%8E-%E5%AE%A3%E4%BC%A0%E9%99%90%E5%88%B6-%E5%95%86%E4%B8%9A%E6%8E%88%E6%9D%83.md).
* The models and weights of MiniCPM are completely free for academic research.
* If you intend to utilize the model for commercial purposes, please reach out to cpm@modelbest.cn to obtain the certificate of authorization.

#### 声明 Statement

* 作为一个语言模型，MiniCPM 通过学习大量的文本来生成内容，但它无法理解、表达个人观点或价值判断，它所输出的任何内容都不代表模型开发者的观点和立场。
* 因此用户在使用 MiniCPM 生成的内容时，应自行负责对其进行评估和验证。
* 如果由于使用 MinCPM 开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。

* As a language model, MiniCPM generates content by learning from a vast amount of text. 
* However, it does not possess the ability to comprehend or express personal opinions or value judgments. 
* Any content generated by MiniCPM does not represent the viewpoints or positions of the model developers. 
* Therefore, when using content generated by MiniCPM, users should take full responsibility for evaluating and verifying it on their own.

<p id="8"></p>

## 工作引用 Citation

* 如果觉得MiniCPM有助于您的工作，请考虑引用下列[技术报告](https://shengdinghu.notion.site/MiniCPM-c805a17c5c8046398914e47f0542095a?pvs=4)
* Please cite our [techinical report](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20?pvs=4) if you find our work valuable.

```
@inproceedings{minicpm2024,
 title={MiniCPM：Unveiling the Potential of End-side Large Language Models},
 booktitle={OpenBMB Blog},
 year={2024}
}
```
