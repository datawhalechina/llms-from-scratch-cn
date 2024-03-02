<center>
# 动手实现LLM中文版
  ---
# Implementing LLM Hands-on: Building Your Own Large Language Model from Scratch

</center>


[![GitHub stars](https://img.shields.io/github/stars/username/repo.svg?style=social)](https://github.com/datawhalechina/llms-from-scratch-cn)
[![GitHub forks](https://img.shields.io/github/forks/username/repo.svg?style=social)](https://github.com/datawhalechina/llms-from-scratch-cn)
[![GitHub issues](https://img.shields.io/github/issues/username/repo.svg)](https://github.com/datawhalechina/llms-from-scratch-cn/issues)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-brightgreen.svg)](https://github.com/datawhalechina/llms-from-scratch-cn/blob/main/LICENSE.txt)


GitHub上的"rasbt/LLMs-from-scratch"项目是一个关于如何从头开始实现类似ChatGPT的大语言模型（LLM）的教程。这个项目包含了编码、预训练和微调GPT-like LLM的代码，并且是《Build a Large Language Model (From Scratch)》这本书的官方代码库。书中详细介绍了LLM的内部工作原理，并逐步指导读者创建自己的LLM，包括每个阶段的清晰文本、图表和示例。这种方法用于训练和开发自己的小型但功能性的模型，用于教育目的，与创建大型基础模型（如ChatGPT背后的模型）的方法相似，翻译后的版本可以服务于国内的开发者。

| 章节标题                                        | 主要代码（快速访问）                                                                                                           | 所有代码 + 补充           |
|------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| 第1章: 理解大型语言模型                       | 没有代码                                                                                                                        | 没有代码                      |
| 第2章: 处理文本数据                            | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb) (摘要)<br/>- [exercise-solutions.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb) | [./ch02](./ch02)              |
| 第3章: 编写注意力机制                          | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb) (摘要) <br/>- [exercise-solutions.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb)| [./ch03](./ch03)              |
| 第4章: 从零开始实现GPT模型                     | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py) (摘要)<br/>- [exercise-solutions.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb) | [./ch04](./ch04)           |
| 第5章: 使用未标记数据进行预训练               | 2024年第1季度                                                                                                                  | ...                           |
| 第6章: 用于文本分类的微调                     | 2024年第2季度                                                                                                                  | ...                           |
| 第7章: 使用人类反馈进行微调                   | 2024年第2季度                                                                                                                  | ...                           |
| 第8章: 在实践中使用大型语言模型               | 2024年第2/3季度                                                                                                                | ...                           |
| 附录A: PyTorch简介*                            | - [code-part1.ipynb](appendix-A/03_main-chapter-code/code-part1.ipynb)<br/>- [code-part2.ipynb](appendix-A/03_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/03_main-chapter-code/DDP-script.py)<br/>- [exercise-solutions.ipynb](appendix-A/03_main-chapter-code/exercise-solutions.ipynb) | [./appendix-A](./appendix-A) |

（* 如果您需要关于安装Python和Python包的更多指导，请参阅[附录1](appendix-A/01_optional-python-setup-preferences)和[附录2](appendix-A/02_installing-python-libraries)文件夹。）


<br>
<br>

<img src="images/mental-model.jpg" width="600px">

（这是一个总结了本书内容的思维导图。）

- 项目受众

  - 技术背景：该项目适合有一定编程基础的人员，特别是对大型语言模型（LLM）感兴趣的开发者和研究者。
  - 学习目标：适合那些希望深入了解LLM工作原理，并愿意投入时间从零开始构建和训练自己的LLM的学习者。
  - 应用领域：适用于对自然语言处理、人工智能领域感兴趣的开发者，以及希望在教育或研究环境中应用LLM的人员。
- 项目亮点

  - 系统化学习：该项目提供了一个系统化的学习路径，从理论基础到实际编码，帮助学习者全面理解LLM。
  - 实践导向：与仅仅介绍理论或API使用不同，该项目强调实践，让学习者通过实际操作来掌握LLM的开发和训练。
  - 深入浅出：该项目以清晰的语言、图表和示例来解释复杂的概念，使得非专业背景的学习者也能较好地理解。

## Roadmap

*注：说明当前项目的规划，并将每个任务通过 Issue 形式进行对外进行发布。*

## 参与贡献

- 如果你想参与到项目中来欢迎查看项目的 [Issue]() 查看没有被分配的任务。
- 如果你发现了一些问题，欢迎在 [Issue]() 中进行反馈🐛。
- 如果你对本项目感兴趣想要参与进来可以通过 [Discussion]() 进行交流💬。

如果你对 Datawhale 很感兴趣并想要发起一个新的项目，欢迎查看 [Datawhale 贡献指南](https://github.com/datawhalechina/DOPMC#%E4%B8%BA-datawhale-%E5%81%9A%E5%87%BA%E8%B4%A1%E7%8C%AE)。

## 贡献者名单

| 姓名   | 职责        | 简介         | GitHub |
| :-----:| :----------:| :-----------:|:------:|
| 陈可为 | 项目负责人  | 华中科技大学 |[@Ethan-Chen-plus](https://github.com/Ethan-Chen-plus)|
| 王训志 | 第2章贡献者 | 南开大学     |[@aJupyter](https://github.com/aJupyter)|
| 汪健麟 | 第2章贡献者 |              ||
| 张友东 | 第3章贡献者 |              ||
| 邹雨衡 | 第3章贡献者 |              ||
| 陈嘉诺 | 第4章贡献者 |              ||
| 高立业 | 第4章贡献者 |              ||
| 周景林 | 附录贡献者  |              ||
| 陈可为 | 附录贡献者  |              ||


## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

*注：默认使用CC 4.0协议，也可根据自身项目情况选用其他协议*
