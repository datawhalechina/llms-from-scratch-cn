# 从零开始构建大型语言模型

本存储库包含了编码、预训练和微调类似于GPT的LLM的代码，并且是书籍[从零开始构建大型语言模型](http://mng.bz/orYv)的官方代码存储库。

（如果您从Manning网站下载了代码包，请考虑访问GitHub上的官方代码存储库：[https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)。）

<br>
<br>

<a href="http://mng.bz/orYv"><img src="images/cover.jpg" width="250px"></a>

在[*从零开始构建大型语言模型*](http://mng.bz/orYv)中，您将从内部了解LLMs的工作原理。在本书中，我将逐步指导您创建自己的LLM，用清晰的文本、图表和示例解释每个阶段。

本书描述的用于培训和开发您自己的用于教育目的的小型但功能齐全模型的方法，与创建ChatGPT等大规模基础模型所使用的方法相似。

- 官方[源代码存储库链接](https://github.com/rasbt/LLMs-from-scratch)
- [Manning早期访问版本链接](http://mng.bz/orYv)
- ISBN 9781633437166
- 预计于2025年初出版

<br>
<br>


# 目录

请注意，`Readme.md`文件是一个Markdown（.md）文件。如果您从Manning网站下载了此代码包，并在本地计算机上查看它，我建议使用Markdown编辑器或预览器进行正确的查看。如果您尚未安装Markdown编辑器，[MarkText](https://www.marktext.cc)是一个不错的免费选择。

或者，您可以在GitHub上查看本文和其他文件：[https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)。

<br>
<br>

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

（* 如果您需要关于安装Python和Python包的更多指导，请参阅[此](appendix-A/01_optional-python-setup-preferences)和[此](appendix-A/02_installing-python-libraries)文件夹。）



<br>
<br>

<img src="images/mental-model.jpg" width="600px">

（这是一个总结了本书内容的思维模型。）