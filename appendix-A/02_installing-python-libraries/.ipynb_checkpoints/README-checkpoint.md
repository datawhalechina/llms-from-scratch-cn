# 本书使用的库

本文档提供有关检查已安装的 Python 版本和包的更多信息。（有关安装 Python 和 Python 包的更多信息，请参见 [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) 文件夹。）

我们在本书中使用了以下主要库。较新版本的这些库可能也是兼容的。但是，如果您在代码中遇到任何问题，可以尝试使用以下库版本作为备用：

-  numpy  1.24.3
-  scipy 1.10.1
-  pandas  2.0.2
-  matplotlib  3.7.1
-  jupyterlab  4.0
-  watermark  2.4.2
-  torch  2.0.1
-  tiktoken  0.5.1

要最方便地安装这些依赖，您可以使用 `requirements.txt` 文件：

```
pip install -r requirements.txt
```

然后，在完成安装后，请使用以下命令检查所有包是否已安装并且是否为最新版本：

```
python_environment_check.py
```

<img src="figures/check_1.jpg" width="600px">

还建议在 JupyterLab 中检查版本，方法是在此目录中运行 `jupyter_environment_check.ipynb`，这应该理想地给您与上面相同的结果。

<img src="figures/check_2.jpg" width="500px">

如果您看到以下问题，则可能您的 JupyterLab 实例连接到错误的 conda 环境：

<img src="figures/jupyter-issues.jpg" width="450px">


在这种情况下，您可以使用 `watermark` 来检查是否使用 `--conda` 标志在正确的 conda 环境中打开了 JupyterLab 实例：

<img src="figures/watermark.jpg" width="350px">


<br>
<br>


## 安装 PyTorch

PyTorch 可以像安装其他任何 Python 库或包一样使用 pip 安装。例如：

```bash
pip install torch==2.0.1
```

但是，由于 PyTorch 是一个全面的库，具有 CPU 和 GPU 兼容的代码，安装可能需要额外的设置和说明（有关更多信息，请参见书中的 *A.1.3 安装 PyTorch*）。

同时强烈建议在官方 PyTorch 网站的安装指南菜单中查看更多信息 [https://pytorch.org](https://pytorch.org)。

<img src="figures/pytorch-installer.jpg" width="600px">



---




有任何问题，请随时在 [Discussion Forum](https://github.com/rasbt/LLMs-from-scratch/discussions) 中联系我们。