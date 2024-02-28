# Python 设置提示



有几种不同的方法可以安装 Python 并设置您的计算环境。在这里，我将介绍我的个人偏好。

（我使用运行 macOS 的计算机，但此工作流程对于运行 Linux 的计算机是类似的，并且可能也适用于其他操作系统。）

<br>
<br>

## 1. 下载并安装 Miniforge

从 GitHub 仓库 [这里](https://github.com/conda-forge/miniforge) 下载 miniforge。

<img src="figures/download.png" alt="download" width="600px">

根据您的操作系统，这应该会下载一个 `.sh`（macOS，Linux）或 `.exe` 文件（Windows）。

对于 `.sh` 文件，请打开您的命令行终端并执行以下命令

```bash
sh ~/Desktop/Miniforge3-MacOSX-arm64.sh
```

其中 `Desktop/` 是 Miniforge 安装程序下载到的文件夹。在您的计算机上，您可能需要用 `Downloads/` 替换它。

<img src="figures/miniforge-install.png" alt="miniforge-install" width="600px">

接下来，按照下载说明步骤进行操作，并按下 "Enter" 确认。

如果您使用许多包，Conda 可能会因为其彻底但复杂的依赖解析过程以及处理大型包索引和元数据而变慢。为了加快 Conda 的速度，您可以使用以下设置，它将切换到更有效的 Rust 重新实现以解决依赖关系：

```
conda config --set solver libmamba
```

<br>
<br>

## 2. 创建一个新的虚拟环境

安装成功后，我建议创建一个名为 `dl-fundamentals` 的新虚拟环境，您可以通过执行以下命令来完成

```bash
conda create -n LLMs python=3.10
```

<img src="figures/new-env.png" alt="new-env" width="600px">

> 许多科学计算库不会立即支持最新版本的 Python。因此，在安装 PyTorch 时，建议使用较旧的 Python 版本，即一两个版本。例如，如果最新版本的 Python 是 3.13，则建议使用 Python 3.10 或 3.11。

接下来，激活您的新虚拟环境（每次打开新的终端窗口或选项卡时都必须执行）：

```bash
conda activate dl-workshop
```

<img src="figures/activate-env.png" alt="activate-env" width="600px">

<br>
<br>

## 可选: 美化您的终端

如果您想将终端样式设置为与我的类似，以便您可以看到哪个虚拟环境是活动的，请查看 [Oh My Zsh](https://github.com/ohmyzsh/ohmyzsh) 项目。

<br>
<br>

## 3. 安装新的 Python 库



要安装新的 Python 库，您现在可以使用 `conda` 包安装程序。例如，您可以安装 [JupyterLab](https://jupyter.org/install) 和 [watermark](https://github.com/rasbt/watermark) 如下：

```bash
conda install jupyterlab watermark
```

<img src="figures/conda-install.png" alt="conda-install" width="600px">

您也仍然可以使用 `pip` 安装库。默认情况下，`pip` 应该已链接到您的新的 `LLms` conda 环境：

<img src="figures/check-pip.png" alt="check-pip" width="600px">

<br>
<br>

## 4. 安装 PyTorch

PyTorch 可以像安装其他任何 Python 库或包一样使用 pip 安装。例如：

```bash
pip install torch==2.0.1
```

但是，由于 PyTorch 是一个全面的库，具有 CPU 和 GPU 兼容的代码，安装可能需要额外的设置和说明（有关更多信息，请参见书中的 *A.1.3 安装 PyTorch*）。

还强烈建议在官方 PyTorch 网站的安装指南菜单中查看更多信息 [https://pytorch.org](https://pytorch.org)。

<img src="figures/pytorch-installer.jpg" width="600px">



---




有任何问题吗？请随时在 [Discussion Forum](https://github.com/rasbt/LLMs-from-scratch/discussions) 中联系我们。