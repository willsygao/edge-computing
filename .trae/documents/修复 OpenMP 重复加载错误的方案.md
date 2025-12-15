## 原因
- 出现 `OMP: Error #15 ... libiomp5md.dll already initialized`，说明同一进程中加载了多个 OpenMP 运行时（常见来源：Anaconda 的 MKL/Intel OpenMP 与通过 pip 安装的 PyTorch/NumPy/Scipy 的 OpenMP 版本混用，或系统 PATH 中存在额外的 `libiomp5md.dll`）。重复加载会导致性能下降或结果不正确。

## 快速绕过（仅调试）
- 临时设置环境变量跳过重复检测（不推荐长期使用）：
  - PowerShell：`$env:KMP_DUPLICATE_LIB_OK="TRUE"` 后再运行训练
  - 永久：`setx KMP_DUPLICATE_LIB_OK TRUE`
  - 代码内（在任何科学库导入之前）：
    ```python
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    ```

## 推荐修复
1. 统一安装源，避免混用 OpenMP 运行时
- 在 `ljj_lunwen` 环境中，改用 conda 安装 PyTorch 与依赖，确保与 MKL/Intel OpenMP 一致：
  - CPU：`conda install pytorch torchvision torchaudio cpuonly -c pytorch`
  - 如果之前用 pip 安装过 torch：`pip uninstall -y torch torchvision torchaudio`
- 或者走 `conda-forge` 的 `nomkl` 路线，用 OpenBLAS 替代 MKL（避免 Intel OpenMP）：
  - `conda install -c conda-forge nomkl numpy scipy`

2. 清理 PATH 中的重复 `libiomp5md.dll`
- 检查当前位置加载的 DLL：`where libiomp5md.dll`
- 如果显示多个路径（如 Intel oneAPI 安装目录与 Anaconda 的 `Library\bin`），将系统 PATH 调整为仅保留你所用环境的副本，或临时运行前移除其它路径。

3. 代码层面防护（可选）
- 在训练入口文件最顶部（所有科学库导入之前），设置线程与禁用 MKL 冲突的最小化风险：
  ```python
  import os
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  os.environ.setdefault("MKL_NUM_THREADS", "1")
  ```
- 若仍报错，可在最顶部加上 `KMP_DUPLICATE_LIB_OK` 作为兜底（仅开发环境使用）。

## 验证
- 执行 `where libiomp5md.dll` 的结果应只指向一个来源（优先是当前 Anaconda 环境）。
- 使用 `python -m onpolicy.scripts.train.train_my` 正常启动且无 OMP #15 报错。

## 建议优先顺序
- 首选：在 `ljj_lunwen` 环境用 conda 统一安装 PyTorch 与数值库，避免 pip/conda 混装；随后确认 PATH 中只有一个 `libiomp5md.dll`。
- 次选：代码最顶部临时设置 `KMP_DUPLICATE_LIB_OK`，作为开发调试兜底。