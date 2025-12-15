## 目标
- 彻底消除 “OMP Error #15: libiomp5md.dll 已被初始化” 的冲突，确保训练脚本稳定运行并维持当前 W&B 记录。

## 背景与原则
- 该错误由多个 OpenMP 运行时同时被加载导致（Intel 官方建议仅保留单一运行时来源）。
- 优先采用环境清理与统一依赖方式；仅在必须时使用临时绕过变量。

## 诊断步骤
- 列出当前 PATH 中可能包含 `libiomp5md.dll` 的目录，并定位重复来源：
  - `where libiomp5md.dll`
- 在指定 Python 环境中快速自检：
  - `"D:/anaconda/Anaconda/envs/ljj_lunwen/python.exe" -c "import numpy, scipy, numexpr, torch; print('ok')"`

## 修复方案A（推荐：单一 Intel OpenMP 来源）
- 统一由 Conda 环境提供 OpenMP/MKL，清理系统级或其他环境的干扰：
  - 安装/修复依赖：
    - `conda install -n ljj_lunwen -y intel-openmp mkl numpy scipy numexpr`
  - 清理 PATH（保留 `.../envs/ljj_lunwen/Library/bin` 中的 `libiomp5md.dll`，移除系统级 Intel 目录如 `C:\Program Files (x86)\Intel...` 等）。
- 验证训练初始化与无报错后继续训练。

## 修复方案B（备选：避开 Intel OpenMP）
- 切换 BLAS 到 OpenBLAS，避免加载 Intel OpenMP：
  - `conda remove -n ljj_lunwen -y mkl intel-openmp`
  - `conda install -n ljj_lunwen -c conda-forge -y nomkl blas=*=openblas numpy scipy numexpr`
- 再次验证 `numpy/scipy/torch` 的导入与训练执行。

## 代码级临时绕过（仅短期应急）
- 在训练入口（`onpolicy/scripts/train/train_my.py:98-117, 164-168` 附近的初始化前）加入：
  - `import os`
  - `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"`
  - `os.environ.setdefault("OMP_NUM_THREADS", "1")`
  - `os.environ.setdefault("MKL_NUM_THREADS", "1")`
- 风险：该方法可能掩盖潜在问题（官方提示不推荐），建议在完成环境清理后移除。

## 验证
- 快速运行：`"D:/anaconda/Anaconda/envs/ljj_lunwen/python.exe" -m onpolicy.scripts.train.train_my --use_wandb --env_name MEC`
- 观察不再出现 OMP 错误，W&B 正常记录。

## 执行安排
- 我将先执行诊断与A方案；若仍有冲突，再尝试B方案；必要时临时加入代码级绕过，并在清理完成后撤除。
- 完成后提供变更说明与验证结果。