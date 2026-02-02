@echo off
setlocal

rem 设置 Python解释器的绝对路径 (绕过环境激活问题)
set "PYTHON_EXE=D:\anaconda\Anaconda\envs\ljj_lunwen\python.exe"
rem 设置脚本路径
set "SCRIPT_PATH=onpolicy\scripts\train\train_my.py"

@REM echo ==========================================================
@REM echo 开始 Task Size 实验 (Group 1/5: 600 KB)
@REM echo ==========================================================
@REM "%PYTHON_EXE%" %SCRIPT_PATH% --num_agents 10 --experiment_name task_200 --algorithm_name greedy --task_input_size 200

@REM echo ==========================================================
@REM echo 开始 Task Size 实验 (Group 2/5: 800 KB)
@REM echo ==========================================================
@REM "%PYTHON_EXE%" %SCRIPT_PATH% --num_agents 10 --experiment_name task_400 --algorithm_name greedy --task_input_size 400

@REM echo ==========================================================
@REM echo 开始 Task Size 实验 (Group 3/5: 1000 KB)
@REM echo ==========================================================
@REM "%PYTHON_EXE%" %SCRIPT_PATH% --num_agents 10 --experiment_name task_600 --algorithm_name greedy --task_input_size 600

@REM echo ==========================================================
@REM echo 开始 Task Size 实验 (Group 4/5: 1200 KB)
@REM echo ==========================================================
@REM echo "%PYTHON_EXE%" %SCRIPT_PATH% --num_agents 10 --experiment_name task_1200 --algorithm_name mappo --task_input_size 1200

@REM echo ==========================================================
@REM echo 开始 Task Size 实验 (Group 5/5: 1400 KB)
@REM echo ==========================================================
@REM "%PYTHON_EXE%" %SCRIPT_PATH% --num_agents 10 --experiment_name task_1000 --algorithm_name greedy --task_input_size 1000

@REM echo ==========================================================
@REM echo 开始 Server Frequency 实验 (Group 1/5: 10 GHz)
@REM echo ==========================================================
@REM "%PYTHON_EXE%" %SCRIPT_PATH% --num_agents 10 --experiment_name freq_2 --algorithm_name mappo --server_freq 4.0

@REM echo ==========================================================
@REM echo 开始 Server Frequency 实验 (Group 2/5: 15 GHz)
@REM echo ==========================================================
@REM "%PYTHON_EXE%" %SCRIPT_PATH% --num_agents 10 --experiment_name freq_4 --algorithm_name mappo --server_freq 8.0

echo ==========================================================
echo 开始 Server Frequency 实验 (Group 3/5: 20 GHz)
echo ==========================================================
"%PYTHON_EXE%" %SCRIPT_PATH% --num_agents 10 --experiment_name freq_6 --algorithm_name mappo --server_freq 12.0

echo ==========================================================
echo 开始 Server Frequency 实验 (Group 4/5: 25 GHz)
echo ==========================================================
"%PYTHON_EXE%" %SCRIPT_PATH% --num_agents 10 --experiment_name freq_8 --algorithm_name mappo --server_freq 16.0

@REM echo ==========================================================
@REM echo 开始 Server Frequency 实验 (Group 5/5: 30 GHz)
@REM echo ==========================================================
@REM "%PYTHON_EXE%" %SCRIPT_PATH% --num_agents 10 --experiment_name freq_10 --algorithm_name maddpg --server_freq 10.0

echo 所有实验完成！
pause
