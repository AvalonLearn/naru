# Naru_v1.0
Naru模型的代码实现

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/AvalonLearn/naru)
# 使用方法
启动激活环境
1. conda
    ```shell
    conda env create -f .\environment.yaml
    conda activate naru
    ```
2. pip
    ```shell
    pip install -r requirements.txt
    ```
运行模型
1. 正常运行
    ```shell
    python naru_demo.py 
    # 略去不必要的输出细节
    ```
2. 展示详细输出
    ```shell
    python naru_demo.py -d
    # 详细输出在output.txt中
    # 终端仅有主要输出
    ```
3. 清除model文件和中间文件
    
    **Shell**
    ```bash
    ./clear.sh
    ```
    **Powershell or cmd**
    ```powershell
    .\clear.bat
    ```