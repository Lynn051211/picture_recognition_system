"""
图像识别系统入口：初始化识别引擎并启动 GUI 主窗口。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ui import MainWindow


def main():
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()
