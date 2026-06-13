# -*- coding: utf-8 -*-
"""
水库调度命令行工具主入口
支持通过 python -m reservoir_scheduling 调用
"""

import sys
from .cli.main import main

if __name__ == "__main__":
    sys.exit(main())
