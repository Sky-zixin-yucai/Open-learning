"""
OpenLearning RGA 命令行接口
===========================

通过以下方式使用：
    openlearning          # 运行完整演示
    openlearning --demo   # 运行演示
    openlearning --test   # 运行测试
"""

import sys

def main():
    """CLI入口函数"""
    # 直接从 __main__ 导入主函数
    from openlearning.__main__ import main as app_main
    return app_main()

if __name__ == "__main__":
    sys.exit(main())