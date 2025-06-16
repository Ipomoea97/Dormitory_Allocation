"""
系统测试脚本
用于验证各个模块的基本功能
"""

import os
import sys


def test_imports():
    """测试模块导入"""
    print("=" * 50)
    print("测试模块导入...")

    try:
        import pandas as pd

        print("✓ pandas导入成功")
    except ImportError:
        print("✗ pandas导入失败")
        return False

    try:
        import numpy as np

        print("✓ numpy导入成功")
    except ImportError:
        print("✗ numpy导入失败")
        return False

    try:
        import xgboost as xgb

        print("✓ xgboost导入成功")
    except ImportError:
        print("✗ xgboost导入失败")
        return False

    return True


def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 50)
    print("测试数据加载...")

    try:
        import pandas as pd

        # 检查数据文件是否存在
        if not os.path.exists("Data.xlsx"):
            print("✗ 数据文件 Data.xlsx 不存在")
            return False

        # 加载数据
        df = pd.read_excel("Data.xlsx")
        print(f"✓ 数据加载成功，共 {len(df)} 行")
        print(f"✓ 数据列: {list(df.columns)}")

        # 显示前几行样本
        print("\n前3行数据样本:")
        print(df.head(3))

        return True

    except Exception as e:
        print(f"✗ 数据加载失败: {str(e)}")
        return False


def test_model_file():
    """测试模型文件"""
    print("\n" + "=" * 50)
    print("测试模型文件...")

    model_path = "compatibility_model_xgb.json"
    meta_path = "compatibility_model_meta.json"

    model_exists = os.path.exists(model_path)
    meta_exists = os.path.exists(meta_path)

    if model_exists:
        print(f"✓ XGBoost模型文件 ({model_path}) 存在")
    else:
        print(f"✗ XGBoost模型文件 ({model_path}) 不存在")

    if meta_exists:
        print(f"✓ 模型元数据文件 ({meta_path}) 存在")
    else:
        print(f"✗ 模型元数据文件 ({meta_path}) 不存在")

    if model_exists and meta_exists:
        return True
    else:
        print("  => 请先运行 'python init_system.py' 来训练和生成模型文件。")
        return False


def test_modules():
    """测试自定义模块"""
    print("\n" + "=" * 50)
    print("测试自定义模块...")

    modules = [
        "data_preprocessing",
        "compatibility_model",
        "allocation_optimizer",
        "explanation_module",
    ]

    success_count = 0
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name} 模块导入成功")
            success_count += 1
        except Exception as e:
            print(f"✗ {module_name} 模块导入失败: {str(e)}")

    return success_count == len(modules)


def main():
    """主测试函数"""
    print("宿舍分配系统 - 基础测试")
    print("=" * 50)

    # 显示Python和工作目录信息
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"文件列表: {os.listdir('.')}")

    # 运行测试
    tests = [
        ("模块导入", test_imports),
        ("数据加载", test_data_loading),
        ("模型文件", test_model_file),
        ("自定义模块", test_modules),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {str(e)}")

    # 总结
    print("\n" + "=" * 50)
    print(f"测试总结: {passed}/{total} 项测试通过")

    if passed == total:
        print("🎉 所有测试通过！系统准备就绪")
        return True
    else:
        print("⚠️  部分测试失败，请检查环境配置")
        return False


if __name__ == "__main__":
    main()
