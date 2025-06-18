#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
系统初始化脚本
用于首次设置和训练模型
"""

import logging
import os

# 警告修复：在导入numpy/sklearn之前设置OMP_NUM_THREADS，以避免MKL在Windows上的内存泄漏警告
if os.name == 'nt' and 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '2'

import numpy as np
import pandas as pd

from compatibility_model import CompatibilityModel
from data_preprocessing import DataPreprocessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("init_system.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def initialize_system():
    """初始化系统"""
    logger.info("=== 开始系统初始化 ===")

    # 检查数据文件
    if not os.path.exists("Data.xlsx"):
        logger.error("找不到Data.xlsx文件，请确保文件存在")
        return False

    try:
        # 1. 加载和预处理数据
        logger.info("步骤1: 加载和预处理数据")
        df = pd.read_excel("Data.xlsx")
        logger.info(f"加载了 {len(df)} 名学生的数据")
        
        preprocessor = DataPreprocessor()
        preprocessor.fit(df)
        processed_df = preprocessor.transform(df)
        logger.info(f"数据预处理完成，特征数量: {processed_df.shape[1]}")

        # 2. 训练兼容性模型
        logger.info("步骤2: 训练兼容性模型")
        model = CompatibilityModel()
        model.train_model(df, processed_df, preprocessor, n_samples=1000000)  # 使用优化后的样本数量

        # 3. 保存模型和预处理器
        logger.info("步骤3: 保存模型和预处理器")
        model.save_model("compatibility_model")
        logger.info("模型已保存至 compatibility_model_xgb.json 和 compatibility_model_meta.json")

        # 4. 验证模型
        logger.info("步骤4: 验证模型")
        # 加载刚刚保存的模型进行验证
        loaded_model = CompatibilityModel.load_model("compatibility_model")
        # 选取前几个学生进行测试
        test_student_ids = df.index[:4].tolist()
        predicted_scores = loaded_model.predict_from_ids(test_student_ids, df, preprocessor)
        logger.info(f"为学生 {test_student_ids} 预测的兼容性得分: {predicted_scores}")

        logger.info("✅ 系统初始化成功完成！")
        return True

    except FileNotFoundError as e:
        logger.error(f"系统初始化失败：找不到数据文件。错误：{e}")
        return False


if __name__ == "__main__":
    success = initialize_system()
    if success:
        print("\n✅ 系统初始化成功！")
        print("现在可以运行 dashboard_app.py 启动Web界面")
    else:
        print("\n❌ 系统初始化失败，请检查错误信息")
