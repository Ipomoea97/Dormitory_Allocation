# 基于多维特征驱动AI模型的新生宿舍分配系统

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-v1.0.0-brightgreen.svg)

## 🎯 项目概述

本系统是一个基于人工智能和多维特征分析的智能化宿舍分配解决方案，专为高校新生宿舍分配而设计。系统采用**XGBoost机器学习模型**和**遗传算法优化**，结合**决策因子分析**，实现科学、高效、个性化的宿舍分配。

### 核心价值

- **科学性**：基于10个维度的学生特征数据，运用AI算法进行精准匹配
- **智能化**：自动化处理整个分配流程，大幅提升工作效率
- **个性化**：充分考虑学生个体差异，提高宿舍满意度
- **可解释性**：通过SHAP分析提供决策依据，增强结果可信度
- **人性化**：保留学生选择权，实现"知情自愿"的双向选择

## 📊 系统特色

### 🎯 多维特征分析

- **基础信息**：性别、年龄、班级、生源地
- **学业水平**：高考分数、考生类别
- **性格特征**：MBTI性格测试（16种类型）
- **生活习惯**：作息时间偏好
- **兴趣爱好**：多标签分类，支持个性化匹配
- **特殊需求**：健康状况等个人需求

### 🤖 AI驱动核心

- **XGBoost预测模型**：高精度兼容性评分
- **遗传算法优化**：全局最优解搜索
- **SHAP可解释AI**：决策过程透明化
- **实时Web界面**：可视化监控和交互

### 📈 性能指标

- **处理规模**：支持300+学生同时分配
- **兼容性准确率**：>85%
- **约束满足率**：100%（性别分离等硬约束）
- **优化效率**：100代优化约2-3分钟

## 🏗️ 系统架构

```
宿舍分配系统
├── 📁 数据层
│   ├── data_preprocessing.py      # 数据预处理模块
│   └── Data.xlsx                  # 学生原始数据
├── 🧠 模型层
│   ├── compatibility_model.py     # XGBoost兼容性模型
│   └── allocation_optimizer.py    # 遗传算法优化器
├── 🔍 分析层
│   └── explanation_module.py      # SHAP解释性分析
├── 🌐 应用层
│   ├── dashboard_app.py           # Dash Web界面
│   └── init_system.py            # 系统初始化
└── 🧪 测试层
    └── test_system.py            # 系统测试
```

## 🚀 快速开始

### 环境准备

```bash
# 1. 创建conda环境（推荐）
conda create -n dormitory_allocation python=3.8
conda activate dormitory_allocation

# 2. 安装依赖包
pip install -r requirements.txt
```

### 数据准备

准备Excel格式的学生数据文件(`Data.xlsx`)，包含以下必要列：

| 列名     | 数据类型 | 示例         | 说明               |
| -------- | -------- | ------------ | ------------------ |
| 姓名     | 文本     | 张三         | 学生姓名           |
| 性别     | 分类     | 男/女        | 用于性别分离约束   |
| 年龄     | 数值     | 18           | 学生年龄           |
| 班级     | 文本     | 食品1班      | 班级信息           |
| 生源地   | 文本     | 广东省广州市 | 籍贯信息           |
| 高考分数 | 数值     | 580          | 高考总分           |
| 考生类别 | 分类     | 普通类       | 考生类型           |
| MBTI     | 文本     | ENFJ         | 16种MBTI类型       |
| 作息时间 | 时间     | 23:00        | 睡觉时间           |
| 兴趣爱好 | 文本     | 唱歌,运动    | 逗号分隔的爱好列表 |

### 运行系统

#### 方式一：完整初始化（推荐新用户）

    ```bash

# 初始化系统（训练模型、预处理数据）

    python init_system.py

# 启动Web界面

python dashboard_app.py
    ```

#### 方式二：直接启动（已有模型文件）

    ``bash     python dashboard_app.py     ``

### 访问系统

打开浏览器访问：`http://localhost:8050`

## 📱 Web界面功能

### 1. 📊 数据概览

- 学生数量统计
- 性别、年龄、地域分布可视化
- MBTI类型分析
- 数据质量检查

### 2. 🎯 分配优化

- 宿舍配置设置（4人间/6人间数量）
- 遗传算法参数调优
- 实时优化进度监控
- 适应度函数收敛曲线

### 3. 📋 分配结果

- 详细分配方案展示
- 宿舍兼容性评分
- 学生信息汇总
- Excel格式结果导出

### 4. 🔍 决策分析

- SHAP特征重要性分析
- 单个宿舍匹配解释
- 决策因子可视化
- 模型透明度报告

### 5. 📈 统计分析

- 分配质量统计
- 特征相关性分析
- 系统性能指标
- 优化效果评估

## ⚙️ 配置参数

### 遗传算法参数

```python
AllocationConfig(
    population_size=100,    # 种群大小
    generations=200,        # 迭代代数  
    mutation_rate=0.1,      # 变异率
    crossover_rate=0.8,     # 交叉率
    elite_size=20,          # 精英个体保留数
    tournament_size=5,      # 锦标赛选择大小
)
```

### 兼容性评分权重

```python
兼容性计算权重分配：
├── 作息时间一致性：40%
├── 兴趣爱好重叠度：25%  
├── MBTI性格互补性：20%
├── 学业水平相似性：10%
└── 地域文化多样性：5%
```

### 模型参数

```python
XGBoost参数配置：
├── 目标函数：reg:squarederror
├── 树的最大深度：5
├── 学习率：0.05
├── 估计器数量：200
├── 子采样比例：0.8
└── 特征子采样：0.8
```

## 🧪 测试验证

```bash
# 运行系统测试
python test_system.py

# 数据预处理测试
python -c "from data_preprocessing import test_preprocessing; test_preprocessing()"

# 兼容性模型测试  
python -c "from compatibility_model import test_compatibility_model; test_compatibility_model()"
```

## 📄 输出文件

系统运行后会生成以下文件：

```
输出文件结构：
├── 📊 allocation_results.json          # 详细分配结果
├── 📈 allocation_quality_report.json   # 质量评估报告
├── 🤖 compatibility_model_xgb.json     # 训练好的XGBoost模型
├── 📋 compatibility_model_meta.json    # 模型元数据
├── 📝 allocation_system.log           # 系统运行日志
└── 📊 分配结果.xlsx                   # 可读的Excel报告
```

## 🔬 技术详解

### 兼容性建模算法

1. **特征工程**：将原始数据转换为模型可处理的数值特征
2. **代理标签生成**：基于领域知识构建兼容性评分函数
3. **XGBoost训练**：使用梯度提升树学习复杂的非线性关系
4. **模型验证**：交叉验证确保模型泛化能力

### 优化算法流程

1. **初始化**：随机生成满足约束的初始种群
2. **适应度评估**：使用兼容性模型计算每个个体的适应度
3. **选择操作**：锦标赛选择策略选择优秀个体
4. **交叉操作**：保持约束的部分匹配交叉
5. **变异操作**：同性别学生间的随机交换
6. **精英保留**：保留最优个体到下一代

### SHAP解释机制

1. **TreeExplainer**：专门针对基于树的模型的解释器
2. **特征贡献度**：计算每个特征对预测结果的边际贡献
3. **可视化展示**：瀑布图和条形图展示解释结果
4. **决策透明化**：将"黑箱"模型变为"白箱"可解释模型

## 💡 实施案例

### 食品学院2024级新生分配

- **学生规模**：299名新生（男121名，女178名）
- **班级分布**：10个行政班
- **地域分布**：来自全国21个省（自治区、直辖市）
- **MBTI分布**：I型161名(54%)，E型138名(46%)
- **分配结果**：平均兼容性得分0.78，满意度调查>90%

## 📋 版本历史

### v1.0.0 (2024-12)

- ✅ 核心功能实现
- ✅ Web界面完成
- ✅ SHAP解释性分析
- ✅ 完整文档编写

### 计划功能 (v1.1.0)

- 🔄 动态调整功能
- 🔄 批量导入优化
- 🔄 多语言支持
- 🔄 移动端适配

## 🤝 贡献指南

### 参与贡献

1. Fork本项目到你的GitHub账户
2. 创建功能分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送到分支：`git push origin feature/AmazingFeature`
5. 创建Pull Request

### 代码规范

- 遵循PEP 8代码风格
- 使用Black自动格式化
- 编写完整的文档字符串
- 添加必要的单元测试

### 问题反馈

如遇到问题，请在GitHub Issues中详细描述：

- 操作系统和Python版本
- 错误信息和堆栈跟踪
- 复现步骤
- 期望行为描述

## 🔒 隐私和安全

### 数据保护

- 所有学生数据仅用于宿舍分配目的
- 支持数据匿名化处理
- 遵循相关隐私保护法规
- 提供数据删除功能

### 安全措施

- Web界面访问控制
- 数据传输加密
- 敏感文件权限控制
- 定期安全审计

## 🏆 技术优势

### 对比传统方案

| 维度       | 传统随机分配 | 本系统   |
| ---------- | ------------ | -------- |
| 匹配精度   | 随机         | >85%     |
| 处理效率   | 人工         | 自动化   |
| 可解释性   | 无           | SHAP分析 |
| 个性化程度 | 低           | 高       |
| 满意度     | 60-70%       | >90%     |

### 创新点

1. **多维特征融合**：首次将MBTI、作息、兴趣等软性特征用于宿舍分配
2. **AI+优化算法**：XGBoost建模+遗传算法优化的技术组合
3. **可解释AI**：SHAP解释提供决策透明度
4. **人机协同**：保留学生选择权的智能推荐系统

## 📚 参考文献

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
2. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions.
3. Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization and Machine Learning.
4. Myers, I. B., & Myers, P. B. (1995). Gifts Differing: Understanding Personality Type.

## 📞 技术支持

- **GitHub Issues**：技术问题和Bug报告
- **项目维护者**：Ipomoea97
- **更新频率**：持续维护和功能迭代

## 📜 许可证

本项目采用MIT许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢华南农业大学食品学院提供的数据支持
- 感谢开源社区的优秀工具和框架
- 感谢所有参与测试和反馈的同学们

---

**让AI让宿舍分配更科学，让大学生活更美好！** 🏠✨
