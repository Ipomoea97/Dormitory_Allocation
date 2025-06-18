"""
SHAP解释模块
用于解释宿舍分配决策的可解释性分析
"""

import logging
import warnings
import random
from typing import Any, Dict, List, Optional, Tuple
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import shap
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from sklearn.metrics import silhouette_score

from compatibility_model import CompatibilityModel
from data_preprocessing import DataPreprocessor

# 设置中文字体和样式
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
# 建议避免使用全局的 'ignore'，因为它会抑制所有警告，可能隐藏重要问题。
# warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AllocationExplainer:
    """宿舍分配解释器"""

    def __init__(
        self,
        model: CompatibilityModel,
        preprocessor: DataPreprocessor,
        student_df: pd.DataFrame,
        allocation_df: pd.DataFrame,
    ):
        """
        初始化解释器

        Args:
            model: 已训练的兼容性模型
            preprocessor: 数据预处理器
            student_df: 原始学生数据
            allocation_df: 优化后的分配结果DataFrame
        """
        self.compatibility_model = model
        self.preprocessor = preprocessor
        self.student_df = student_df
        self.allocation_df = allocation_df

        # 预处理特征
        self.processed_features = self.preprocessor.transform(self.student_df)
        self.feature_names = self.preprocessor.get_feature_names()

        # 使用KMeans为SHAP创建背景数据，这是标准做法，比随机抽样更稳定
        background_data_summary_shap = shap.kmeans(self.processed_features, 10)
        # 将SHAP的DenseData对象转换为numpy数组
        background_data_summary = background_data_summary_shap.data

        # 组合背景数据特征，以匹配模型输入
        # 我们随机组合背景数据中的点来创建配对
        num_background_points = background_data_summary.shape[0]
        idx1 = np.random.randint(0, num_background_points, num_background_points)
        idx2 = np.random.randint(0, num_background_points, num_background_points)

        background_pairs = np.array([
            self.compatibility_model.create_pair_feature_vector(background_data_summary, i, j)
            for i, j in zip(idx1, idx2)
        ])

        self.explainer = shap.TreeExplainer(
            self.compatibility_model.model, background_pairs
        )

        logger.info(f"解释器初始化完成，特征数量: {len(self.feature_names)}")

    def get_allocation_metrics(self) -> Dict:
        """计算分配方案的多个质量指标"""
        metrics = {}
        room_scores = {} # 新增：用于存储每个宿舍的兼容度
        
        # 1. 平均兼容度
        all_pairs_compat = []
        for room_id, students_in_room in self.allocation_df.groupby("RoomID"):
            if len(students_in_room) < 2:
                room_scores[room_id] = np.nan # 少于2人无法计算兼容度
                continue
            
            # 获取真实的StudentID，但需要转换为DataFrame索引
            student_real_ids = students_in_room["StudentID"].tolist()
            # 将真实StudentID转换为DataFrame的数字索引
            student_indices = []
            for real_id in student_real_ids:
                # 在DataFrame中查找对应的索引位置
                index_pos = self.student_df[self.student_df["StudentID"] == real_id].index
                if not index_pos.empty:
                    student_indices.append(index_pos[0])
            
            # 从模型获取兼容性分数
            pair_scores = self.compatibility_model.predict_from_ids(
                student_indices, self.student_df, self.preprocessor
            )
            all_pairs_compat.extend(pair_scores)
            
            # 计算并存储该宿舍的平均分
            room_scores[room_id] = np.mean(pair_scores) if pair_scores else np.nan

        metrics["mean_compatibility"] = np.mean(all_pairs_compat) if all_pairs_compat else 0
        metrics["compatibilities"] = all_pairs_compat  # 添加所有兼容度分数列表

        # 2. MBTI 轮廓系数
        try:
            # 仅对优化宿舍进行计算
            optimized_df = self.allocation_df[self.allocation_df['RoomType'].str.contains('人间')]
            if len(optimized_df['RoomID'].unique()) > 1:
                # 获取真实StudentID并转换为DataFrame索引
                real_student_ids = optimized_df['StudentID'].tolist()
                # 根据真实StudentID找到对应的DataFrame行
                valid_indices = []
                for real_id in real_student_ids:
                    matching_rows = self.student_df[self.student_df["StudentID"] == real_id]
                    if not matching_rows.empty:
                        valid_indices.append(matching_rows.index[0])
                
                # 提取MBTI特征用于聚类评估
                if valid_indices:
                    mbti_features = self.preprocessor.transform(
                        self.student_df.iloc[valid_indices]
                    )
                    labels = optimized_df['RoomID'].values
                    metrics["mbti_silhouette"] = silhouette_score(mbti_features, labels)
                else:
                    metrics["mbti_silhouette"] = 0
            else:
                metrics["mbti_silhouette"] = 0
        except Exception as e:
            metrics["mbti_silhouette"] = f"计算失败: {e}"

        metrics["room_scores"] = room_scores # 将宿舍得分添加到结果中
        return metrics

    def get_feature_importance(self, num_pairs=100):
        """
        使用SHAP计算并返回全局特征重要性。
        通过对随机抽样的学生对进行分析来估算。
        """
        logger.info(f"开始计算 {num_pairs} 个学生对的SHAP特征重要性...")

        # 1. 准备背景数据和待解释数据
        # 我们需要从所有学生中随机抽样来创建配对
        all_student_ids = self.student_df.index.tolist()
        if len(all_student_ids) < 2:
            return pd.DataFrame() # 学生太少无法计算

        X_explain = []
        sample_indices = random.sample(all_student_ids, min(num_pairs * 2, len(all_student_ids)))
        
        for s1_id, s2_id in combinations(sample_indices, 2):
            if len(X_explain) >= num_pairs:
                break
            feature_vector = CompatibilityModel.create_pair_feature_vector(
                self.processed_features, s1_id, s2_id
            )
            X_explain.append(feature_vector)
        
        if not X_explain:
            logger.warning("未能创建任何用于解释的学生对。")
            return pd.DataFrame()

        X_explain = np.array(X_explain)
        
        # 2. 使用在__init__中创建的、正确的TreeExplainer计算SHAP值
        shap_values_for_pairs = self.explainer.shap_values(X_explain)

        # 3. 创建并返回DataFrame
        feature_names = self.compatibility_model.feature_names 
        if not feature_names:
            logger.warning("在模型中未找到特征名称，无法生成特征重要性报告。")
            return pd.DataFrame()
            
        # 确保shap_values.values和feature_names长度一致
        if shap_values_for_pairs.shape[1] != len(feature_names):
            logger.error(f"SHAP值的维度 ({shap_values_for_pairs.shape[1]}) 与特征名称的数量 ({len(feature_names)}) 不匹配！")
            return pd.DataFrame()

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": np.abs(shap_values_for_pairs).mean(axis=0),
            }
        ).sort_values("importance", ascending=False)

        logger.info("SHAP特征重要性计算完成。")
        return importance_df

    def get_room_shap_plot(self, room_id: str):
        """为特定宿舍生成SHAP瀑布图，解释其平均兼容性得分"""
        room_students = self.allocation_df[self.allocation_df["RoomID"] == room_id][
            "StudentID"
        ].tolist()

        if len(room_students) < 2:
            fig = go.Figure()
            fig.update_layout(title=f"宿舍 {room_id}: 人数不足，无法分析")
            return fig

        # 将真实StudentID转换为数字索引
        room_indices = []
        for student_id in room_students:
            try:
                # 在student_df中找到对应StudentID的行索引
                idx = self.student_df[self.student_df['StudentID'] == student_id].index[0]
                room_indices.append(idx)
            except IndexError:
                logger.warning(f"在数据中找不到StudentID {student_id}")
                continue

        if len(room_indices) < 2:
            fig = go.Figure()
            fig.update_layout(title=f"宿舍 {room_id}: 有效学生不足，无法分析")
            return fig

        room_pairs = [
            (room_indices[i], room_indices[j])
            for i in range(len(room_indices))
            for j in range(i + 1, len(room_indices))
        ]

        # 准备数据
        feature_vectors = np.array([
            self.compatibility_model.create_pair_feature_vector(self.processed_features, s1, s2)
            for s1, s2 in room_pairs
        ])

        # 计算该宿舍的平均SHAP值 (更正逻辑)
        # 错误的做法是解释平均特征向量，正确做法是平均每个特征向量的解释
        shap_values_for_pairs = self.explainer.shap_values(feature_vectors)
        shap_values = np.mean(shap_values_for_pairs, axis=0)

        base_names = self.preprocessor.get_feature_names()
        combined_feature_names = (
            [f"s1_{n}" for n in base_names]
            + [f"s2_{n}" for n in base_names]
            + [f"prod_{n}" for n in base_names]
            + [f"diff_{n}" for n in base_names]
        )

        # 过滤掉重要性很低的特征，使图表更清晰
        top_n = 15
        abs_shap_values = np.abs(shap_values)
        top_indices = np.argsort(abs_shap_values)[-top_n:]

        fig = go.Figure(
            go.Waterfall(
                name=f"宿舍 {room_id}",
                orientation="h",
                measure=["relative"] * len(top_indices) + ["total"],
                y=[combined_feature_names[i] for i in top_indices] + ["最终得分"],
                x=list(shap_values[top_indices]) + [np.sum(shap_values)],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                base=self.explainer.expected_value,
            )
        )

        fig.update_layout(
            title=f"宿舍 {room_id} 兼容性得分归因 (SHAP)",
            showlegend=False,
            height=max(500, top_n * 25),
            yaxis_title="特征",
            xaxis_title="对兼容性得分的贡献",
        )
        return fig

    def get_compatibility_distribution_plot(self) -> go.Figure:
        """创建宿舍兼容性得分的分布图（直方图和箱线图）"""
        metrics = self.get_allocation_metrics()
        compatibilities = metrics.get("compatibilities", [])

        if not compatibilities:
            return go.Figure().update_layout(title="无兼容性数据显示")

        fig = make_subplots(rows=1, cols=2, subplot_titles=("分布直方图", "箱线图"))

        fig.add_trace(
            go.Histogram(x=compatibilities, name="直方图", marker_color="#330C73"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Box(y=compatibilities, name="箱线图", marker_color="#330C73"),
            row=1,
            col=2,
        )

        fig.update_layout(title_text="宿舍平均兼容性得分分布", showlegend=False)
        return fig

    def get_feature_similarity_plot(self) -> go.Figure:
        """创建宿舍内学生关键特征的相似度热力图"""
        # 只选择数值型和类别型特征进行分析
        features_to_analyze = self.preprocessor.get_feature_names(
            numeric_only=True
        ) + self.preprocessor.get_feature_names(categorical_only=True)

        avg_similarities = {feat: [] for feat in features_to_analyze}

        for room_id, group in self.allocation_df.groupby("RoomID"):
            if len(group) < 2:
                continue

            # 将真实StudentID转换为数字索引
            room_indices = []
            for student_id in group["StudentID"].tolist():
                try:
                    idx = self.student_df[self.student_df['StudentID'] == student_id].index[0]
                    room_indices.append(idx)
                except IndexError:
                    logger.warning(f"在数据中找不到StudentID {student_id}")
                    continue
            
            if len(room_indices) < 2:
                continue

            room_features = self.processed_features[room_indices]

            # 计算每列特征的平均距离（不相似度）
            for i, feat_name in enumerate(self.feature_names):
                if feat_name in avg_similarities:
                    # pdist 计算成对距离，我们取平均值
                    # 对于类别特征，距离为0或1，对于数值特征，是归一化后的差值
                    avg_dist = np.mean(pdist(room_features[:, i].reshape(-1, 1)))
                    # 转换为相似度 (1 - distance)
                    avg_similarities[feat_name].append(1 - avg_dist)

        # 转换为DataFrame并计算每个特征的平均相似度
        similarity_df = pd.DataFrame(avg_similarities).mean().reset_index()
        similarity_df.columns = ["feature", "similarity"]
        similarity_df = similarity_df.sort_values(by="similarity", ascending=False)

        fig = px.bar(
            similarity_df,
            x="similarity",
            y="feature",
            orientation="h",
            title="宿舍内关键特征平均相似度",
            labels={"similarity": "平均相似度 (越高越好)", "feature": "特征"},
            color="similarity",
            color_continuous_scale=px.colors.sequential.Viridis,
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        return fig


# 用于独立测试的简单 main 函数
def main():
    """用于独立测试解释器功能的脚本"""
    logger.info("开始独立测试 AllocationExplainer...")

    # 1. 加载数据和模型
    try:
        student_df = pd.read_excel("Data.xlsx")
        # 更新为新的模型加载方法
        model = CompatibilityModel.load_model("compatibility_model")
        preprocessor = DataPreprocessor().fit(student_df)
    except FileNotFoundError as e:
        logger.error(f"测试失败：找不到必要的文件 - {e}。请确保从项目根目录运行此脚本，并已成功执行init_system.py。")
        return

    # 2. 创建一个模拟的分配结果
    allocation_data = {
        "RoomID": [1, 1, 2, 2, 3, 3],
        "StudentID": [1, 2, 3, 4, 5, 6],
        "Class": ["ClassA", "ClassA", "ClassB", "ClassB", "ClassC", "ClassC"]
    }
    allocation_df = pd.DataFrame(allocation_data)

    # 3. 初始化并使用解释器
    try:
        explainer = AllocationExplainer(model, preprocessor, student_df, allocation_df)

        # a. 获取指标
        metrics = explainer.get_allocation_metrics()
        print("\n--- 分配指标 ---")
        print(metrics)

        # b. 获取特征重要性
        importance = explainer.get_feature_importance()
        print("\n--- 特征重要性 (Top 5) ---")
        print(importance.head())

        # c. 生成图表（在测试中我们只调用函数，不显示）
        shap_fig = explainer.get_room_shap_plot(room_id=1)
        dist_fig = explainer.get_compatibility_distribution_plot()
        sim_fig = explainer.get_feature_similarity_plot()

        assert shap_fig is not None
        assert dist_fig is not None
        assert sim_fig is not None

        logger.info("\n所有解释器功能测试成功！")

    except Exception as e:
        logger.error(f"解释器测试期间发生错误: {e}", exc_info=True)


if __name__ == "__main__":
    main()
