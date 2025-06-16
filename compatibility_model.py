import json
import random
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

# 尝试导入cupy，如果失败则将其设为None
try:
    import cupy as cp
except ImportError:
    cp = None

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 建议避免使用全局的 'ignore'，因为它会抑制所有警告，可能隐藏重要问题。
# warnings.filterwarnings("ignore")


class CompatibilityModel:
    """
    宿舍兼容性预测模型，使用XGBoost预测一组学生的兼容性得分
    """

    def __init__(self, room_size=4):
        self.room_size = room_size
        self.feature_names = None
        self.model = self._initialize_model()

    def _initialize_model(self):
        """根据环境（CPU/GPU）初始化XGBoost模型"""
        use_gpu = False
        if cp is not None:
            try:
                # 简单的GPU可用性检查
                cp.cuda.runtime.getDeviceCount()
                use_gpu = True
                logger.info("检测到GPU和CuPy，将启用GPU加速。")
            except Exception:
                logger.info("未检测到可用GPU，使用CPU。")
        
        params = {
            "objective": "reg:squarederror",
            "booster": "gbtree",
            "tree_method": "hist",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }
        if use_gpu:
            params["device"] = "cuda"
            
        return xgb.XGBRegressor(**params)

    def calculate_compatibility_score(self, student_group):
        """
        计算一组学生的兼容性得分（代理目标函数）

        参数:
        student_group: list of dict，每个dict包含学生的特征信息

        返回:
        float: 兼容性得分 (0-1之间，越高越兼容)
        """
        if len(student_group) < 2:
            return 0.5

        scores = []

        # 1. 作息时间一致性 (权重: 0.4)
        sleep_times = [s["sleep_minutes"] for s in student_group]
        sleep_std = np.std(sleep_times)
        # 将标准差转换为0-1分数，标准差越小分数越高
        sleep_score = max(0, 1 - sleep_std / 120)  # 120分钟作为参考标准差
        scores.append(("sleep_consistency", sleep_score, 0.4))

        # 2. 兴趣爱好重叠度 (权重: 0.25)
        hobby_sets = [set(s["hobbies"]) for s in student_group]
        if len(hobby_sets) >= 2:
            # 计算所有配对的Jaccard相似度
            hobby_similarities = []
            for i in range(len(hobby_sets)):
                for j in range(i + 1, len(hobby_sets)):
                    intersection = len(hobby_sets[i] & hobby_sets[j])
                    union = len(hobby_sets[i] | hobby_sets[j])
                    if union > 0:
                        hobby_similarities.append(intersection / union)
                    else:
                        hobby_similarities.append(0)
            hobby_score = np.mean(hobby_similarities) if hobby_similarities else 0
        else:
            hobby_score = 0
        scores.append(("hobby_overlap", hobby_score, 0.25))

        # 3. MBTI互补性 (权重: 0.2)
        mbti_vectors = [s["mbti_vector"] for s in student_group]
        # 计算MBTI多样性（既不能太相似也不能太不同）
        mbti_diversity = []
        for dim in range(4):  # 4个MBTI维度
            dim_values = [vec[dim] for vec in mbti_vectors]
            # 理想情况是有适度的多样性
            diversity = np.std(dim_values)
            # 标准差在0.3-0.5之间为最优
            if 0.3 <= diversity <= 0.5:
                mbti_diversity.append(1.0)
            else:
                mbti_diversity.append(max(0, 1 - abs(diversity - 0.4) * 2))
        mbti_score = np.mean(mbti_diversity)
        scores.append(("mbti_complementarity", mbti_score, 0.2))

        # 4. 学业水平相似性 (权重: 0.1)
        gaokao_scores = [s["gaokao_scaled"] for s in student_group]
        gaokao_std = np.std(gaokao_scores)
        # 学业水平不要差距太大
        academic_score = max(0, 1 - gaokao_std)
        scores.append(("academic_similarity", academic_score, 0.1))

        # 5. 地域多样性奖励 (权重: 0.05)
        provinces = [s["province"] for s in student_group]
        unique_provinces = len(set(provinces))
        diversity_bonus = min(1.0, unique_provinces / len(student_group))
        scores.append(("geographical_diversity", diversity_bonus, 0.05))

        # 计算加权平均分
        total_score = sum(score * weight for _, score, weight in scores)

        return max(0, min(1, total_score))  # 确保分数在0-1之间

    def extract_student_features(self, student_df, processed_df, student_indices):
        """
        一个已废弃的方法，保留以避免破坏旧的调用，但不执行任何操作。
        """
        pass

    @staticmethod
    def create_pair_feature_vector(
        processed_features: np.ndarray,
        student1_pos: int,
        student2_pos: int,
    ) -> np.ndarray:
        """
        根据两个学生的位置索引创建他们的组合特征向量。
        这是一个静态方法，因为它不依赖于任何实例状态。
        """
        vec1 = processed_features[student1_pos]
        vec2 = processed_features[student2_pos]
        
        if student1_pos > student2_pos:
            vec1, vec2 = vec2, vec1
            
        diff = np.abs(vec1 - vec2)
        prod = vec1 * vec2
        
        return np.concatenate([diff, prod])

    def train_model(self, student_df, processed_df, preprocessor, n_samples=5000):
        """
        使用合成数据训练XGBoost兼容性模型，并评估其性能。
        """
        logger.info(f"开始生成 {n_samples} 个训练样本...")
        self.feature_names = preprocessor.get_feature_names_for_pairs()
        
        X = []
        y = []
        
        student_indices = list(range(len(student_df)))

        for _ in tqdm(range(n_samples), desc="生成训练数据"):
            s1_pos, s2_pos = random.sample(student_indices, 2)
            
            feature_vector = CompatibilityModel.create_pair_feature_vector(processed_df, s1_pos, s2_pos)
            X.append(feature_vector)
            
            label = self._generate_pseudo_label(
                student_df.iloc[s1_pos], student_df.iloc[s2_pos]
            )
            y.append(label)

        X = np.array(X)
        y = np.array(y)
        
        logger.info("训练数据生成完毕。")
        
        # 划分训练集和测试集进行评估
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info("正在使用训练集训练评估模型...")
        self.model.fit(X_train, y_train)
        
        # 在测试集上评估
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"模型评估结果 - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # 使用所有数据重新训练最终模型
        logger.info("正在使用所有合成数据重新训练最终模型...")
        self.model.fit(X, y)
        logger.info("最终模型训练完成。")

    def _generate_pseudo_label(self, student1: pd.Series, student2: pd.Series) -> float:
        """
        根据一些启发式规则为学生对生成伪标签 (0到1之间)
        """
        score = 0.5  # 基础分

        # 1. MBTI 相似度
        mbti_similarity = self._calculate_mbti_similarity(student1["MBTI"], student2["MBTI"])
        score += mbti_similarity * 0.3 # 权重

        # 2. 兴趣爱好重叠
        hobbies1 = set(str(student1["Hobby"]).split(","))
        hobbies2 = set(str(student2["Hobby"]).split(","))
        common_hobbies = len(hobbies1.intersection(hobbies2))
        score += min(common_hobbies * 0.1, 0.3)

        # 3. 生活习惯: 早睡 vs 晚睡
        if student1["Sleep"] == student2["Sleep"]:
            score += 0.15
        
        return np.clip(score, 0, 1)

    def _calculate_mbti_similarity(self, mbti1, mbti2):
        if not isinstance(mbti1, str) or not isinstance(mbti2, str) or len(mbti1) != 4 or len(mbti2) != 4:
            return 0
        similarity = 0
        if mbti1[0] == mbti2[0]: similarity += 0.1
        if mbti1[1] == mbti2[1]: similarity += 0.3
        if mbti1[2] == mbti2[2]: similarity += 0.3
        if mbti1[3] == mbti2[3]: similarity += 0.1
        return similarity

    def save_model(self, path_prefix: str):
        """保存模型和元数据"""
        if not self.model:
            raise RuntimeError("模型尚未训练，无法保存。")

        model_filepath = f"{path_prefix}_xgb.json"
        meta_filepath = f"{path_prefix}_meta.json"

        self.model.save_model(model_filepath)

        meta_data = {
            "feature_names": self.feature_names,
            "room_size": self.room_size,
        }
        with open(meta_filepath, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"模型已保存至 {path_prefix}_xgb.json 和 {path_prefix}_meta.json")

    @classmethod
    def load_model(cls, path_prefix: str):
        """加载模型和元数据"""
        model_filepath = f"{path_prefix}_xgb.json"
        meta_filepath = f"{path_prefix}_meta.json"

        try:
            instance = cls()
            instance.model.load_model(model_filepath)
            with open(meta_filepath, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            instance.feature_names = meta_data.get("feature_names")
            instance.room_size = meta_data.get("room_size")
            logger.info(f"模型已从 {path_prefix} 相关文件成功加载。")
            return instance
        except Exception as e:
            logger.error(f"加载模型失败: {e}。请确保文件存在且格式正确，或运行 init_system.py 重新生成。")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用加载的模型进行预测"""
        if self.model is None:
            raise RuntimeError("模型尚未加载。")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def predict_compatibility(self, processed_df: np.ndarray, room_indices: List[int]) -> float:
        """
        预测一个完整宿舍的兼容性得分。
        通过预测宿舍内所有学生对的兼容性得分，然后取平均值来实现。
        """
        if len(room_indices) < 2:
            return 0.5  # 少于2人的宿舍得分设为中性值

        # 1. 创建宿舍内所有学生对
        student_pairs = list(combinations(room_indices, 2))
        
        # 2. 为所有对创建特征向量
        feature_vectors = np.array([
            CompatibilityModel.create_pair_feature_vector(processed_df, s1_idx, s2_idx)
            for s1_idx, s2_idx in student_pairs
        ])
        
        # 3. 批量预测
        if feature_vectors.shape[0] > 0:
            pair_scores = self.predict(feature_vectors)
            # 4. 返回平均分
            return np.mean(pair_scores)
        
        return 0.5

    def get_booster(self):
        """获取底层的XGBoost booster对象"""
        return self.model.get_booster()

    def predict_from_ids(self, student_ids: List[int], student_df: pd.DataFrame, preprocessor: 'DataPreprocessor') -> List[float]:
        """
        根据学生ID列表，预测所有学生对的兼容性得分。
        """
        if len(student_ids) < 2:
            return []

        processed_features = preprocessor.transform(student_df)
        student_id_to_pos = {student_id: i for i, student_id in enumerate(student_df.index)}
        
        student_pairs = list(combinations(student_ids, 2))
        
        feature_vectors = np.array([
            CompatibilityModel.create_pair_feature_vector(
                processed_features,
                student_id_to_pos[s1],
                student_id_to_pos[s2]
            )
            for s1, s2 in student_pairs
        ])

        if feature_vectors.size == 0:
            return []

        scores = self.predict(feature_vectors)
        return scores.tolist()


def train_and_evaluate_model(
    student_df: pd.DataFrame,
    preprocessor: "DataPreprocessor",
    n_samples=10000,
):
    """独立的模型训练和评估函数"""
    logger.info("--- 开始兼容性模型训练与评估 ---")
    
    processed_df = preprocessor.transform(student_df)
    model = CompatibilityModel()
    
    logger.info("开始训练...")
    model.train_model(student_df, processed_df, preprocessor, n_samples=n_samples)
    
    logger.info("训练完成。")
    return model


def test_compatibility_model():
    """用于测试模型功能的简单脚本"""
    from data_preprocessing import DataPreprocessor

    try:
        df = pd.read_excel("Data.xlsx")
        preprocessor = DataPreprocessor().fit(df)
        
        # 训练
        model = train_and_evaluate_model(df, preprocessor, n_samples=1000)
        
        # 保存
        model.save_model("compatibility_model")
        
        # 加载
        loaded_model = CompatibilityModel.load_model("compatibility_model")
        
        # 预测
        student_ids_to_test = df.index[:4].tolist()
        scores = loaded_model.predict_from_ids(student_ids_to_test, df, preprocessor)
        
        logger.info(f"为学生 {student_ids_to_test} 预测的兼容性得分: {scores}")
        logger.info("测试成功!")

    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)


if __name__ == "__main__":
    test_compatibility_model()
