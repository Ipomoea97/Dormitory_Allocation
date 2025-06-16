import warnings
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 建议避免使用全局的 'ignore'，因为它会抑制所有警告，可能隐藏重要问题。
# 如果需要，请使用 'with warnings.catch_warnings():' 在特定代码块中管理警告。
# warnings.filterwarnings("ignore")


class DataPreprocessor:
    """
    数据预处理类，负责将原始学生数据转换为机器学习模型可用的特征向量。
    该类遵循scikit-learn的fit/transform设计模式，以确保转换的一致性。
    """

    def __init__(self):
        """初始化所有需要的转换器。"""
        self.scaler = StandardScaler()
        self.hobby_vectorizer = CountVectorizer(
            token_pattern=r"[^,]+", 
            preprocessor=lambda text: str(text).strip()
        )
        self.province_encoder = LabelEncoder()
        self.type_encoder = LabelEncoder()
        self.mbti_dimensions = ["MBTI_E", "MBTI_S", "MBTI_T", "MBTI_J"]
        self.numerical_features = ["Age", "Gaokao", "Sleep_minutes"]
        self.categorical_features = ["Province", "Type"]
        self._is_fitted = False

    def _parse_sleep_time(self, time_str: str) -> int:
        """
        将睡觉时间字符串（如 "23:30"）转换为从午夜开始的分钟数。
        处理跨午夜的情况，例如 "1:00"（凌晨1点）。
        """
        try:
            hour, minute = map(int, str(time_str).split(":"))
            # 凌晨0-5点的时间认为是第二天
            if 0 <= hour <= 5:
                return (hour + 24) * 60 + minute
            return hour * 60 + minute
        except (ValueError, AttributeError):
            # 如果格式不正确或为空，返回一个平均值（例如晚上11点）
            return 23 * 60

    def _parse_mbti(self, mbti_str: str) -> List[int]:
        """
        将MBTI类型字符串拆分为四个二元维度。
        ENFJ -> [1, 0, 0, 1] (E=1/I=0, S=1/N=0, T=1/F=0, J=1/P=0)
        """
        if not isinstance(mbti_str, str) or len(mbti_str) != 4:
            return [0, 0, 0, 0]

        return [
            1 if mbti_str[0] == "E" else 0,  # E/I: E=1, I=0
            1 if mbti_str[1] == "S" else 0,  # S/N: S=1, N=0
            1 if mbti_str[2] == "T" else 0,  # T/F: T=1, F=0
            1 if mbti_str[3] == "J" else 0,  # J/P: J=1, P=0
        ]

    def fit(self, df: pd.DataFrame):
        """
        使用完整的数据集"学习"转换规则（拟合scaler和encoder）。
        此方法应在任何transform调用之前被调用一次。
        """
        df_copy = df.copy()

        # 拟合数值特征的scaler
        df_copy["Sleep_minutes"] = df_copy["Sleep"].apply(self._parse_sleep_time)
        self.scaler.fit(df_copy[self.numerical_features])

        # 拟合类别特征的encoder
        self.province_encoder.fit(df_copy["Province"])
        self.type_encoder.fit(df_copy["Type"])

        # 拟合兴趣爱好的vectorizer
        # 使用fillna('')确保即使有空值也能正常处理
        # 注意: token_pattern=r'[^,]+' 在处理 '爱好1, 爱好2' 时可能会产生带空格的词条 ' 爱好2'。
        # 如果数据格式不规范，考虑提供一个自定义的tokenizer来分割和清理空白。
        self.hobby_vectorizer.fit(df_copy["Hobby"].fillna(""))

        self._is_fitted = True
        print("预处理器拟合完成。")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        使用已学习的规则转换数据。
        """
        if not self._is_fitted:
            raise RuntimeError("预处理器尚未拟合。请在transform之前调用fit方法。")

        df_copy = df.copy()

        # 1. MBTI特征
        mbti_features = pd.DataFrame(
            df_copy["MBTI"].apply(self._parse_mbti).tolist(),
            columns=self.mbti_dimensions,
            index=df_copy.index,
        )

        # 2. 数值特征
        df_copy["Sleep_minutes"] = df_copy["Sleep"].apply(self._parse_sleep_time)
        numerical_scaled = self.scaler.transform(df_copy[self.numerical_features])
        numerical_df = pd.DataFrame(
            numerical_scaled,
            columns=[f"{col}_scaled" for col in self.numerical_features],
            index=df_copy.index,
        )

        # 3. 类别特征
        province_encoded = self.province_encoder.transform(df_copy["Province"])
        type_encoded = self.type_encoder.transform(df_copy["Type"])
        categorical_df = pd.DataFrame(
            {"Province_encoded": province_encoded, "Type_encoded": type_encoded},
            index=df_copy.index,
        )

        # 4. 兴趣爱好特征
        hobby_matrix = self.hobby_vectorizer.transform(
            df_copy["Hobby"].fillna("")
        ).toarray()
        hobby_df = pd.DataFrame(
            hobby_matrix,
            columns=self.get_feature_names(categorical_only=True, hobby_only=True),
            index=df_copy.index,
        )

        # 5. 性别特征
        sex_df = pd.DataFrame(
            {"Sex_encoded": (df_copy["Sex"] == "男").astype(int)}, index=df_copy.index
        )

        # 组合所有特征
        final_features = pd.concat(
            [sex_df, numerical_df, mbti_features, categorical_df, hobby_df], axis=1
        )

        # 确保特征顺序与get_feature_names一致
        return final_features[self.get_feature_names()].values

    def get_feature_names(
        self, numeric_only=False, categorical_only=False, hobby_only=False
    ) -> List[str]:
        """
        获取用于模型训练的特征列名列表，顺序是固定的。
        """
        hobby_features = self.hobby_vectorizer.get_feature_names_out()

        if hobby_only:
            return [f"Hobby_{hobby}" for hobby in hobby_features]

        numerical = [f"{col}_scaled" for col in self.numerical_features]
        if numeric_only:
            return numerical

        categorical = ["Province_encoded", "Type_encoded"]
        if categorical_only:
            return categorical

        base_features = ["Sex_encoded"] + numerical + self.mbti_dimensions + categorical

        return base_features + [f"Hobby_{hobby}" for hobby in hobby_features]

    def get_feature_names_for_pairs(self) -> List[str]:
        """获取用于学生对组合特征的名称列表"""
        base_features = self.get_feature_names()
        diff_features = [f"diff_{name}" for name in base_features]
        prod_features = [f"prod_{name}" for name in base_features]
        return diff_features + prod_features

    def _get_mbti_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """将MBTI类型转换为多列二进制特征"""
        return pd.DataFrame(
            df["MBTI"].apply(self._parse_mbti).tolist(),
            columns=self.mbti_dimensions,
            index=df.index,
        )


def test_preprocessing():
    """
    测试数据预处理功能
    """
    print("=== 测试数据预处理模块 ===")

    # 加载数据
    try:
        df = pd.read_excel("Data.xlsx")
    except FileNotFoundError:
        print("错误：测试需要 'Data.xlsx' 文件，但未在当前目录找到。")
        return
        
    print(f"原始数据形状: {df.shape}")

    # 初始化预处理器并拟合
    preprocessor = DataPreprocessor()
    preprocessor.fit(df)

    # 转换整个数据集
    features = preprocessor.transform(df)
    print(f"\n转换后的特征矩阵形状: {features.shape}")

    # 检查特征名称
    feature_names = preprocessor.get_feature_names()
    print(f"总特征数量: {len(feature_names)}")
    print(f"部分特征名称: {feature_names[:5]}...{feature_names[-5:]}")

    # 测试转换一个子集，检查形状是否一致
    subset_features = preprocessor.transform(df.head(10))
    print(f"\n转换子集后的特征矩阵形状: {subset_features.shape}")
    assert subset_features.shape[1] == features.shape[1], "子集转换后的特征维度不一致！"
    print("子集转换维度检查通过！")


if __name__ == "__main__":
    test_preprocessing()
