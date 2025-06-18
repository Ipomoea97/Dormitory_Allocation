"""
宿舍分配优化模块
使用遗传算法优化宿舍分配方案
"""

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from compatibility_model import CompatibilityModel
from data_preprocessing import DataPreprocessor

# 尝试导入cupy，如果失败则将其设为None，以便后续进行设备检查
try:
    import cupy as cp
except ImportError:
    cp = None

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AllocationConfig:
    """分配配置参数"""

    population_size: int = 100  # 种群大小
    generations: int = 100  # 迭代代数
    mutation_rate: float = 0.1  # 变异率 (增加以促进多样性)
    crossover_rate: float = 0.8  # 交叉率
    elite_size: int = 10  # 精英个体数量
    tournament_size: int = 5  # 锦标赛选择大小
    
    # 奖励与惩罚权重
    same_class_bonus: float = 0.5  # 同班奖励 (大幅提高权重以确保班级优先)
    # 不再需要 penalty
    # same_class_penalty: float = 0.1

    # 宿舍配置 (现在由外部传入，这里不再需要)
    # male_6_rooms: int = 0
    # female_6_rooms: int = 0
    # male_4_rooms: int = 0
    # female_4_rooms: int = 0

class Individual:
    """个体类 - 代表一种分配方案"""

    def __init__(self, allocation: Dict[int, List[int]], fitness: float = 0.0):
        # 新的表示方法: {room_id: [student_id1, student_id2, ...]}
        self.allocation = allocation
        self.fitness = fitness
        # 不再需要 constraint_violations，因为约束将在生成时被满足
        # self.constraint_violations: float = 0.0

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

Population = List[Individual]


class AllocationOptimizer:
    """遗传算法分配优化器"""

    def __init__(
        self,
        student_df: pd.DataFrame,
        preprocessor: DataPreprocessor,
        model: CompatibilityModel,
        room_config: Dict,
        config: AllocationConfig,
        prioritize_class: bool = False, # 新增：班级优先开关
    ):
        self.student_df = student_df
        self.preprocessor = preprocessor
        self.compatibility_model = model
        self.room_config = room_config
        self.config = config
        self.prioritize_class = prioritize_class # 保存开关状态

        logger.info("优化器初始化...")
        logger.info(f"班级优先模式: {'开启' if self.prioritize_class else '关闭'}")

        # 1. 预处理数据
        self.processed_features = self.preprocessor.transform(self.student_df)
        self.student_id_to_pos = {student_id: i for i, student_id in enumerate(self.student_df.index)}
        logger.info(f"已转换 {len(self.student_df)} 名学生的特征。")

        # 2. 按性别分组学生ID
        self.students_by_gender = {
            "male": self.student_df[self.student_df["Sex"] == "男"].index.tolist(),
            "female": self.student_df[self.student_df["Sex"] == "女"].index.tolist(),
        }

        # 3. 初始化宿舍并创建床位（现在为所有学生）
        self._initialize_rooms()
        self._validate_capacity()

        # 4. 预计算兼容性矩阵以提高性能
        self.compatibility_matrix = {}
        # 预计算应针对所有学生，而不仅是优化子集
        self._precompute_compatibility_matrix("male")
        self._precompute_compatibility_matrix("female")

        logger.info("优化器设置完成。")

    def _initialize_rooms(self):
        """根据总学生数和宿舍配置创建所有宿舍"""
        self.rooms: List[Dict] = []
        self.room_capacities: Dict[int, int] = {}
        room_id_counter = 0

        def create_rooms_for_gender(gender, capacity, count):
            nonlocal room_id_counter
            students_needed = len(self.students_by_gender[gender])
            
            # 计算实际需要的宿舍数
            num_rooms_to_create = (students_needed + capacity - 1) // capacity
            actual_count = min(count, num_rooms_to_create)

            for _ in range(actual_count):
                room_id = room_id_counter
                self.rooms.append({"id": room_id, "capacity": capacity, "gender": gender})
                self.room_capacities[room_id] = capacity
                room_id_counter += 1
            return actual_count * capacity

        self.total_slots = {"male": 0, "female": 0}
        
        # 优先创建6人间
        self.total_slots["male"] += create_rooms_for_gender("male", 6, self.room_config.get("male_6", 0))
        self.total_slots["female"] += create_rooms_for_gender("female", 6, self.room_config.get("female_6", 0))
        # 再创建4人间
        self.total_slots["male"] += create_rooms_for_gender("male", 4, self.room_config.get("male_4", 0))
        self.total_slots["female"] += create_rooms_for_gender("female", 4, self.room_config.get("female_4", 0))
        
        self.num_rooms = len(self.rooms)
        logger.info(f"总共创建了 {self.num_rooms} 个宿舍。")

    def _validate_capacity(self):
        """验证总床位数是否足够"""
        for gender in ["male", "female"]:
            num_students = len(self.students_by_gender[gender])
            num_slots = self.total_slots[gender]
            if num_students > num_slots:
                logger.error(
                    f"性别 '{gender}' 的床位不足！"
                    f"需要 {num_students} 个床位, 但只创建了 {num_slots} 个。"
                )
                raise ValueError(f"CapacityError: Not enough beds for gender '{gender}'")
            else:
                 logger.info(f"性别 '{gender}': {num_slots} 个床位足以容纳 {num_students} 名学生。")


    def _precompute_compatibility_matrix(self, gender: str):
        """为特定性别的所有学生对预先计算兼容性得分"""
        logging.info(f"正在计算 {gender} 学生对的兼容性...")
        
        student_indices = self.students_by_gender[gender]
        
        if len(student_indices) < 2:
            logging.info(f"性别 {gender} 的学生不足两人，跳过预计算。")
            self.compatibility_matrix[gender] = {}
            return

        all_pairs = list(combinations(student_indices, 2))
        
        if not all_pairs:
            self.compatibility_matrix[gender] = {}
            return
        
        feature_vectors = np.array(
            [
                self.compatibility_model.create_pair_feature_vector(
                    self.processed_features,
                    self.student_id_to_pos[s1_idx],
                    self.student_id_to_pos[s2_idx],
                )
                for s1_idx, s2_idx in all_pairs
            ]
        )
        
        use_gpu = self.compatibility_model.get_booster().attributes().get('device') == 'cuda' and cp is not None
        if use_gpu:
            try:
                feature_vectors_device = cp.asarray(feature_vectors)
                scores = self.compatibility_model.predict(feature_vectors_device)
                scores = cp.asnumpy(scores)
            except Exception as e:
                logging.error(f"GPU预测失败: {e}，回退到CPU预测。")
                scores = self.compatibility_model.predict(feature_vectors)
        else:
            scores = self.compatibility_model.predict(feature_vectors)

        scores = np.clip(scores, 0, 1)

        gender_matrix = {}
        for i, (s1_idx, s2_idx) in enumerate(all_pairs):
            pair_key = tuple(sorted((s1_idx, s2_idx)))
            gender_matrix[pair_key] = scores[i]
        
        self.compatibility_matrix[gender] = gender_matrix
        logging.info(f"为 {gender} 学生预计算了 {len(all_pairs)} 对兼容性得分。")


    def _create_initial_individual(self) -> Individual:
        """
        创建一个满足硬约束的初始个体。
        如果开启班级优先，则优先将同班学生分配到同一宿舍。
        否则随机分配。
        """
        allocation = {room['id']: [] for room in self.rooms}
        
        for gender in ["male", "female"]:
            students = self.students_by_gender[gender].copy()
            gender_rooms = [r for r in self.rooms if r['gender'] == gender]
            
            if not gender_rooms:
                continue
            
            if self.prioritize_class:
                # 班级优先模式：按班级分组学生
                class_groups = {}
                for student_idx in students:
                    student_class = self.student_df.loc[student_idx, "Class"]
                    if student_class not in class_groups:
                        class_groups[student_class] = []
                    class_groups[student_class].append(student_idx)
                
                # 按班级大小排序，优先处理人数多的班级
                sorted_classes = sorted(class_groups.items(), key=lambda x: len(x[1]), reverse=True)
                
                room_idx = 0
                for class_name, class_students in sorted_classes:
                    random.shuffle(class_students)  # 班级内随机打乱
                    
                    student_idx = 0
                    while student_idx < len(class_students) and room_idx < len(gender_rooms):
                        room = gender_rooms[room_idx]
                        room_id = room['id']
                        capacity = self.room_capacities[room_id]
                        current_occupancy = len(allocation[room_id])
                        available_space = capacity - current_occupancy
                        
                        if available_space <= 0:
                            room_idx += 1
                            continue
                        
                        # 尽可能多地将同班学生分配到同一宿舍
                        num_to_assign = min(available_space, len(class_students) - student_idx)
                        assigned_students = class_students[student_idx : student_idx + num_to_assign]
                        allocation[room_id].extend(assigned_students)
                        student_idx += num_to_assign
                        
                        # 如果宿舍满了，移动到下一个宿舍
                        if len(allocation[room_id]) >= capacity:
                            room_idx += 1
            else:
                # 随机分配模式
                random.shuffle(students)
                
                student_idx = 0
                for room in gender_rooms:
                    room_id = room['id']
                    capacity = self.room_capacities[room_id]
                    
                    # 计算可以放入此宿舍的学生数量
                    num_to_assign = min(capacity, len(students) - student_idx)
                    
                    if num_to_assign <= 0:
                        break

                    assigned_students = students[student_idx : student_idx + num_to_assign]
                    allocation[room_id].extend(assigned_students)
                    student_idx += num_to_assign
        
        return Individual(allocation)


    def evaluate_fitness(self, individual: Individual) -> float:
        """
        评估个体的适应度
        适应度 = 基础兼容性得分 + [可选]班级优先奖励
        
        修复说明：
        1. 移除重复的班级纯度奖励，避免双重计分
        2. 简化奖励机制，使兼容性得分变化更加敏感
        3. 确保每一代都有机会产生有意义的适应度变化
        """
        total_compatibility = 0
        class_bonus = 0  # 重命名以明确其作用
        num_pairs = 0

        for room_id, students_in_room in individual.allocation.items():
            if len(students_in_room) < 2:
                continue

            gender = self.student_df.loc[students_in_room[0], "Sex"]
            gender_key = "male" if gender == "男" else "female"
            
            # 计算宿舍内学生对的兼容性
            room_compatibility = 0
            room_pairs = 0
            same_class_pairs = 0  # 记录同班学生对数量
            
            for s1_idx, s2_idx in combinations(students_in_room, 2):
                pair_key = tuple(sorted((s1_idx, s2_idx)))
                
                compatibility = self.compatibility_matrix[gender_key].get(pair_key, 0.5)
                total_compatibility += compatibility
                room_compatibility += compatibility
                num_pairs += 1
                room_pairs += 1

                # 检查是否为同班学生对
                if self.prioritize_class:
                    if self.student_df.loc[s1_idx, "Class"] == self.student_df.loc[s2_idx, "Class"]:
                        same_class_pairs += 1
            
            # 班级优先奖励：基于同班学生对的比例，而不是绝对数量
            if self.prioritize_class and room_pairs > 0:
                # 计算该宿舍同班学生对的比例
                same_class_ratio = same_class_pairs / room_pairs
                
                # 根据比例给予渐进式奖励，避免过度奖励
                # 这样可以让算法在不同的班级混合比例之间进行细致优化
                if same_class_ratio > 0:
                    # 奖励强度与同班比例成正比，但不过于强烈
                    room_class_bonus = same_class_ratio * self.config.same_class_bonus * 0.1
                    class_bonus += room_class_bonus
        
        # 计算平均兼容性作为基础得分
        base_fitness = total_compatibility / num_pairs if num_pairs > 0 else 0
        
        # 最终适应度 = 基础兼容性 + 班级奖励
        # 重要：确保基础兼容性仍然是主导因素
        final_fitness = base_fitness + class_bonus
        
        individual.fitness = final_fitness
        return final_fitness

    def evaluate_fitness_batch(self, individuals: List[Individual]) -> List[float]:
        """批量评估个体适应度，提高效率"""
        with ThreadPoolExecutor() as executor:
            fitness_values = list(executor.map(self.evaluate_fitness, individuals))
        return fitness_values

    def tournament_selection(self, population: List[Individual]) -> Individual:
        """锦标赛选择 - 使用已计算好的适应度"""
        tournament = random.sample(population, self.config.tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        健壮的交叉算子：交换两个宿舍的学生，保证约束。
        """
        child1_alloc = {rid: students.copy() for rid, students in parent1.allocation.items()}
        child2_alloc = {rid: students.copy() for rid, students in parent2.allocation.items()}

        if random.random() < self.config.crossover_rate:
            for gender in ["male", "female"]:
                gender_rooms = [r for r in self.rooms if r['gender'] == gender and len(parent1.allocation[r['id']]) > 0]
                if len(gender_rooms) < 2:
                    continue
                
                # 随机选择两个不同的宿舍进行交换
                room1_id, room2_id = random.sample([r['id'] for r in gender_rooms], 2)
                
                # 直接交换两个宿舍的全部学生
                child1_alloc[room1_id], child1_alloc[room2_id] = child1_alloc[room2_id], child1_alloc[room1_id]
                
                # 对child2做同样操作（可以使用不同的房间对以增加多样性）
                room3_id, room4_id = random.sample([r['id'] for r in gender_rooms], 2)
                child2_alloc[room3_id], child2_alloc[room4_id] = child2_alloc[room4_id], child2_alloc[room3_id]

        return Individual(child1_alloc), Individual(child2_alloc)

    def _targeted_class_mutation(self, allocation: Dict[int, List[int]], gender: str) -> bool:
        """
        定向变异：专为"班级优先"模式设计。
        尝试将一个"落单"的学生和他/她的同班同学安排在一起。
        返回一个布尔值，指示是否成功执行了定向变异。
        """
        # 1. 识别所有"落单"的学生（宿舍里没有同班同学）
        misplaced_students = []
        for room_id, students in allocation.items():
            if not students:
                continue
            
            room_gender = "male" if self.student_df.loc[students[0], "Sex"] == "男" else "female"
            if room_gender != gender:
                continue

            class_counts = {}
            for student_id in students:
                student_class = self.student_df.loc[student_id, "Class"]
                class_counts[student_class] = class_counts.get(student_class, 0) + 1
            
            for student_id in students:
                student_class = self.student_df.loc[student_id, "Class"]
                if class_counts[student_class] == 1: # 这个班的就他一个人
                    misplaced_students.append((student_id, room_id))

        if not misplaced_students:
            return False # 没有落单的学生，无法进行此变异
        
        random.shuffle(misplaced_students)

        # 2. 尝试为落单学生找到一个"家"
        for student_id, source_room_id in misplaced_students:
            student_class = self.student_df.loc[student_id, "Class"]
            
            # 3. 寻找一个有该学生同班同学的目标宿舍
            for target_room_id, target_students in allocation.items():
                if source_room_id == target_room_id or not target_students:
                    continue
                
                # 确保目标宿舍性别一致
                target_gender = "male" if self.student_df.loc[target_students[0], "Sex"] == "男" else "female"
                if target_gender != gender:
                    continue

                # 检查目标宿舍是否有同班同学
                has_classmates = any(self.student_df.loc[s, "Class"] == student_class for s in target_students)
                if not has_classmates:
                    continue

                # 4. 在目标宿舍中寻找一个可以交换的"非同班"学生
                swappable_targets = [
                    s for s in target_students 
                    if self.student_df.loc[s, "Class"] != student_class
                ]

                if swappable_targets:
                    # 找到了一个完美的交换机会
                    target_student_id = random.choice(swappable_targets)
                    
                    # 执行交换
                    allocation[source_room_id].remove(student_id)
                    allocation[source_room_id].append(target_student_id)
                    allocation[target_room_id].remove(target_student_id)
                    allocation[target_room_id].append(student_id)
                    
                    # 成功执行了一次定向变异，立即返回
                    return True
        
        return False # 遍历完所有可能也没找到合适的交换机会


    def mutate(self, individual: Individual) -> Individual:
        """
        混合变异策略：
        - 在班级优先模式下，高概率执行"定向变异"，小概率执行"随机变异"。
        - 在普通模式下，只执行"随机变异"。
        """
        mutated_allocation = {rid: students.copy() for rid, students in individual.allocation.items()}

        if random.random() >= self.config.mutation_rate:
            return Individual(mutated_allocation) # 未达到变异率，不发生变异

        for gender in ["male", "female"]:
            # 找到该性别所有有学生的宿舍
            occupied_rooms = [
                r['id'] for r in self.rooms 
                if r['gender'] == gender and len(mutated_allocation[r['id']]) > 0
            ]
            
            if len(occupied_rooms) < 2:
                continue

            # --- 混合变异逻辑 ---
            mutated = False
            # 如果是班级优先模式，并且随机数落在定向变异的概率区间
            if self.prioritize_class and random.random() < 0.7: # 70% 概率尝试定向变异
                mutated = self._targeted_class_mutation(mutated_allocation, gender)

            # 如果定向变异没有执行，或者是非班级优先模式，则执行随机变异
            if not mutated:
                # 随机选择两个不同的宿舍
                room1_id, room2_id = random.sample(occupied_rooms, 2)
                
                # 如果两个宿舍都有学生，则可以进行交换
                if len(mutated_allocation[room1_id]) > 0 and len(mutated_allocation[room2_id]) > 0:
                    student1 = random.choice(mutated_allocation[room1_id])
                    student2 = random.choice(mutated_allocation[room2_id])
                    
                    # 交换他们
                    mutated_allocation[room1_id].remove(student1)
                    mutated_allocation[room1_id].append(student2)
                    mutated_allocation[room2_id].remove(student2)
                    mutated_allocation[room2_id].append(student1)
        
        return Individual(mutated_allocation)


    def run(self, progress_callback=None) -> Tuple[List[Individual], Individual]:
        """
        运行遗传算法，对所有学生进行优化。
        
        优化说明：
        1. 增加适应度变化的记录频率
        2. 改善进展报告机制
        3. 确保每一代都有机会展示变化
        """
        logger.info("开始遗传算法优化...")
        
        # 1. 初始化种群并首次评估
        population = [self._create_initial_individual() for _ in range(self.config.population_size)]
        
        fitness_values = self.evaluate_fitness_batch(population)
        for ind, fitness in zip(population, fitness_values):
            ind.fitness = fitness

        best_individual = max(population, key=lambda ind: ind.fitness)
        previous_best_fitness = best_individual.fitness

        for generation in tqdm(range(self.config.generations), desc="遗传算法优化进度"):
            # 2. 选择 - 精英直接进入下一代
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            elites = population[:self.config.elite_size]
            
            # 3. 生成后代
            offspring = []
            while len(offspring) < self.config.population_size - self.config.elite_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                # 每个孩子都有机会变异
                offspring.append(self.mutate(child1))
                if len(offspring) < self.config.population_size - self.config.elite_size:
                    offspring.append(self.mutate(child2))

            # 4. 评估新生成的后代
            new_fitness_values = self.evaluate_fitness_batch(offspring)
            for ind, fitness in zip(offspring, new_fitness_values):
                ind.fitness = fitness

            # 5. 组成新一代种群
            population = elites + offspring

            # 6. 更新全局最优解并报告进展
            current_best_in_gen = max(population, key=lambda ind: ind.fitness)
            current_best_fitness = current_best_in_gen.fitness
            
            # 如果发现更好的解，更新最佳个体
            if current_best_fitness > best_individual.fitness:
                best_individual = current_best_in_gen
                progress_message = f"发现更优解! 适应度提升至 {current_best_fitness:.6f}"
            else:
                # 即使没有找到更好的解，也报告当前状态
                # 这有助于用户了解算法仍在运行
                fitness_diff = current_best_fitness - previous_best_fitness
                if abs(fitness_diff) > 1e-8:  # 有微小变化
                    progress_message = f"种群进化中... 当前最佳: {current_best_fitness:.6f} (变化: {fitness_diff:+.6f})"
                else:
                    progress_message = f"稳定搜索中... 当前最佳: {current_best_fitness:.6f}"

            # 无论是否有改进，都调用进度回调
            if progress_callback:
                progress_callback(
                    generation + 1,
                    best_individual.fitness,  # 始终报告历史最佳适应度
                    progress_message,
                )
            
            # 更新前一代最佳适应度，用于下次比较
            previous_best_fitness = current_best_fitness
            
            time.sleep(0.01)

        logger.info("遗传算法优化完成。")
        
        return population, best_individual

    def get_allocation_dataframe(self, best_individual: Individual) -> pd.DataFrame:
        """
        根据最优个体的分配方案创建最终的分配结果DataFrame。
        """
        allocations = []
        for room_id, student_ids in best_individual.allocation.items():
            if not student_ids:
                continue

            room_info = next((r for r in self.rooms if r["id"] == room_id), None)
            if room_info:
                for student_id in student_ids:
                    student_info = self.student_df.loc[student_id]
                    allocations.append({
                        "StudentID": student_info["StudentID"],  # 使用真实的StudentID
                        "Name": student_info["Name"],
                        "Sex": student_info["Sex"],
                        "Class": student_info["Class"],
                        "MBTI": student_info["MBTI"],
                        "RoomID": f"D{room_id}",
                        "RoomType": f"{room_info['capacity']}人间" # 统一房间类型名称
                    })
        
        return pd.DataFrame(allocations)


# 用于独立测试的简单 main 函数
def main():
    """用于测试优化器功能的独立脚本"""
    from data_preprocessing import DataPreprocessor

    logger.info("开始独立测试 AllocationOptimizer...")

    # 1. 加载和准备数据/模型
    try:
        student_df = pd.read_excel("Data.xlsx")
        # 保持数字索引，与主系统保持一致
        preprocessor = DataPreprocessor().fit(student_df)
        model = CompatibilityModel.load_model("compatibility_model")
    except FileNotFoundError as e:
        logger.error(f"测试失败：找不到必要的文件 - {e}。")
        return

    # 2. 设置配置
    num_male_students = len(student_df[student_df["Sex"] == "男"])
    num_female_students = len(student_df[student_df["Sex"] == "女"])
    
    # 提供足够的宿舍以容纳所有学生
    room_config = {
        "male_6": (num_male_students + 5) // 6,
        "female_6": (num_female_students + 5) // 6,
        "male_4": (num_male_students + 3) // 4, # 提供4人间作为备选
        "female_4": (num_female_students + 3) // 4,
    }

    ga_config = AllocationConfig(
        population_size=50,
        generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_size=5,
        tournament_size=3,
        same_class_bonus=0.2, # 为测试设置奖励
    )

    # 3. 初始化并运行优化器
    try:
        # 测试班级优先模式开启
        optimizer = AllocationOptimizer(
            student_df=student_df,
            preprocessor=preprocessor,
            model=model,
            room_config=room_config,
            config=ga_config,
            prioritize_class=True, # 开启班级优先
        )

        def simple_progress_callback(gen, fitness, msg):
            print(
                f"Generation {gen}/{ga_config.generations}, Best Fitness: {fitness:.4f}"
            )

        _, best_solution = optimizer.run(simple_progress_callback)

        # 4. 打印结果
        logger.info("优化完成，显示最佳分配结果：")
        allocation_df = optimizer.get_allocation_dataframe(best_solution)
        print(allocation_df.sort_values(by=["RoomID", "StudentID"]).to_string())

        # 验证每个宿舍的人数是否等于或小于其容量
        print("\n--- 宿舍容量验证 ---")
        room_counts = allocation_df.groupby('RoomID').size().reset_index(name='counts')
        for _, row in room_counts.iterrows():
            room_id_num = int(row['RoomID'][1:]) # 'D1' -> 1
            capacity = optimizer.room_capacities[room_id_num]
            count = row['counts']
            status = "OK" if count <= capacity else "OVERLOADED!"
            print(f"宿舍 {row['RoomID']} (容量: {capacity}) | 入住人数: {count} -> {status}")

    except Exception as e:
        logger.error(f"测试期间发生错误: {e}", exc_info=True)


if __name__ == "__main__":
    main()
