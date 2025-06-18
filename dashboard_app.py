"""
宿舍分配系统Web界面
"""

import os

# 警告修复：在导入numpy/sklearn之前设置OMP_NUM_THREADS，以避免MKL在Windows上的内存泄漏警告
# 根据警告信息，将线程数设置为一个较小的值
if os.name == 'nt' and 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '2'

import logging
import re
import threading
import uuid
from datetime import datetime
from typing import Any, Dict
import json
import numpy as np

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dash_table, dcc, html
from plotly.colors import sample_colorscale
import dash_bootstrap_components as dbc

from allocation_optimizer import AllocationConfig, AllocationOptimizer
from compatibility_model import CompatibilityModel
# 导入自定义模块
from data_preprocessing import DataPreprocessor
from explanation_module import AllocationExplainer
from waitress import serve

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# 仅显示 werkzeug 的 WARNING 及更高级别的日志，隐藏 "GET /" 等信息
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# --- 辅助工具 ---
class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理Numpy数据类型"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# 初始化Dash应用 - Apple Design风格
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",  # Font Awesome图标
        "/assets/apple_style.css",  # 引入Apple Design样式
    ],
    suppress_callback_exceptions=True,
    url_base_pathname="/",
    title="🏠 宿舍分配系统",
    update_title="🏠 宿舍分配系统 - 正在加载...",
)

# --- 全局状态和组件 ---
# 全局组件（应用启动时加载一次，后续只读）
preprocessor = DataPreprocessor()
compatibility_model = None  # 在初始化时设为None，将由initialize_system填充
student_data = None

# 用于在后台线程和Dash回调之间安全传递状态的全局字典
# 键是 session_id，值是该会话的状态信息
optimization_threads = {}


# --- 系统初始化 ---
def initialize_system():
    """初始化系统组件，加载数据、模型和预处理器"""
    global student_data, preprocessor, compatibility_model
    try:
        logger.info("初始化系统组件...")
        student_data = pd.read_excel("Data.xlsx")
        # 保存原始StudentID列，但不设置为索引，以避免索引错误
        # 系统内部使用数字索引，显示时使用StudentID
        logger.info(f"成功加载学生数据: {len(student_data)} 名学生")

        logger.info("正在拟合数据预处理器...")
        preprocessor.fit(student_data)
        logger.info("数据预处理器拟合完成。")

        # 使用新的类方法加载模型，该方法返回一个完整的实例
        compatibility_model = CompatibilityModel.load_model("compatibility_model")
        logger.info("兼容性模型加载成功。")

        logger.info("系统初始化完成!")
        return True
    except FileNotFoundError as e:
        logger.error(f"初始化失败：找不到必须的文件。请确保 Data.xlsx 和相关的模型文件（如 compatibility_model_xgb.json）存在。详细错误: {e}")
        return False
    except Exception as e:
        logger.error(f"系统初始化时发生未知错误: {e}", exc_info=True)
        return False


# --- 应用布局 ---
# Apple Design样式定义
# 这些样式现在都在CSS文件中定义，不再需要在Python中定义


def create_layout():
    """创建并返回应用的主布局 - Apple Design风格"""
    sidebar = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src="/assets/SCAU_LOGO.png", 
                                style={
                                    "height": "36px", 
                                    "width": "36px", 
                                    "margin-right": "8px",
                                    "flex-shrink": "0"
                                }
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "华南农业大学食品学院", 
                                        style={
                                            "font-size": "18px",
                                            "font-weight": "700",
                                            "line-height": "1.2",
                                            "background": "linear-gradient(135deg, var(--apple-blue) 0%, var(--apple-purple) 100%)",
                                            "-webkit-background-clip": "text",
                                            "-webkit-text-fill-color": "transparent",
                                            "background-clip": "text",
                                            "margin": "0"
                                        }
                                    ),
                                    html.Div(
                                        "宿舍分配系统", 
                                        style={
                                            "font-size": "18px",
                                            "font-weight": "600",
                                            "line-height": "1.2",
                                            "color": "var(--apple-text-secondary)",
                                            "margin": "0"
                                        }
                                    ),
                                ],
                                style={"flex": "1", "min-width": "0"}
                            ),
                        ],
                        style={
                            "display": "flex", 
                            "align-items": "center",
                            "width": "100%",
                            "overflow": "hidden"
                        }
                    ),
                    html.Div(
                        "v1.0.0", 
                        className="apple-version"
                    ),
                ],
                style={"padding": "32px 20px 24px 20px", "border-bottom": "1px solid #E5E5EA"}
            ),
            html.Div(
            dbc.Nav(
                [
                        dbc.NavLink(
                            [
                                html.I(className="fas fa-chart-bar", style={"margin-right": "12px", "width": "16px"}),
                                "数据概览"
                            ],
                            href="/", 
                            active="exact", 
                            id="nav-overview",
                            className="nav-link"
                        ),
                        dbc.NavLink(
                            [
                                html.I(className="fas fa-cogs", style={"margin-right": "12px", "width": "16px"}),
                                "分配优化"
                            ],
                            href="/optimization", 
                            active="exact", 
                            id="nav-optimization",
                            className="nav-link"
                        ),
                        dbc.NavLink(
                            [
                                html.I(className="fas fa-list-alt", style={"margin-right": "12px", "width": "16px"}),
                                "分配结果"
                            ],
                            href="/results", 
                            active="exact", 
                            id="nav-results",
                            className="nav-link"
                        ),
                        dbc.NavLink(
                            [
                                html.I(className="fas fa-search", style={"margin-right": "12px", "width": "16px"}),
                                "决策分析"
                            ],
                            href="/explanation", 
                            active="exact", 
                            id="nav-explanation",
                            className="nav-link"
                        ),
                        dbc.NavLink(
                            [
                                html.I(className="fas fa-chart-line", style={"margin-right": "12px", "width": "16px"}),
                                "统计分析"
                            ],
                            href="/statistics", 
                            active="exact", 
                            id="nav-statistics",
                            className="nav-link"
                        ),
                ],
                vertical=True,
                    pills=False,
                    style={"padding": "0"}
                ),
                style={"padding": "24px"},
            ),
            # 系统状态指示器
            html.Div(
                id="system-status",
                style={
                    "position": "absolute",
                    "bottom": "24px",
                    "left": "24px",
                    "right": "24px",
                }
            ),
        ],
        className="apple-sidebar",
    )

    # 完整布局
    return html.Div(
        [
            dcc.Location(id="url"),
            # 存储组件
            dcc.Store(id="session-id"),
            dcc.Store(id="allocation-results-data"),
            dcc.Store(id="optimization-progress-data"),
            dcc.Store(id="system-status-data"),
            # 侧边栏
            sidebar,
            # 主内容区
            html.Div(
                id="page-content", 
                className="apple-content apple-fade-in"
            ),
            # 用于触发数据概览页面更新的定时器
            dcc.Interval(id="overview-interval", interval=10 * 1000, disabled=False),
            # 用于触发优化状态更新的定时器
            dcc.Interval(
                id="optimization-interval",
                interval=500,  # 0.5秒更新一次
                n_intervals=0,
                disabled=True,
            ),
        ],
        style={"fontFamily": "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', Helvetica, Arial, sans-serif"}
    )


app.layout = create_layout()


# --- 核心回调与逻辑 ---
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page_content(pathname):
    """根据URL渲染主页面内容"""
    pages = {
        "/": create_overview_page,
        "/optimization": create_optimization_page,
        "/results": create_results_page,
        "/explanation": create_explanation_page,
        "/statistics": create_statistics_page,
    }
    return pages.get(pathname, create_overview_page)()


@app.callback(
    Output("session-id", "data"),
    Input("url", "pathname"),
    State("session-id", "data"),
)
def assign_session_id(pathname, existing_session_id):
    """为首次访问的用户分配一个唯一的会话ID"""
    if existing_session_id is None:
        new_session_id = str(uuid.uuid4())
        logger.info(f"创建新的会话，ID: {new_session_id}")
        return {"session_id": new_session_id}
    return dash.no_update


@app.callback(
    Output("system-status-data", "data"),
    Input("url", "pathname"),
)
def initialize_status_data(pathname):
    """应用加载时，初始化系统状态数据存入Store"""
    if student_data is not None:
        return {"student_count": len(student_data)}
    return dash.no_update


# --- 后台优化线程 ---
def start_optimization_thread(
    session_id,
    male_6,
    female_6,
    male_4,
    female_4,
    population_size,
    generations,
    elite_size,
    tournament_size,
    mutation_rate,
    crossover_rate,
    prioritize_class,
):
    """启动优化线程，根据给定的参数运行遗传算法"""
    
    # 初始化或重置会话状态，清理旧结果
    optimization_threads[session_id] = {
        "running": True,
        "progress": 0,
        "message": "正在初始化优化器...",
        "progress_data": {"generations": [], "fitness_values": []},  # 初始化为正确的字典结构
        "results": None,  # 确保清空旧结果
        "stop_requested": False,
    }
    logger.info(f"为会话 {session_id} 初始化优化状态。")

    def optimize():
        """优化的核心逻辑，在独立线程中运行"""
        try:
            # 进度回调函数
            def progress_callback(generation, fitness, message):
                if session_id in optimization_threads:
                    status = optimization_threads[session_id]
                    if status.get("stop_requested", False):
                        raise InterruptedError("用户请求停止优化")
                        
                    status["progress"] = (generation / generations) * 95  # 为后处理留5%
                    status["message"] = f"第{generation}代, 最佳适应度: {fitness:.4f}"
                    
                    # 更新进度数据用于图表显示
                    progress_data = status.get("progress_data")
                    if progress_data is None:
                        progress_data = {"generations": [], "fitness_values": []}
                        status["progress_data"] = progress_data
                    
                    progress_data["generations"].append(generation)
                    progress_data["fitness_values"].append(fitness)

            logger.info(f"会话 {session_id}: 开始优化...")
            
            # 更新状态
            optimization_threads[session_id]["message"] = "初始化优化器..."
            
            # 创建配置对象
            config = AllocationConfig(
                population_size=population_size,
                generations=generations,
                elite_size=elite_size,
                tournament_size=tournament_size,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
            )
            
            # 创建宿舍配置字典
            room_config = {
                "male_6": male_6,
                "female_6": female_6,
                "male_4": male_4,
                "female_4": female_4,
            }

            # 创建并运行优化器
            optimizer = AllocationOptimizer(
                student_df=student_data.copy(),
                preprocessor=preprocessor,
                model=compatibility_model,
                room_config=room_config,
                config=config,
                prioritize_class=prioritize_class,
            )

            # 运行优化
            population, best_solution = optimizer.run(
                progress_callback=progress_callback
            )
            
            logger.info(f"会话 {session_id}: 优化完成。最佳适应度: {best_solution.fitness}")

            # 处理结果
            results_df = optimizer.get_allocation_dataframe(best_solution)
            
            # 计算可解释性
            explainer = AllocationExplainer(
                model=compatibility_model,
                preprocessor=preprocessor,
                student_df=student_data,
                allocation_df=results_df,
            )
            feature_importance_df = explainer.get_feature_importance()
            allocation_metrics = explainer.get_allocation_metrics()

            # 为dcc.Store准备JSON可序列化的结果
            try:
                # 增加对 fitness 属性的健壮性检查
                fitness_value = 0.0
                if hasattr(best_solution.fitness, 'values'):
                    # 兼容DEAP多目标优化的 Fitness 对象
                    fitness_value = best_solution.fitness.values[0]
                else:
                    # 兼容直接返回数值（如 int, float, numpy.float32）的情况
                    fitness_value = best_solution.fitness

                # 计算额外的统计指标
                allocation_df = pd.DataFrame(results_df.to_dict("records"))
                
                # 计算最大兼容度
                max_compatibility = 0
                if allocation_metrics.get("compatibilities"):
                    max_compatibility = max(allocation_metrics["compatibilities"])
                
                # 计算性别平衡度
                gender_balance_scores = []
                for room_id, group in allocation_df.groupby("RoomID"):
                    room_students = group["StudentID"].tolist()
                    genders = []
                    for sid in room_students:
                        student_row = student_data[student_data["StudentID"] == sid]
                        if not student_row.empty:
                            genders.append(student_row.iloc[0]["Sex"])
                    
                    if genders:
                        male_count = genders.count("男")
                        female_count = genders.count("女")
                        total = len(genders)
                        balance_score = 1 - abs(male_count - female_count) / total if total > 0 else 0
                        gender_balance_scores.append(balance_score)
                
                avg_gender_balance = np.mean(gender_balance_scores) if gender_balance_scores else 0.5
                
                # 计算分配覆盖率
                total_students_in_data = len(student_data)
                allocated_students = len(allocation_df)
                coverage_rate = allocated_students / total_students_in_data if total_students_in_data > 0 else 0

                results_data = {
                    "summary": {
                        "total_students": len(results_df),
                        "total_rooms": results_df["RoomID"].nunique(),
                        "fitness": fitness_value,
                        "mean_compatibility": allocation_metrics.get("mean_compatibility", 0),
                        "max_compatibility": max_compatibility,
                        "gender_balance": avg_gender_balance,
                        "coverage_rate": coverage_rate,
                    },
                    "details": results_df.to_dict("records"),
                    "allocation": allocation_df.to_dict("records"),
                    "explanation": {
                        "feature_importance": feature_importance_df.to_dict("records"),
                        "metrics": allocation_metrics,
                    },
                }

                # --- 核心修复：彻底净化数据类型 ---
                # 使用json序列化再反序列化的方法，强制将所有numpy类型转为python原生类型
                results_json_str = json.dumps(results_data, cls=NumpyEncoder)
                results_json = json.loads(results_json_str)
                
                optimization_threads[session_id]["results"] = results_json
                optimization_threads[session_id]["progress"] = 100
                optimization_threads[session_id]["message"] = "优化任务成功完成！"
                logger.info(f"会话 {session_id} 的优化完成。")

            except Exception as e:
                logger.error(f"为会话 {session_id} 准备或序列化结果时发生错误: {e}", exc_info=True)
                optimization_threads[session_id]["message"] = f"处理优化结果时出现严重错误，无法显示。请检查日志。"

        except InterruptedError as e:
            logger.warning(f"会话 {session_id} 的优化被用户中断。")
            current_status = optimization_threads.get(session_id, {})
            current_status["message"] = f"任务已停止: {e}"
        except Exception as e:
            logger.error(f"会话 {session_id} 的优化线程出现错误: {e}", exc_info=True)
            current_status = optimization_threads.get(session_id, {})
            current_status["message"] = f"出现严重错误: {e}"
        finally:
            current_status = optimization_threads.get(session_id, {})
            current_status["running"] = False
            # 如果没有出错，确保进度是100%
            if "error" not in current_status.get("message", "").lower():
                current_status["progress"] = 100

    # 创建并启动线程
    thread = threading.Thread(target=optimize, name=f"OptimizationThread-{session_id}")
    thread.daemon = True  # 允许主程序退出，即使线程仍在运行
    optimization_threads[session_id]["thread"] = thread
    thread.start()
    logger.info(f"为会话 {session_id} 启动了优化线程。")


# --- 页面创建函数 ---
def create_overview_page():
    """创建数据概览页面的布局"""
    if student_data is None:
        return html.Div(
            [
                html.H4("系统数据概览", className="mb-4"),
                html.Div("错误: 学生数据未能成功加载，无法显示概览。请检查 Data.xlsx 文件是否存在。", className="alert alert-danger")
            ]
        )
    return html.Div(
        [
            html.H4("系统数据概览", className="mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("学生性别分布"),
                                dbc.CardBody(dcc.Graph(id="overview-gender-chart")),
                            ]
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("学生班级分布"),
                                dbc.CardBody(dcc.Graph(id="overview-class-chart")),
                            ]
                        ),
                        width=8,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("MBTI人格分布"),
                                dbc.CardBody(dcc.Graph(id="overview-mbti-chart")),
                            ]
                        ),
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    dbc.Row(
                                        [
                                            dbc.Col(html.H4("学生数据预览", className="card-title mb-0"), width="auto"),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="overview-page-size-dropdown",
                                                    options=[
                                                        {"label": "10 条/页", "value": 10},
                                                        {"label": "20 条/页", "value": 20},
                                                        {"label": "50 条/页", "value": 50},
                                                        {"label": "100 条/页", "value": 100},
                                                    ],
                                                    value=10, # 默认值
                                                    clearable=False,
                                                    style={"width": "150px"}
                                                ),
                                                width="auto",
                                                className="ms-auto"
                                            )
                                        ],
                                        align="center",
                                        justify="between"
                                    )
                                ),
                                dbc.CardBody(
                                    dash_table.DataTable(
                                        id="student-data-table",
                                        columns=[
                                            {"name": "学号", "id": "StudentID"},
                                            {"name": "姓名", "id": "Name"},
                                            {"name": "班级", "id": "Class"},
                                            {"name": "性别", "id": "Sex"},
                                            {"name": "年龄", "id": "Age"},
                                            {"name": "省份", "id": "Province"},
                                            {"name": "高考分数", "id": "Gaokao"},
                                            {"name": "类型", "id": "Type"},
                                            {"name": "MBTI", "id": "MBTI"},
                                            {"name": "睡觉时间", "id": "Sleep"},
                                            {"name": "兴趣爱好", "id": "Hobby"},
                                        ],
                                        data=student_data.to_dict("records"),
                                        page_size=10,
                                        style_cell={"textAlign": "left"},
                                        style_header={
                                            "backgroundColor": "lightgrey",
                                            "fontWeight": "bold",
                                        },
                                        filter_action="native",
                                        sort_action="native",
                                    )
                                ),
                            ]
                        ),
                        width=12,
                    )
                ],
                className="mt-4",
            ),
        ]
    )


def create_optimization_page():
    """创建并返回分配优化页面的布局 - Apple Design风格"""
    return html.Div(
        [
            html.H1("⚙️ 分配系统", className="apple-page-title"),
            
            # 第一行：宿舍配置和算法参数
            html.Div(
                [
                    # 宿舍容量设置
                    html.Div(
                        [
                            html.Div("🏠 宿舍容量设置", className="apple-card-header"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("6人间男生宿舍数量", className="apple-input-label"),
                                            html.Div(
                                                [
                                                    html.Div("6人间男生宿舍数量", className="apple-input-group-text"),
                                                    dcc.Input(
                                                        id="male-6-rooms", 
                                                        type="number", 
                                                        value=21, 
                                                        min=0, 
                                                        step=1,
                                                        className="apple-input"
                                                    ),
                                                ],
                                                className="apple-input-group-horizontal"
                                            ),
                                        ],
                                        className="apple-input-group"
                                    ),
                                    html.Div(
                                        [
                                            html.Label("6人间女生宿舍数量", className="apple-input-label"),
                                            html.Div(
                                                [
                                                    html.Div("6人间女生宿舍数量", className="apple-input-group-text"),
                                                    dcc.Input(
                                                        id="female-6-rooms", 
                                                        type="number", 
                                                        value=30, 
                                                        min=0, 
                                                        step=1,
                                                        className="apple-input"
                                                    ),
                                                ],
                                                className="apple-input-group-horizontal"
                                            ),
                                        ],
                                        className="apple-input-group"
                                    ),
                                    html.Div(
                                        [
                                            html.Label("4人间男生宿舍数量", className="apple-input-label"),
                                            html.Div(
                                                [
                                                    html.Div("4人间男生宿舍数量", className="apple-input-group-text"),
                                                    dcc.Input(
                                                        id="male-4-rooms", 
                                                        type="number", 
                                                        value=0, 
                                                        min=0, 
                                                        step=1,
                                                        className="apple-input"
                                                    ),
                                                ],
                                                className="apple-input-group-horizontal"
                                            ),
                                        ],
                                        className="apple-input-group"
                                    ),
                                    html.Div(
                                        [
                                            html.Label("4人间女生宿舍数量", className="apple-input-label"),
                                            html.Div(
                                                [
                                                    html.Div("4人间女生宿舍数量", className="apple-input-group-text"),
                                                    dcc.Input(
                                                        id="female-4-rooms", 
                                                        type="number", 
                                                        value=0, 
                                                        min=0, 
                                                        step=1,
                                                        className="apple-input"
                                                    ),
                                                ],
                                                className="apple-input-group-horizontal"
                                            ),
                                        ],
                                        className="apple-input-group"
                                    ),
                                    html.Div(id="bed-capacity-info", style={"margin-top": "16px", "color": "var(--apple-gray)"}),
                                ],
                                className="apple-card-body"
                            ),
                        ],
                        className="apple-card"
                    ),
                    
                    # 遗传算法参数设置
                    html.Div(
                        [
                            html.Div("🧬 遗传算法参数", className="apple-card-header"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("种群大小", className="apple-input-label"),
                                            html.Div(
                                                [
                                                    html.Div("种群大小", className="apple-input-group-text"),
                                                    dcc.Input(
                                                        id="population-size", 
                                                        type="number", 
                                                        value=100, 
                                                        min=20, 
                                                        max=500,
                                                        className="apple-input"
                                                    ),
                                                ],
                                                className="apple-input-group-horizontal"
                                            ),
                                        ],
                                        className="apple-input-group"
                                    ),
                                    html.Div(
                                        [
                                            html.Label("迭代代数", className="apple-input-label"),
                                            html.Div(
                                                [
                                                    html.Div("迭代代数", className="apple-input-group-text"),
                                                    dcc.Input(
                                                        id="generations", 
                                                        type="number", 
                                                        value=200, 
                                                        min=50, 
                                                        max=1000,
                                                        className="apple-input"
                                                    ),
                                                ],
                                                className="apple-input-group-horizontal"
                                            ),
                                        ],
                                        className="apple-input-group"
                                    ),
                                        html.Div(
                                        [
                                            html.Label("精英个体数", className="apple-input-label"),
                                            html.Div(
                                                [
                                                    html.Div("精英个体数", className="apple-input-group-text"),
                                                    dcc.Input(
                                                        id="elite-size", 
                                                        type="number", 
                                                        value=10, 
                                                        min=1, 
                                                        max=50,
                                                        className="apple-input"
                                                    ),
                                                ],
                                                className="apple-input-group-horizontal"
                                            ),
                                        ],
                                        className="apple-input-group"
                                    ),
                                    html.Div(
                                        [
                                            html.Label("锦标赛大小", className="apple-input-label"),
                                            html.Div(
                                                [
                                                    html.Div("锦标赛大小", className="apple-input-group-text"),
                                                    dcc.Input(
                                                        id="tournament-size", 
                                                        type="number", 
                                                        value=5, 
                                                        min=2, 
                                                        max=20,
                                                        className="apple-input"
                                                    ),
                                                ],
                                                className="apple-input-group-horizontal"
                                            ),
                                        ],
                                        className="apple-input-group"
                                    ),
                                    html.Div(
                                        [
                                            html.Label("变异率", className="apple-input-label"),
                                            html.Div(
                                                [
                                                    html.Div("变异率", className="apple-input-group-text"),
                                                    dcc.Input(
                                                        id="mutation-rate", 
                                                        type="number", 
                                                        value=0.1, 
                                                        min=0.01, 
                                                        max=0.5, 
                                                        step=0.01,
                                                        className="apple-input"
                                                    ),
                                                ],
                                                className="apple-input-group-horizontal"
                                            ),
                                        ],
                                        className="apple-input-group"
                                    ),
                                    html.Div(
                                        [
                                            html.Label("交叉率", className="apple-input-label"),
                                            html.Div(
                                                [
                                                    html.Div("交叉率", className="apple-input-group-text"),
                                                    dcc.Input(
                                                        id="crossover-rate", 
                                                        type="number", 
                                                        value=0.8, 
                                                        min=0.1, 
                                                        max=1.0, 
                                                        step=0.01,
                                                        className="apple-input"
                                                    ),
                                                ],
                                                className="apple-input-group-horizontal"
                                            ),
                                        ],
                                        className="apple-input-group"
                                    ),
                                    # 同班优先开关 - 修复布局问题
                                    html.Div(
                                        [
                                            html.Label("特殊设置", className="apple-input-label"),
                                            html.Div(
                                                [
                                                    dbc.Switch(
                                                        id="prioritize-class-switch",
                                                        label="优先将同班学生分配在同一宿舍",
                                                        value=False,
                                                        style={"margin-left": "8px"}
                                                    ),
                                                ],
                                                className="apple-switch-container",
                                                style={"padding-left": "8px", "margin-top": "8px"}
                                            ),
                                        ],
                                        className="apple-input-group",
                                        style={"margin-top": "16px"}
                                    ),
                            ],
                                className="apple-card-body"
                        ),
                        ],
                        className="apple-card"
                    ),
                ],
                className="apple-grid apple-grid-2"
            ),
            
            # 控制按钮
            html.Div(
                [
                    html.Button(
                        [
                            html.I(className="fas fa-play", style={"margin-right": "8px"}),
                            "开始分配"
                        ], 
                        id="start-optimization", 
                        className="apple-btn apple-btn-primary", 
                        n_clicks=0, 
                        disabled=False,
                        style={"margin-right": "16px"}
                    ),
                    html.Button(
                        [
                            html.I(className="fas fa-stop", style={"margin-right": "8px"}),
                            "停止分配"
                        ], 
                        id="stop-optimization", 
                        className="apple-btn apple-btn-danger", 
                        n_clicks=0, 
                        disabled=True
                    ),
                ],
                className="text-center mt-4",
            ),
            
            # 优化状态和进度图表
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("📊 分配进度", className="apple-card-header"),
                            html.Div(
                                [
                                    html.Div(id="optimization-status", children="系统就绪，等待分配任务...", style={"margin-bottom": "24px"}),
                                    dcc.Graph(
                                        id="optimization-progress",
                                        config={'displayModeBar': False}
                                    ),
                                ],
                                className="apple-card-body"
                            ),
                        ],
                        className="apple-card"
                    )
                ],
                className="apple-grid",
                style={"margin-top": "32px"}
            ),
        ]
    )


def create_results_page():
    """创建结果页面 - Apple Design风格"""
    return html.Div(
        [
            html.H1("📋 分配结果", className="apple-page-title"),
            
            # 分配汇总信息
            html.Div(id="allocation-summary", style={"margin-bottom": "32px"}),
            
            # 分配详情表格
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("宿舍分配详情", style={"margin": "0", "color": "var(--apple-text-primary)"}),
                                    dcc.Dropdown(
                                        id="results-page-size-dropdown",
                                        options=[
                                            {"label": "10 条/页", "value": 10},
                                            {"label": "20 条/页", "value": 20},
                                            {"label": "50 条/页", "value": 50},
                                            {"label": "100 条/页", "value": 100},
                                        ],
                                        value=10,
                                        clearable=False,
                                        style={"width": "150px"}
                                    ),
                                ],
                                className="apple-card-header",
                                style={
                                    "display": "flex", 
                                    "justify-content": "space-between", 
                                    "align-items": "center"
                                }
                            ),
                            html.Div(id="room-details", className="apple-card-body"),
                        ],
                        className="apple-card"
                    )
                ],
                className="apple-grid",
                style={"margin-bottom": "32px"}
            ),
            
            # 宿舍兼容度排名图表
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("📊 宿舍兼容度排名", className="apple-card-header"),
                            html.Div(
                                dcc.Graph(
                                    id="room-compatibility-barchart",
                                    config={'displayModeBar': False}
                                ), 
                                className="apple-card-body"
                            ),
                        ],
                        className="apple-card"
                    )
                ],
                className="apple-grid",
                style={"margin-bottom": "32px"}
            ),
            
            # 导出按钮
            html.Div(
                [
                    html.Button(
                        [
                            html.I(className="fas fa-download", style={"margin-right": "8px"}),
                            "导出Excel"
                        ], 
                        id="export-excel", 
                        className="apple-btn apple-btn-primary",
                        style={"margin-right": "16px"}
                    ),
                    html.Button(
                        [
                            html.I(className="fas fa-file-pdf", style={"margin-right": "8px"}),
                            "导出PDF报告"
                        ], 
                        id="export-pdf", 
                        className="apple-btn apple-btn-secondary", 
                        disabled=True
                    ),
                    dbc.Tooltip("PDF导出功能正在开发中...", target="export-pdf"),
                ],
                style={"text-align": "center"},
            ),
            dcc.Download(id="download-excel"),
        ]
    )


def create_explanation_page():
    """创建解释页面 - Apple Design风格"""
    return html.Div(
        [
            html.H1("🔍 决策分析", className="apple-page-title"),
            
            # 特征重要性分析
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("🎯 特征重要性分析", className="apple-card-header"),
                            html.Div(
                                dcc.Graph(
                                    id="feature-importance-plot",
                                    config={'displayModeBar': False}
                                ), 
                                className="apple-card-body"
                            ),
                        ],
                        className="apple-card"
                    )
                ],
                className="apple-grid",
                style={"margin-bottom": "32px"}
            ),
            
            # 宿舍兼容性分析
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("🏠 宿舍兼容性分析", className="apple-card-header"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("选择宿舍", className="apple-input-label"),
                                            dcc.Dropdown(
                                                id="room-selector", 
                                                options=[], 
                                                value=None,
                                                placeholder="请选择要分析的宿舍",
                                                style={"margin-bottom": "24px"}
                                            ),
                                        ],
                                        className="apple-input-group"
                                    ),
                        html.Div(id="room-compatibility-analysis"),
                                ], 
                                className="apple-card-body"
                            ),
                        ],
                        className="apple-card"
                    )
                ],
                className="apple-grid"
            ),
        ]
    )


def create_statistics_page():
    """创建统计分析页面 - Apple Design风格"""
    return html.Div(
        [
            html.H1("📈 统计分析", className="apple-page-title"),
            
            # 第一行：分配质量统计和MBTI多样性
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("📊 分配质量统计", className="apple-card-header"),
                            html.Div(
                                dcc.Graph(
                                    id="allocation-quality-stats",
                                    config={'displayModeBar': False}
                                ), 
                                className="apple-card-body"
                            ),
                        ],
                        className="apple-card",
                        style={"width": "48%", "display": "inline-block", "margin-right": "4%"}
                    ),
                    html.Div(
                        [
                            html.Div("🧠 宿舍MBTI多样性", className="apple-card-header"),
                            html.Div(
                                dcc.Graph(
                                    id="mbti-diversity-chart",
                                    config={'displayModeBar': False}
                                ), 
                                className="apple-card-body"
                            ),
                        ],
                        className="apple-card",
                        style={"width": "48%", "display": "inline-block"}
                    )
                ],
                style={"margin-bottom": "32px"}
            ),
            
            # 第二行：地域分布和特征相关性
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("🌍 地域分布分析", className="apple-card-header"),
                            html.Div(
                                dcc.Graph(
                                    id="province-distribution",
                                    config={'displayModeBar': False}
                                ), 
                                className="apple-card-body"
                            ),
                        ],
                        className="apple-card",
                        style={"width": "48%", "display": "inline-block", "margin-right": "4%"}
                    ),
                    html.Div(
                        [
                            html.Div("🔗 学生特征相关性", className="apple-card-header"),
                            html.Div(
                                dcc.Graph(
                                    id="feature-correlation",
                                    config={'displayModeBar': False}
                                ), 
                                className="apple-card-body"
                            ),
                        ],
                        className="apple-card",
                        style={"width": "48%", "display": "inline-block"}
                    )
                ]
            ),
        ]
    )

# --- 优化控制与状态更新回调 ---
@app.callback(
    [
        Output("start-optimization", "disabled"),
        Output("stop-optimization", "disabled"),
        Output("optimization-interval", "disabled"),
        Output("allocation-results-data", "clear_data"),
        # 控制所有输入控件
        Output("male-6-rooms", "disabled"),
        Output("female-6-rooms", "disabled"),
        Output("male-4-rooms", "disabled"),
        Output("female-4-rooms", "disabled"),
        Output("population-size", "disabled"),
        Output("generations", "disabled"),
        Output("elite-size", "disabled"),
        Output("tournament-size", "disabled"),
        Output("mutation-rate", "disabled"),
        Output("crossover-rate", "disabled"),
        Output("prioritize-class-switch", "disabled"),
    ],
    [
        Input("start-optimization", "n_clicks"),
        Input("stop-optimization", "n_clicks"),
        Input("optimization-interval", "n_intervals"), # 添加定时器作为触发器
    ],
    [
        State("session-id", "data"),
        State("male-6-rooms", "value"),
        State("female-6-rooms", "value"),
        State("male-4-rooms", "value"),
        State("female-4-rooms", "value"),
        State("population-size", "value"),
        State("generations", "value"),
        State("elite-size", "value"),
        State("tournament-size", "value"),
        State("mutation-rate", "value"),
        State("crossover-rate", "value"),
        State("prioritize-class-switch", "value"),
    ],
    prevent_initial_call=True,
)
def control_optimization(
    start_clicks,
    stop_clicks,
    n_intervals,
    session_id_data,
    m6,
    f6,
    m4,
    f4,
    pop,
    gen,
    elite,
    tour,
    mut,
    cross,
    prioritize_class,
):
    """根据按钮点击和后台状态来统一控制优化过程和UI元素"""
    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    session_id = session_id_data["session_id"]
    
    # --- 处理按钮点击 ---
    clear_data = dash.no_update
    if triggered_id == "start-optimization" and start_clicks > 0:
        logger.info(f"[{session_id}] 收到启动分配请求。")
        # 仅当没有线程在运行时才启动
        if not optimization_threads.get(session_id, {}).get("running", False):
            start_optimization_thread(
                session_id, m6, f6, m4, f4, pop, gen, elite, tour, mut, cross, prioritize_class
            )
            clear_data = True
    
    elif triggered_id == "stop-optimization" and stop_clicks > 0:
        logger.info(f"[{session_id}] 收到停止分配请求。")
        if session_id in optimization_threads:
            optimization_threads[session_id]["stop_requested"] = True
            optimization_threads[session_id]["message"] = "正在停止分配..."

    # --- 根据线程状态更新UI ---
    is_running = optimization_threads.get(session_id, {}).get("running", False)

    # 禁用所有输入控件
    inputs_disabled = is_running
    
    # 准备返回状态
    outputs = [
        is_running,  # start-optimization disabled
        not is_running,  # stop-optimization disabled
        not is_running,  # optimization-interval disabled
        clear_data,
    ]
    # 添加11个输入控件的状态 (10个旧的 + 1个新的)
    outputs.extend([inputs_disabled] * 11)
    
    return outputs


@app.callback(
    Output("optimization-status", "children"),
    Input("optimization-interval", "n_intervals"),
    State("session-id", "data"),
)
def update_optimization_status(n, session_id_data):
    """从会话存储中轮询并更新优化状态UI - Apple Design风格"""
    if not session_id_data or session_id_data["session_id"] not in optimization_threads:
        return html.Div("✨ 系统就绪，等待分配任务", className="apple-status apple-status-success")

    status = optimization_threads[session_id_data["session_id"]]
    progress = status.get("progress", 0)
    message = status.get("message", "等待中...")

    if status.get("running", False):
        return html.Div([
            html.Div(
                f"{progress:.1f}%",
                style={
                    "font-size": "18px", 
                    "font-weight": "600", 
                    "color": "var(--apple-blue)",
                    "margin-bottom": "8px",
                    "text-align": "center"
                }
            ),
            html.Div(
                [
                    html.Div(
                        style={
                            "width": f"{progress}%",
                            "height": "8px",
                            "background": "linear-gradient(90deg, var(--apple-blue) 0%, var(--apple-purple) 100%)",
                            "border-radius": "4px",
                            "transition": "width 0.3s ease"
                        }
                    )
                ],
                style={
                    "width": "100%",
                    "height": "8px",
                    "background-color": "var(--apple-light-gray)",
                    "border-radius": "4px",
                    "margin-bottom": "12px",
                    "overflow": "hidden"
                }
            ),
            html.Div(
                f"🔄 {message}",
                style={
                    "font-size": "14px", 
                    "color": "var(--apple-text-secondary)",
                    "text-align": "center"
                }
            )
        ])
    else:
        if "完成" in message:
            return html.Div(f"✅ {message}", className="apple-status apple-status-success")
        else:
            return html.Div(f"⚠️ {message}", className="apple-status apple-status-error")


# --- 数据轮询与Store更新回调 ---
@app.callback(
    Output("optimization-progress-data", "data"),
    Input("optimization-interval", "n_intervals"),
    State("session-id", "data"),
)
def update_progress_data_from_thread(n, session_id_data):
    """定期从后台线程状态轮询进度数据并更新到Store中"""
    if session_id_data and session_id_data["session_id"] in optimization_threads:
        progress_data = optimization_threads[session_id_data["session_id"]].get("progress_data")
        if progress_data:
            return progress_data.copy()
    return dash.no_update


@app.callback(
    Output("allocation-results-data", "data"),
    Input("optimization-interval", "n_intervals"),
    State("session-id", "data"),
)
def update_results_data_from_thread(n, session_id_data):
    """优化完成后，从线程状态获取最终结果并更新到Store中"""
    if session_id_data and session_id_data["session_id"] in optimization_threads:
        status = optimization_threads[session_id_data["session_id"]]
        # 修正：采用更健壮的 get/set-None 模式代替 pop
        if not status.get("running") and status.get("results"):
            results = status["results"]
            # 不清空results，允许重复显示，直到新的优化开始
            # status["results"] = None  # 清空，防止重复发送
            logger.info(f"[{session_id_data['session_id']}] 成功从线程获取分配结果并发送至前端。")
            return results
    return dash.no_update


# --- 页面UI更新回调 ---
@app.callback(
    Output("optimization-progress", "figure"),
    Input("optimization-progress-data", "data")
)
def update_optimization_progress_chart(progress_data):
    """根据Store中的数据更新优化进度图表"""
    if not progress_data or not progress_data.get("generations"):
        return go.Figure().update_layout(title="分配进度", template="plotly_white", height=400)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=progress_data["generations"],
        y=progress_data["fitness_values"],
        mode='lines+markers', name='最佳适应度'
    ))
    
    # 添加最高值横线
    if progress_data["fitness_values"]:
        max_fitness = max(progress_data["fitness_values"])
        fig.add_hline(
            y=max_fitness, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"最高值: {max_fitness:.4f}",
            annotation_position="bottom right"
        )
    
    fig.update_layout(title="分配进度 - 适应度变化", xaxis_title="迭代代数", yaxis_title="适应度得分", template="plotly_white", height=400)
    return fig


@app.callback(
    Output("allocation-summary", "children"),
    Input("allocation-results-data", "data")
)
def update_allocation_summary(results_data):
    """更新分配结果摘要 - Apple Design风格"""
    try:
        if not results_data or "summary" not in results_data:
            return html.Div(
                [
                    html.I(className="fas fa-info-circle", style={"margin-right": "8px"}),
                    "暂无分配结果，请先进行分配。"
                ], 
                className="apple-alert apple-alert-info"
            )
        
        summary = results_data["summary"]
        return html.Div(
            [
                # 学生总数
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("👥 学生总数", className="apple-card-header"),
                                html.Div(
                                    [
                                        html.Div(
                                            f"{summary.get('total_students', 'N/A')}", 
                                            style={
                                                "font-size": "32px", 
                                                "font-weight": "700", 
                                                "color": "var(--apple-blue)",
                                                "text-align": "center"
                                            }
                                        ),
                                        html.Div(
                                            "已分配学生",
                                            style={
                                                "color": "var(--apple-text-secondary)",
                                                "text-align": "center",
                                                "margin-top": "8px"
                                            }
                                        )
                                    ],
                                    className="apple-card-body"
                                ),
                            ],
                            className="apple-card"
                        )
                    ],
                    style={"grid-column": "1"}
                ),
                
                # 宿舍总数
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("🏠 宿舍总数", className="apple-card-header"),
                                html.Div(
                                    [
                                        html.Div(
                                            f"{summary.get('total_rooms', 'N/A')}", 
                                            style={
                                                "font-size": "32px", 
                                                "font-weight": "700", 
                                                "color": "var(--apple-green)",
                                                "text-align": "center"
                                            }
                                        ),
                                        html.Div(
                                            "已使用宿舍",
                                            style={
                                                "color": "var(--apple-text-secondary)",
                                                "text-align": "center",
                                                "margin-top": "8px"
                                            }
                                        )
                                    ],
                                    className="apple-card-body"
                                ),
                            ],
                            className="apple-card"
                        )
                    ],
                    style={"grid-column": "2"}
                ),
                
                # 适应度得分
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("🎯 适应度得分", className="apple-card-header"),
                                html.Div(
                                    [
                                        html.Div(
                                            f"{summary.get('fitness', 0):.4f}", 
                                            style={
                                                "font-size": "32px", 
                                                "font-weight": "700", 
                                                "color": "var(--apple-orange)",
                                                "text-align": "center"
                                            }
                                        ),
                                        html.Div(
                                            "分配评分",
                                            style={
                                                "color": "var(--apple-text-secondary)",
                                                "text-align": "center",
                                                "margin-top": "8px"
                                            }
                                        )
                                    ],
                                    className="apple-card-body"
                                ),
                            ],
                            className="apple-card"
                        )
                    ],
                    style={"grid-column": "3"}
                ),
                
                # 平均兼容度
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("💫 平均兼容度", className="apple-card-header"),
                                html.Div(
                                    [
                                        html.Div(
                                            f"{summary.get('mean_compatibility', 0):.4f}", 
                                            style={
                                                "font-size": "32px", 
                                                "font-weight": "700", 
                                                "color": "var(--apple-purple)",
                                                "text-align": "center"
                                            }
                                        ),
                                        html.Div(
                                            "兼容性指数",
                                            style={
                                                "color": "var(--apple-text-secondary)",
                                                "text-align": "center",
                                                "margin-top": "8px"
                                            }
                                        )
                                    ],
                                    className="apple-card-body"
                                ),
                            ],
                            className="apple-card"
                        )
                    ],
                    style={"grid-column": "4"}
                ),
            ],
            className="apple-grid apple-grid-3",
        )
    except Exception as e:
        logger.error(f"渲染摘要时发生错误: {e}", exc_info=True)
        return html.Div(
            [
                html.I(className="fas fa-exclamation-triangle", style={"margin-right": "8px"}),
                f"渲染分配摘要时出错: {e}"
            ], 
            className="apple-alert apple-alert-danger"
        )


@app.callback(
    Output("room-details", "children"),
    [Input("allocation-results-data", "data"),
     Input("results-page-size-dropdown", "value")] # 添加分页控件作为输入
)
def update_room_details(results_data, page_size):
    """更新宿舍的分配详情"""
    try:
        # 添加详细的调试信息
        if not results_data:
            logger.info("room-details回调: results_data为空")
            return html.Div(
                [
                    html.I(className="fas fa-info-circle", style={"margin-right": "8px"}),
                    "暂无详细分配数据，请先完成分配。"
                ], 
                className="apple-alert apple-alert-info"
            )
        
        logger.info(f"room-details回调: results_data键: {list(results_data.keys()) if isinstance(results_data, dict) else '非字典类型'}")
        
        if "details" not in results_data:
            logger.warning("room-details回调: results_data中没有'details'键")
            return html.Div(
                [
                    html.I(className="fas fa-exclamation-triangle", style={"margin-right": "8px"}),
                    "分配数据格式错误，缺少详细信息。"
                ], 
                className="apple-alert apple-alert-warning"
            )

        allocation_details = results_data["details"]
        if not allocation_details:
            logger.warning("room-details回调: allocation_details为空")
            return html.Div(
                [
                    html.I(className="fas fa-exclamation-triangle", style={"margin-right": "8px"}),
                    "分配详情为空。"
                ], 
                className="apple-alert apple-alert-warning"
            )

        logger.info(f"room-details回调: 分配详情数量: {len(allocation_details)}")
        
        df = pd.DataFrame(allocation_details)
        logger.info(f"room-details回调: DataFrame形状: {df.shape}, 列: {df.columns.tolist()}")
        
        # 创建表格的函数
        def create_data_table(dataframe, table_id):
            return dash_table.DataTable(
                id=table_id,
                columns=[
                    {"name": "宿舍号", "id": "RoomID"},
                    {"name": "类型", "id": "RoomType"},
                    {"name": "学号", "id": "StudentID"},
                    {"name": "姓名", "id": "Name"},
                    {"name": "性别", "id": "Sex"},
                    {"name": "班级", "id": "Class"},
                    {"name": "MBTI", "id": "MBTI"},
                ],
                data=dataframe.to_dict("records"),
                page_size=page_size or 10, # 使用回调传入的 page_size，提供默认值
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left", 
                    "padding": "8px",
                    "fontSize": "14px",
                    "fontFamily": "SF Pro Display, -apple-system, BlinkMacSystemFont, sans-serif"
                },
                style_header={
                    "backgroundColor": "#f8f9fa",
                    "fontWeight": "600",
                    "color": "#1d1d1f",
                    "border": "1px solid #d2d2d7"
                },
                style_data={
                    "backgroundColor": "#ffffff",
                    "color": "#1d1d1f",
                    "border": "1px solid #d2d2d7"
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f8f9fa'
                    }
                ]
            )

        all_rooms_table = create_data_table(df, "all-allocation-table")
        logger.info("room-details回调: 表格创建成功")

        return html.Div(
            [
                html.H5(
                    f"分配结果详情 ({len(df)} 名学生)", 
                    className="mb-3",
                    style={
                        "color": "#1d1d1f",
                        "fontWeight": "600",
                        "fontSize": "18px"
                    }
                ),
                all_rooms_table
            ],
            className="apple-card-body"
        )
        
    except Exception as e:
        logger.error(f"渲染房间详情时发生错误: {e}", exc_info=True)
        return html.Div(
            [
                html.I(className="fas fa-exclamation-triangle", style={"margin-right": "8px"}),
                f"渲染房间详情时出错: {str(e)}"
            ], 
            className="apple-alert apple-alert-danger"
        )


@app.callback(
    Output("student-data-table", "page_size"),
    Input("overview-page-size-dropdown", "value")
)
def update_overview_table_page_size(page_size):
    """更新数据概览页表格的分页大小"""
    return page_size


@app.callback(
    Output("feature-importance-plot", "figure"),
    Input("allocation-results-data", "data"),
)
def update_feature_importance(results_data):
    """更新全局特征重要性图表"""
    try:
        if (
            not results_data
            or "explanation" not in results_data
            or "feature_importance" not in results_data["explanation"]
        ):
            return go.Figure().update_layout(
                title="全局特征重要性 (SHAP)",
                template="plotly_white",
                height=400,
            )

        importance_df = pd.DataFrame(results_data["explanation"]["feature_importance"])
        
        # 仅显示前 20 个最重要的特征
        top_features = importance_df.head(20)

        fig = px.bar(
            top_features,
            x="importance",
            y="feature",
            orientation="h",
            title="全局特征重要性 (Top 20)",
            labels={"importance": "SHAP 平均绝对值", "feature": "特征"},
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white", height=600)
        return fig
    except Exception as e:
        logger.error(f"渲染特征重要性图表时发生错误: {e}", exc_info=True)
        return go.Figure().update_layout(title=f"渲染图表时出错: {e}")


@app.callback(
    Output("room-selector", "options"),
    Input("allocation-results-data", "data")
)
def update_room_selector_options(results_data):
    try:
        if not results_data or "details" not in results_data:
            return []
        
        allocation_details = results_data["details"]
        if not isinstance(allocation_details, list) or not allocation_details:
            return []

        df = pd.DataFrame(allocation_details)
        if "RoomID" not in df.columns:
            return []

        room_ids = sorted(df["RoomID"].unique())
        return [{"label": f"宿舍 {room_id}", "value": room_id} for room_id in room_ids]
    except Exception as e:
        logger.error(f"更新宿舍选择器时发生错误: {e}", exc_info=True)
        return []


@app.callback(
    Output("room-compatibility-analysis", "children"),
    Input("room-selector", "value"),
    State("allocation-results-data", "data"),
)
def update_room_compatibility_analysis(selected_room, results_data):
    """动态计算并显示选定宿舍的兼容性分析"""
    if not selected_room:
        return html.Div("请选择一个宿舍进行分析。", className="alert alert-info")

    if not results_data or "details" not in results_data:
        return html.Div("无有效的分配结果可供分析。", className="alert alert-warning")

    try:
        # 1. 准备所需数据
        allocation_df = pd.DataFrame(results_data["details"])

        # 2. 初始化解释器
        explainer = AllocationExplainer(
            model=compatibility_model,
            preprocessor=preprocessor,
            student_df=student_data,
            allocation_df=allocation_df,
        )

        # 3. 生成并返回图表
        fig = explainer.get_room_shap_plot(selected_room)
        return dcc.Graph(figure=fig)

    except Exception as e:
        logger.error(f"为宿舍 {selected_room} 生成SHAP分析时出错: {e}", exc_info=True)
        return html.Div(f"为宿舍 {selected_room} 生成分析图表时遇到错误，请检查日志。", className="alert alert-danger")


@app.callback(
    [
        Output("allocation-quality-stats", "figure"),
        Output("mbti-diversity-chart", "figure"),
        Output("province-distribution", "figure"),
        Output("feature-correlation", "figure"),
    ],
    Input("allocation-results-data", "data"),
)
def update_statistics_charts(results_data):
    try:
        if not results_data:
            empty_figs = [go.Figure() for _ in range(4)]
            return empty_figs
        
        summary = results_data.get("summary", {})
        
        # 1. 分配质量统计 - 删除分配覆盖率和性别平衡度指标
        quality_metrics = {
            "平均兼容度": summary.get("mean_compatibility", 0),
            "最佳宿舍兼容度": summary.get("max_compatibility", 0),
        }
        
        quality_fig = go.Figure(go.Bar(
            x=list(quality_metrics.keys()),
            y=list(quality_metrics.values()),
            marker_color=["#1f77b4", "#2ca02c"],
            text=[f"{v:.3f}" for v in quality_metrics.values()],
            textposition='auto'
        ))
        quality_fig.update_layout(
            title="分配质量综合指标",
            template="plotly_white",
            yaxis_range=[0, 1],
            height=400
        )

        if student_data is None:
            empty_figs = [quality_fig] + [go.Figure().update_layout(title="数据未加载") for _ in range(3)]
            return empty_figs

        allocation_df = pd.DataFrame(results_data.get("allocation", []))
        
        # 2. MBTI多样性分析
        if len(allocation_df) > 0:
            mbti_diversity_data = []
            for room_id, group in allocation_df.groupby("RoomID"):
                room_students = group["StudentID"].tolist()
                mbtis = []
                for sid in room_students:
                    student_row = student_data[student_data["StudentID"] == sid]
                    if not student_row.empty:
                        mbti = student_row.iloc[0]["MBTI"]
                        if pd.notna(mbti) and mbti != "未知":
                            mbtis.append(mbti)
                
                if mbtis:
                    unique_mbtis = len(set(mbtis))
                    total_students = len(mbtis)
                    diversity_score = unique_mbtis / total_students if total_students > 0 else 0
                    mbti_diversity_data.append({
                        "宿舍": room_id,
                        "MBTI类型数": unique_mbtis,
                        "学生数": total_students,
                        "多样性": diversity_score
                    })
            
            if mbti_diversity_data:
                mbti_df = pd.DataFrame(mbti_diversity_data)
                mbti_fig = px.bar(
                    mbti_df,
                    x="宿舍",
                    y="多样性",
                    title="宿舍MBTI人格多样性",
                    color="多样性",
                    color_continuous_scale="viridis"
                )
                mbti_fig.update_layout(
                    template="plotly_white",
                    height=400  # 添加明确的高度设置，与其他图表保持一致
                )
            else:
                mbti_fig = go.Figure().update_layout(
                    title="无MBTI多样性数据",
                    height=400  # 确保空状态图表也有一致的高度
                )
        else:
            mbti_fig = go.Figure().update_layout(
                title="无分配数据",
                height=400  # 确保空状态图表也有一致的高度
            )

        # 3. 地域分布分析 - 按六大区域分组统计
        province_counts = student_data["Province"].value_counts()
        
        # 定义中国六大区域分组
        region_mapping = {
            # 华北地区
            "北京": "华北", "天津": "华北", "河北": "华北", "山西": "华北", "内蒙古": "华北",
            "北京市": "华北", "天津市": "华北", "河北省": "华北", "山西省": "华北", "内蒙古自治区": "华北",
            
            # 东北地区
            "辽宁": "东北", "吉林": "东北", "黑龙江": "东北",
            "辽宁省": "东北", "吉林省": "东北", "黑龙江省": "东北",
            
            # 华东地区
            "上海": "华东", "江苏": "华东", "浙江": "华东", "安徽": "华东", 
            "福建": "华东", "江西": "华东", "山东": "华东",
            "上海市": "华东", "江苏省": "华东", "浙江省": "华东", "安徽省": "华东",
            "福建省": "华东", "江西省": "华东", "山东省": "华东",
            
            # 华南地区
            "河南": "华南", "湖北": "华南", "湖南": "华南", "广东": "华南", 
            "广西": "华南", "海南": "华南", "台湾": "华南", "香港": "华南", "澳门": "华南",
            "河南省": "华南", "湖北省": "华南", "湖南省": "华南", "广东省": "华南",
            "广西壮族自治区": "华南", "海南省": "华南", "台湾省": "华南", 
            "香港特别行政区": "华南", "澳门特别行政区": "华南",
            
            # 西南地区
            "重庆": "西南", "四川": "西南", "贵州": "西南", "云南": "西南", "西藏": "西南",
            "重庆市": "西南", "四川省": "西南", "贵州省": "西南", "云南省": "西南", "西藏自治区": "西南",
            
            # 西北地区
            "陕西": "西北", "甘肃": "西北", "青海": "西北", "宁夏": "西北", "新疆": "西北",
            "陕西省": "西北", "甘肃省": "西北", "青海省": "西北", "宁夏回族自治区": "西北", "新疆维吾尔自治区": "西北"
        }
        
        # 统计各区域学生数量
        region_counts = {}
        region_details = {}  # 记录每个区域包含的省份详情
        
        for province, count in province_counts.items():
            region = region_mapping.get(province, "其他")
            if region not in region_counts:
                region_counts[region] = 0
                region_details[region] = []
            region_counts[region] += count
            region_details[region].append(f"{province}: {count}人")
        
        # 按数量从高到低排序
        sorted_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 定义区域颜色
        region_colors = {
            "华北": "#1f77b4",  # 蓝色
            "东北": "#ff7f0e",  # 橙色
            "华东": "#2ca02c",  # 绿色
            "华南": "#d62728",  # 红色
            "西南": "#9467bd",  # 紫色
            "西北": "#8c564b",  # 棕色
            "其他": "#7f7f7f"   # 灰色
        }
        
        # 准备图表数据
        regions = [item[0] for item in sorted_regions]
        counts = [item[1] for item in sorted_regions]
        colors = [region_colors.get(region, "#7f7f7f") for region in regions]
        
        # 创建垂直条形图
        province_fig = go.Figure(data=[
            go.Bar(
                x=regions,
                y=counts,
                text=counts,
                texttemplate='%{text}人',
                textposition='outside',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255,255,255,0.8)', width=1)
                ),
                hovertemplate='<b>%{x}</b><br>' +
                             '学生数量: %{y}人<br>' +
                             '<extra></extra>'
            )
        ])
        
        # 添加详细悬停信息
        hover_texts = []
        for region in regions:
            details = region_details.get(region, [])
            hover_text = f"<b>{region}地区</b><br>总计: {region_counts[region]}人<br><br>省份分布:<br>" + "<br>".join(details[:5])
            if len(details) > 5:
                hover_text += f"<br>... 等{len(details)}个省份"
            hover_texts.append(hover_text)
        
        province_fig.update_traces(
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        )
        
        # 优化图表样式
        province_fig.update_layout(
            template="plotly_white",
            title=dict(
                text="学生地域分布（按六大区域统计）",
                x=0.5,
                font=dict(size=16, color='var(--apple-text-primary)')
            ),
            xaxis=dict(
                title="地区",
                tickangle=0,
                showgrid=False
            ),
            yaxis=dict(
                title="学生数量",
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            height=450,
            margin=dict(l=60, r=40, t=60, b=80),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # 4. 特征相关性分析
        # 修复: 将numpy数组转换为带有特征名称的DataFrame以计算相关性
        processed_df = pd.DataFrame(
            preprocessor.transform(student_data),
            columns=preprocessor.get_feature_names()
        )
        correlation_matrix = processed_df.corr()
        correlation_fig = px.imshow(
            correlation_matrix,
            color_continuous_scale="RdBu_r",
            title="学生特征相关性矩阵",
        )
        correlation_fig.update_layout(template="plotly_white")

        return quality_fig, mbti_fig, province_fig, correlation_fig
        
    except Exception as e:
        logger.error(f"渲染统计图表时发生错误: {e}", exc_info=True)
        error_fig = go.Figure().update_layout(title=f"渲染图表时出错: {e}")
        error_figs = [error_fig for _ in range(4)]
        return error_figs


@app.callback(
    Output("bed-capacity-info", "children"),
    [
        Input("male-6-rooms", "value"), Input("female-6-rooms", "value"),
        Input("male-4-rooms", "value"), Input("female-4-rooms", "value"),
    ],
)
def update_bed_capacity_info(m6, f6, m4, f4):
    if not all(v is not None for v in [m6, f6, m4, f4]):
        return html.Div([
            html.Div([
                html.I(className="fas fa-info-circle", style={"margin-right": "8px", "color": "var(--apple-blue)"}),
                "请输入所有宿舍数量以查看床位统计"
            ], style={"color": "var(--apple-gray)", "font-style": "italic"})
        ])

    male_beds = m6 * 6 + m4 * 4
    female_beds = f6 * 6 + f4 * 4
    total_beds = male_beds + female_beds
    
    male_students = len(student_data[student_data["Sex"] == "男"])
    female_students = len(student_data[student_data["Sex"] == "女"])
    total_students = male_students + female_students
    
    male_sufficient = male_beds >= male_students
    female_sufficient = female_beds >= female_students
    overall_sufficient = male_sufficient and female_sufficient
    
    # 计算利用率
    male_utilization = (male_students / male_beds * 100) if male_beds > 0 else 0
    female_utilization = (female_students / female_beds * 100) if female_beds > 0 else 0
    overall_utilization = (total_students / total_beds * 100) if total_beds > 0 else 0
    
    return html.Div([
        # 床位统计卡片
        html.Div([
            html.Div([
                html.I(className="fas fa-mars", style={"margin-right": "8px", "color": "#1f77b4"}),
                html.Span("男生床位", style={"font-weight": "600"})
            ], style={"margin-bottom": "8px"}),
            html.Div([
                html.Span(f"{male_beds}", style={"font-size": "24px", "font-weight": "700", "color": "var(--apple-text-primary)"}),
                html.Span(f" / {male_students}", style={"color": "var(--apple-gray)", "margin-left": "4px"}),
                html.Span(f" ({male_utilization:.1f}%)", style={"color": "var(--apple-gray)", "font-size": "14px", "margin-left": "8px"})
            ]),
            html.Div([
                html.I(className="fas fa-check-circle" if male_sufficient else "fas fa-exclamation-triangle", 
                      style={"margin-right": "6px", "color": "#28a745" if male_sufficient else "#dc3545"}),
                html.Span("充足" if male_sufficient else f"不足 {male_students - male_beds} 个", 
                         style={"color": "#28a745" if male_sufficient else "#dc3545", "font-size": "14px"})
            ], style={"margin-top": "4px"})
        ], className="apple-info-card", style={"margin-bottom": "16px"}),
        
        # 女生床位统计
        html.Div([
            html.Div([
                html.I(className="fas fa-venus", style={"margin-right": "8px", "color": "#e377c2"}),
                html.Span("女生床位", style={"font-weight": "600"})
            ], style={"margin-bottom": "8px"}),
            html.Div([
                html.Span(f"{female_beds}", style={"font-size": "24px", "font-weight": "700", "color": "var(--apple-text-primary)"}),
                html.Span(f" / {female_students}", style={"color": "var(--apple-gray)", "margin-left": "4px"}),
                html.Span(f" ({female_utilization:.1f}%)", style={"color": "var(--apple-gray)", "font-size": "14px", "margin-left": "8px"})
            ]),
            html.Div([
                html.I(className="fas fa-check-circle" if female_sufficient else "fas fa-exclamation-triangle", 
                      style={"margin-right": "6px", "color": "#28a745" if female_sufficient else "#dc3545"}),
                html.Span("充足" if female_sufficient else f"不足 {female_students - female_beds} 个", 
                         style={"color": "#28a745" if female_sufficient else "#dc3545", "font-size": "14px"})
            ], style={"margin-top": "4px"})
        ], className="apple-info-card", style={"margin-bottom": "16px"}),
        
        # 总体统计
        html.Div([
            html.Div([
                html.I(className="fas fa-chart-pie", style={"margin-right": "8px", "color": "var(--apple-blue)"}),
                html.Span("总体统计", style={"font-weight": "600"})
            ], style={"margin-bottom": "8px"}),
            html.Div([
                html.Span(f"{total_beds}", style={"font-size": "24px", "font-weight": "700", "color": "var(--apple-text-primary)"}),
                html.Span(f" / {total_students}", style={"color": "var(--apple-gray)", "margin-left": "4px"}),
                html.Span(f" ({overall_utilization:.1f}%)", style={"color": "var(--apple-gray)", "font-size": "14px", "margin-left": "8px"})
            ]),
            html.Div([
                html.I(className="fas fa-check-circle" if overall_sufficient else "fas fa-exclamation-triangle", 
                      style={"margin-right": "6px", "color": "#28a745" if overall_sufficient else "#dc3545"}),
                html.Span("床位配置合理" if overall_sufficient else "需要调整床位配置", 
                         style={"color": "#28a745" if overall_sufficient else "#dc3545", "font-size": "14px"})
            ], style={"margin-top": "4px"})
        ], className="apple-info-card"),
        
        # 宿舍类型分布
        html.Div([
            html.Div([
                html.I(className="fas fa-building", style={"margin-right": "8px", "color": "var(--apple-purple)"}),
                html.Span("宿舍分布", style={"font-weight": "600"})
            ], style={"margin-bottom": "12px", "margin-top": "20px"}),
            html.Div([
                html.Div([
                    html.Span("6人间", style={"font-weight": "500", "margin-right": "8px"}),
                    html.Span(f"{m6 + f6} 间", style={"color": "var(--apple-gray)"})
                ], style={"display": "flex", "justify-content": "space-between", "margin-bottom": "4px"}),
                html.Div([
                    html.Span("4人间", style={"font-weight": "500", "margin-right": "8px"}),
                    html.Span(f"{m4 + f4} 间", style={"color": "var(--apple-gray)"})
                ], style={"display": "flex", "justify-content": "space-between"})
            ], style={"font-size": "14px"})
        ])
    ])


@app.callback(
    Output("system-status", "children"),
    Input("system-status-data", "data"),
)
def update_system_status(status_data):
    if not status_data or "student_count" not in status_data:
        return html.Div(
            [
                html.Div("⚡ 系统状态", style={"font-weight": "600", "margin-bottom": "8px", "color": "var(--apple-text-primary)"}),
                html.Div("❌ 未初始化", className="apple-status apple-status-error")
            ]
        )
    return html.Div([
        html.Div("⚡ 系统状态", style={"font-weight": "600", "margin-bottom": "8px", "color": "var(--apple-text-primary)"}),
        html.Div("✅ 已就绪", className="apple-status apple-status-success"),
        html.Div(f"📊 学生数量: {status_data['student_count']}", style={"font-size": "12px", "color": "var(--apple-gray)", "margin-top": "8px"}),
        html.Div(f"🕒 {datetime.now().strftime('%H:%M:%S')}", style={"font-size": "12px", "color": "var(--apple-gray)", "margin-top": "4px"}),
    ])


@app.callback(
    [
        Output("overview-gender-chart", "figure"),
        Output("overview-class-chart", "figure"),
        Output("overview-mbti-chart", "figure"),
    ],
    [Input("overview-interval", "n_intervals")],
    [State("system-status-data", "data")],
)
def update_overview_charts(n_intervals, system_status):
    """更新数据概览页的图表"""
    if not system_status or "student_count" not in system_status:
        return go.Figure(), go.Figure(), go.Figure()

    df = student_data.copy()

    # 定义性别配色方案
    gender_color_map = {'男': '#1f77b4', '女': '#e377c2'}  # 蓝色和粉色

    # 图1：性别分布饼图
    gender_counts = df["Sex"].value_counts().reset_index()
    gender_counts.columns = ["Sex", "Count"]
    fig_gender = px.pie(
        gender_counts,
        names="Sex",
        values="Count",
        title="学生性别分布",
        color="Sex",
        color_discrete_map=gender_color_map
    )
    fig_gender.update_layout(title_x=0.5)

    # 图2：班级分布条形图 - 按专业分色，显示所有数据
    class_counts = df["Class"].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    
    # 从班级名称中提取专业信息
    def extract_major(class_name):
        """从班级名称中提取专业"""
        if '食工' in class_name or '食品工程' in class_name:
            return '食品工程'
        elif '食品安全' in class_name:
            return '食品安全'
        elif '生物工程' in class_name:
            return '生物工程'
        elif '包装工程' in class_name:
            return '包装工程'
        else:
            # 尝试提取其他专业模式
            # 匹配类似 "24专业名1" 的模式
            match = re.search(r'24([^0-9]+)', class_name)
            if match:
                return match.group(1)
            return '其他专业'
    
    class_counts['Major'] = class_counts['Class'].apply(extract_major)
    
    # 定义专业配色方案
    major_colors = {
        '食品工程': '#FF6B6B',    # 红色
        '食品安全': '#4ECDC4',    # 青色
        '生物工程': '#45B7D1',    # 蓝色
        '包装工程': '#96CEB4',    # 绿色
        '其他专业': '#FECA57'     # 黄色
    }
    
    fig_class = px.bar(
        class_counts, 
        x="Class", 
        y="Count", 
        title="学生班级分布",
        color="Major",
        color_discrete_map=major_colors,
        labels={"Major": "专业", "Class": "班级", "Count": "人数"}
    )
    fig_class.update_layout(
        title_x=0.5,
        xaxis_tickangle=-45,
        height=500
    )
    
    # 图3：MBTI分布条形图 - 显示所有数据
    mbti_counts = df["MBTI"].value_counts().reset_index()
    mbti_counts.columns = ["MBTI", "Count"]
    fig_mbti = px.bar(
        mbti_counts, 
        x="MBTI", 
        y="Count", 
        title="MBTI人格分布",
        color="MBTI", 
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_mbti.update_layout(
        title_x=0.5,
        showlegend=False,  # MBTI类型太多，隐藏图例
        height=400
    )

    return fig_gender, fig_class, fig_mbti


# --- 页面渲染与导航 ---
@app.callback(
    [
        Output("nav-overview", "active"),
        Output("nav-optimization", "active"),
        Output("nav-results", "active"),
        Output("nav-explanation", "active"),
        Output("nav-statistics", "active"),
    ],
    [Input("url", "pathname")],
)
def update_nav_active(pathname):
    """根据URL更新导航栏的激活状态"""
    return (
        pathname == "/",
        pathname == "/optimization",
        pathname == "/results",
        pathname == "/explanation",
        pathname == "/statistics",
    )


@app.callback(
    Output("download-excel", "data"),
    Input("export-excel", "n_clicks"),
    State("allocation-results-data", "data"),
    prevent_initial_call=True,
)
def export_excel(n_clicks, results_data):
    """将分配结果导出为Excel文件"""
    if not results_data or "details" not in results_data:
        return dash.no_update
    
    df = pd.DataFrame(results_data["details"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dormitory_allocation_{timestamp}.xlsx"
    
    return dcc.send_data_frame(df.to_excel, filename, sheet_name="Allocation", index=False)


@app.callback(
    Output("room-compatibility-barchart", "figure"),
    Input("allocation-results-data", "data")
)
def update_room_compatibility_chart(results_data):
    """根据Store中的数据更新宿舍兼容度排名图表"""
    if (
        not results_data
        or "explanation" not in results_data
        or "metrics" not in results_data["explanation"]
        or "room_scores" not in results_data["explanation"]["metrics"]
    ):
        return go.Figure().update_layout(
            title="各宿舍平均兼容度 (暂无数据)",
            template="plotly_white",
            height=400
        )
    
    try:
        room_scores = results_data["explanation"]["metrics"]["room_scores"]
        
        scores_df = pd.DataFrame.from_dict(room_scores, orient="index", columns=["Compatibility"])
        scores_df.dropna(inplace=True)
        scores_df = scores_df.reset_index().rename(columns={"index": "RoomID"})
        
        scores_df.sort_values("Compatibility", ascending=False, inplace=True)
        
        # 优化：创建更有意义的颜色分组标签
        bins = [i for i in range(0, len(scores_df) + 10, 10)]
        labels = [f"排名 {i+1}-{i+10}" for i in range(0, len(scores_df), 10)]
        if not labels: # 处理宿舍数小于10的情况
             labels = [f"排名 1-{len(scores_df)}"] if len(scores_df) > 0 else []

        scores_df["Rank Group"] = pd.cut(
            np.arange(len(scores_df)),
            bins=bins,
            right=False,
            labels=labels[:len(bins)-1] # 确保标签数量正确
        )
        
        fig = px.bar(
            scores_df,
            x="RoomID",
            y="Compatibility",
            color="Rank Group", # 使用新的分组列进行着色
            title="各宿舍平均兼容度排名",
            labels={"RoomID": "宿舍号", "Compatibility": "平均兼容度得分", "Rank Group": "排名区间"},
            text_auto=".3f"
        )
        fig.update_traces(textangle=0, textposition="outside")
        fig.update_layout(
            xaxis={'categoryorder':'total descending'},
            template="plotly_white",
            height=600, # 增加高度以容纳图例
            yaxis_range=[0,1]
        )
        return fig
    except Exception as e:
        logger.error(f"渲染宿舍兼容度图表时发生错误: {e}", exc_info=True)
        return go.Figure().update_layout(title=f"渲染图表时出错: {e}")


# --- 主函数 ---
def main():
    """主函数，初始化并运行Web服务器"""
    if initialize_system():
        logger.info("系统初始化成功，启动生产级Web服务器 (Waitress)...")
        logger.info("访问 http://localhost:8050 查看应用")
        serve(app.server, host="0.0.0.0", port=8050)
    else:
        logger.error("系统初始化失败，Web服务器未启动。请检查错误日志。")


if __name__ == "__main__":
    main()