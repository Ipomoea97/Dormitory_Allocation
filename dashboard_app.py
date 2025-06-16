"""
Dash可视化界面
宿舍分配系统的Web界面
"""

import os

# 警告修复：在导入numpy/sklearn之前设置OMP_NUM_THREADS，以避免MKL在Windows上的内存泄漏警告
# 根据警告信息，将线程数设置为一个较小的值
if os.name == 'nt' and 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '2'

import logging
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


# 初始化Dash应用
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css",
        "https://codepen.io/chriddyp/pen/bWLwgP.css",
    ],
    suppress_callback_exceptions=True,
    url_base_pathname="/",
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
# 样式定义
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "border-right": "1px solid #dee2e6",
}

CONTENT_STYLE = {
    "margin-left": "20rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


def create_layout():
    """创建并返回应用的主布局"""
    sidebar = html.Div(
        [
            html.H2("宿舍分配系统", className="display-4"),
            html.Hr(),
            html.P("v1.0", className="lead"),
            dbc.Nav(
                [
                    dbc.NavLink("📊 数据概览", href="/", active="exact", id="nav-overview"),
                    dbc.NavLink("🎯 分配优化", href="/optimization", active="exact", id="nav-optimization"),
                    dbc.NavLink("📋 分配结果", href="/results", active="exact", id="nav-results"),
                    dbc.NavLink("🔍 决策分析", href="/explanation", active="exact", id="nav-explanation"),
                    dbc.NavLink("📈 统计分析", href="/statistics", active="exact", id="nav-statistics"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
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
            html.Div(id="page-content", style=CONTENT_STYLE),
            # 用于触发数据概览页面更新的定时器
            dcc.Interval(id="overview-interval", interval=10 * 1000, disabled=False),
            # 用于触发优化状态更新的定时器
            dcc.Interval(
                id="optimization-interval",
                interval=500,  # 0.5秒更新一次
                n_intervals=0,
                disabled=True,
            ),
        ]
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
    """在后台线程中启动和管理优化过程"""

    # 在主线程中立即创建初始状态，确保UI可以即时响应
    optimization_threads[session_id] = {
        "thread": None,  # 将存储线程对象
        "running": True,
        "stop_requested": False,
        "progress": 0,
        "message": "正在准备优化...",
        "results": None,
        "progress_data": {"generations": [], "fitness_values": []},  # 初始化进度数据
    }
    logger.info(f"为会话 {session_id} 初始化优化状态。")

    def optimize():
        """优化执行函数，在线程内运行"""

        def progress_callback(generation, fitness, message):
            """用于从优化器内部更新进度的回调"""
            if optimization_threads[session_id].get("stop_requested"):
                raise InterruptedError("用户中断了优化。")

            current_status = optimization_threads.get(session_id, {})
            # 计算进度百分比
            progress = (generation / generations) * 100 if generations > 0 else 0
            current_status["progress"] = progress
            current_status["message"] = (
                f"代: {generation}/{generations} - 适应度: {fitness:.4f} - {message}"
            )
            
            # 记录详细进度用于绘图
            current_status["progress_data"]["generations"].append(generation)
            current_status["progress_data"]["fitness_values"].append(fitness)


        try:
            logger.info(f"会话 {session_id}: 开始优化...")
            
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

                results_data = {
                    "summary": {
                        "total_students": len(results_df),
                        "total_rooms": results_df["RoomID"].nunique(),
                        "fitness": fitness_value,
                        "mean_compatibility": allocation_metrics.get("mean_compatibility", 0),
                    },
                    "details": results_df.to_dict("records"),
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
                                dbc.CardHeader("学生班级分布 (Top 10)"),
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
                                dbc.CardHeader("MBTI人格分布 (Top 10)"),
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
                                            {"name": i, "id": i}
                                            for i in student_data.columns
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
    """创建并返回分配优化页面的布局"""
    return html.Div(
        [
            html.H3("分配优化设置", className="mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("宿舍容量设置", className="mb-3"),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("6人间男生宿舍数量"),
                                    dbc.Input(id="male-6-rooms", type="number", value=21, min=0, step=1),
                                ],
                                className="mb-3",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("6人间女生宿舍数量"),
                                    dbc.Input(id="female-6-rooms", type="number", value=30, min=0, step=1),
                                ],
                                className="mb-3",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("4人间男生宿舍数量"),
                                    dbc.Input(id="male-4-rooms", type="number", value=0, min=0, step=1),
                                ],
                                className="mb-3",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("4人间女生宿舍数量"),
                                    dbc.Input(id="female-4-rooms", type="number", value=0, min=0, step=1),
                                ],
                                className="mb-3",
                            ),
                            html.Div(id="bed-capacity-info", className="mt-3 text-muted"),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("遗传算法超参数"),
                                dbc.CardBody(
                                    [
                                        dbc.InputGroup(
                                            [dbc.InputGroupText("种群大小"), dbc.Input(id="population-size", type="number", value=100, min=20, max=500)],
                                            className="mb-3",
                                        ),
                                        dbc.InputGroup(
                                            [dbc.InputGroupText("迭代代数"), dbc.Input(id="generations", type="number", value=200, min=50, max=1000)],
                                            className="mb-3",
                                        ),
                                        dbc.InputGroup(
                                            [dbc.InputGroupText("精英个体数"), dbc.Input(id="elite-size", type="number", value=10, min=1, max=50)],
                                            className="mb-3",
                                        ),
                                        dbc.InputGroup(
                                            [dbc.InputGroupText("锦标赛大小"), dbc.Input(id="tournament-size", type="number", value=5, min=2, max=20)],
                                            className="mb-3",
                                        ),
                                        dbc.InputGroup(
                                            [dbc.InputGroupText("变异率"), dbc.Input(id="mutation-rate", type="number", value=0.1, min=0.01, max=0.5, step=0.01)],
                                            className="mb-3",
                                        ),
                                        dbc.InputGroup(
                                            [dbc.InputGroupText("交叉率"), dbc.Input(id="crossover-rate", type="number", value=0.8, min=0.1, max=1.0, step=0.01)],
                                            className="mb-3",
                                        ),
                                        # 新增：同班优先开关
                                        html.Div(
                                            dbc.Switch(
                                                id="prioritize-class-switch",
                                                label="优先将同班学生分配在同一宿舍",
                                                value=False,
                                            ),
                                            className="mt-3",
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        md=6,
                    ),
                ],
            ),
            html.Div(
                [
                    html.Button("开始优化", id="start-optimization", className="btn btn-primary btn-lg me-3", n_clicks=0, disabled=False),
                    html.Button("停止优化", id="stop-optimization", className="btn btn-danger btn-lg", n_clicks=0, disabled=True),
                ],
                className="text-center mt-4",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("优化状态"),
                    dbc.CardBody(
                        [
                            html.Div(id="optimization-status", children="系统就绪，等待优化任务..."),
                            dcc.Graph(id="optimization-progress"),
                        ]
                    ),
                ],
                className="mt-4",
            ),
        ]
    )


def create_results_page():
    """创建结果页面"""
    return html.Div(
        [
            html.H1("分配结果", className="mb-4"),
            html.Div(id="allocation-summary", className="mb-4"),
            dbc.Card(
                [
                    dbc.CardHeader(
                        dbc.Row(
                            [
                                dbc.Col(html.H4("宿舍分配详情", className="card-title mb-0"), width="auto"),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="results-page-size-dropdown",
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
                                    className="ms-auto" # 靠右对齐
                                )
                            ],
                            align="center",
                            justify="between"
                        )
                    ),
                    dbc.CardBody(html.Div(id="room-details")),
                ],
                className="mb-4",
            ),
            # 新增：宿舍兼容度可视化
            dbc.Card(
                [
                    dbc.CardHeader(html.H4("各宿舍平均兼容度排名", className="card-title mb-0")),
                    dbc.CardBody(dcc.Graph(id="room-compatibility-barchart"))
                ],
                className="mb-4",
            ),
            html.Div(
                [
                    html.Button("导出Excel", id="export-excel", className="btn btn-success me-3"),
                    html.Button("导出PDF报告", id="export-pdf", className="btn btn-info", disabled=True),
                    dbc.Tooltip("PDF导出功能正在开发中...", target="export-pdf"),
                ],
                className="text-center",
            ),
            dcc.Download(id="download-excel"),
        ]
    )


def create_explanation_page():
    """创建解释页面"""
    return html.Div(
        [
            html.H1("可解释性分析", className="mb-4"),
            html.Div([html.H4("特征重要性分析", className="card-title"), dcc.Graph(id="feature-importance-plot")], className="card card-body mb-4"),
            html.Div(
                [
                    html.H4("宿舍兼容性分析", className="card-title"),
                    html.Div([
                        html.Label("选择宿舍:", className="form-label"),
                        dcc.Dropdown(id="room-selector", options=[], value=None, className="mb-3"),
                        html.Div(id="room-compatibility-analysis"),
                    ]),
                ],
                className="card card-body mb-4",
            ),
        ]
    )


def create_statistics_page():
    """创建统计分析页面"""
    return html.Div(
        [
            html.H1("统计分析", className="mb-4"),
            html.Div([html.H4("分配质量统计", className="card-title"), dcc.Graph(id="allocation-quality-stats")], className="card card-body mb-4"),
            html.Div([html.H4("特征相关性分析", className="card-title"), dcc.Graph(id="feature-correlation")], className="card card-body mb-4"),
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
        logger.info(f"[{session_id}] 收到启动优化请求。")
        # 仅当没有线程在运行时才启动
        if not optimization_threads.get(session_id, {}).get("running", False):
            start_optimization_thread(
                session_id, m6, f6, m4, f4, pop, gen, elite, tour, mut, cross, prioritize_class
            )
            clear_data = True
    
    elif triggered_id == "stop-optimization" and stop_clicks > 0:
        logger.info(f"[{session_id}] 收到停止优化请求。")
        if session_id in optimization_threads:
            optimization_threads[session_id]["stop_requested"] = True
            optimization_threads[session_id]["message"] = "正在停止优化..."

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
    """从会话存储中轮询并更新优化状态UI"""
    if not session_id_data or session_id_data["session_id"] not in optimization_threads:
        return html.P("系统就绪，等待优化任务...", className="text-muted")

    status = optimization_threads[session_id_data["session_id"]]
    progress = status.get("progress", 0)
    message = status.get("message", "等待中...")

    if status.get("running", False):
        return html.Div([
            html.P(f"状态: {message}"),
            dbc.Progress(value=progress, style={"height": "20px"}),
            html.P(f"进度: {progress:.1f}%", className="mt-2 text-muted"),
        ])
    else:
        return html.P(f"状态: {message}", className="text-success" if "完成" in message else "text-danger")


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
            status["results"] = None  # 清空，防止重复发送
            logger.info(f"[{session_id_data['session_id']}] 成功从线程获取优化结果并发送至前端。")
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
        return go.Figure().update_layout(title="优化进度", template="plotly_white", height=400)
    
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
    
    fig.update_layout(title="优化进度 - 适应度变化", xaxis_title="迭代代数", yaxis_title="适应度得分", template="plotly_white", height=400)
    return fig


@app.callback(
    Output("allocation-summary", "children"),
    Input("allocation-results-data", "data")
)
def update_allocation_summary(results_data):
    """更新分配结果摘要"""
    try:
        if not results_data or "summary" not in results_data:
            return html.Div("暂无分配结果，请先进行优化。", className="alert alert-info")
        
        summary = results_data["summary"]
        return html.Div(
            [
                html.Div(html.Div([html.H4(f"{summary.get('total_students', 'N/A')}", className="card-title text-primary"), html.P("学生总数", className="card-text")], className="card-body text-center"), className="card col-md-3"),
                html.Div(html.Div([html.H4(f"{summary.get('total_rooms', 'N/A')}", className="card-title text-info"), html.P("宿舍总数", className="card-text")], className="card-body text-center"), className="card col-md-3"),
                html.Div(html.Div([html.H4(f"{summary.get('fitness', 0):.4f}", className="card-title text-success"), html.P("适应度得分", className="card-text")], className="card-body text-center"), className="card col-md-3"),
                html.Div(html.Div([html.H4(f"{summary.get('mean_compatibility', 0):.4f}", className="card-title text-secondary"), html.P("平均兼容度", className="card-text")], className="card-body text-center"), className="card col-md-3"),
            ],
            className="row",
        )
    except Exception as e:
        logger.error(f"渲染摘要时发生错误: {e}", exc_info=True)
        return html.Div(f"渲染分配摘要时出错: {e}", className="alert alert-danger")


@app.callback(
    Output("room-details", "children"),
    [Input("allocation-results-data", "data"),
     Input("results-page-size-dropdown", "value")] # 添加分页控件作为输入
)
def update_room_details(results_data, page_size):
    """更新宿舍的分配详情"""
    try:
        if not results_data or "details" not in results_data:
            return html.Div("暂无详细分配数据，请先完成优化。", className="alert alert-info")

        allocation_details = results_data["details"]
        if not allocation_details:
            return html.Div("分配详情为空。", className="alert alert-warning")

        df = pd.DataFrame(allocation_details)
        
        # 创建表格的函数
        def create_data_table(dataframe, table_id):
            return dash_table.DataTable(
                id=table_id,
                columns=[
                    {"name": "宿舍号", "id": "RoomID"},
                    {"name": "类型", "id": "RoomType"},
                    {"name": "学生ID", "id": "StudentID"},
                    {"name": "姓名", "id": "Name"},
                    {"name": "性别", "id": "Sex"},
                    {"name": "班级", "id": "Class"},
                    {"name": "MBTI", "id": "MBTI"},
                ],
                data=dataframe.to_dict("records"),
                page_size=page_size, # 使用回调传入的 page_size
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "5px"},
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold",
                },
            )

        all_rooms_table = create_data_table(df, "all-allocation-table")

        return all_rooms_table
    except Exception as e:
        logger.error(f"渲染房间详情时发生错误: {e}", exc_info=True)
        return html.Div(f"渲染房间详情时出错: {e}", className="alert alert-danger")


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
        Output("feature-correlation", "figure"),
    ],
    Input("allocation-results-data", "data"),
)
def update_statistics_charts(results_data):
    try:
        if not results_data:
            return go.Figure(), go.Figure()
        
        summary = results_data.get("summary", {})
        quality_fig = go.Figure(go.Bar(
            x=["适应度得分", "平均兼容度"],
            y=[summary.get("fitness", 0), summary.get("mean_compatibility", 0)],
            marker_color=["#1f77b4", "#2ca02c"],
        ))
        quality_fig.update_layout(title="分配质量统计", template="plotly_white")

        if student_data is None:
            return quality_fig, go.Figure().update_layout(title="特征相关性矩阵 (数据未加载)")

        # 修复: 将numpy数组转换为带有特征名称的DataFrame以计算相关性
        processed_df = pd.DataFrame(
            preprocessor.transform(student_data),
            columns=preprocessor.get_feature_names()
        )
        correlation_matrix = processed_df.corr()
        correlation_fig = px.imshow(
            correlation_matrix,
            color_continuous_scale="RdBu_r",
            title="特征相关性矩阵",
        )
        return quality_fig, correlation_fig
    except Exception as e:
        logger.error(f"渲染统计图表时发生错误: {e}", exc_info=True)
        error_fig = go.Figure().update_layout(title=f"渲染图表时出错: {e}")
        return error_fig, error_fig


@app.callback(
    Output("bed-capacity-info", "children"),
    [
        Input("male-6-rooms", "value"), Input("female-6-rooms", "value"),
        Input("male-4-rooms", "value"), Input("female-4-rooms", "value"),
    ],
)
def update_bed_capacity_info(m6, f6, m4, f4):
    if not all(v is not None for v in [m6, f6, m4, f4]):
        return "请输入所有宿舍数量。"

    male_beds = m6 * 6 + m4 * 4
    female_beds = f6 * 6 + f4 * 4
    
    male_students = len(student_data[student_data["Sex"] == "男"])
    female_students = len(student_data[student_data["Sex"] == "女"])
    
    male_sufficient = male_beds >= male_students
    female_sufficient = female_beds >= female_students
    
    return html.Div([
        html.P(f"男生床位: {male_beds} (需: {male_students})", className="text-success" if male_sufficient else "text-danger"),
        html.P(f"女生床位: {female_beds} (需: {female_students})", className="text-success" if female_sufficient else "text-danger"),
        html.P(f"总床位: {male_beds + female_beds} (总学生: {male_students + female_students})"),
        html.P("✅ 床位充足" if male_sufficient and female_sufficient else "⚠️ 床位不足，请调整！", className="text-success" if male_sufficient and female_sufficient else "text-danger fw-bold")
    ])


@app.callback(
    Output("system-status", "children"),
    Input("system-status-data", "data"),
)
def update_system_status(status_data):
    if not status_data or "student_count" not in status_data:
        return html.Div([html.P("系统状态:", className="fw-bold"), html.P("❌ 未初始化", className="text-danger")])
    return html.Div([
        html.P("系统状态:", className="fw-bold"),
        html.P("✅ 已初始化", className="text-success"),
        html.P(f"学生数量: {status_data['student_count']}", className="small text-muted"),
        html.P(f"当前时间: {datetime.now().strftime('%H:%M:%S')}", className="small text-muted"),
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

    # 图2：班级分布条形图
    class_counts = df["Class"].value_counts().nlargest(10).reset_index()
    class_counts.columns = ["Class", "Count"]
    fig_class = px.bar(
        class_counts, x="Class", y="Count", title="学生班级分布 (Top 10)"
    )
    fig_class.update_layout(title_x=0.5)
    
    # 图3：MBTI分布条形图
    mbti_counts = df["MBTI"].value_counts().nlargest(10).reset_index()
    mbti_counts.columns = ["MBTI", "Count"]
    fig_mbti = px.bar(
        mbti_counts, x="MBTI", y="Count", title="MBTI人格分布 (Top 10)",
        color="MBTI", color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_mbti.update_layout(title_x=0.5)

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