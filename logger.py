import os
import sys
from datetime import datetime
from typing import Any, Optional
from enum import Enum
from dataclasses import dataclass


class LogLevel(Enum):
    """日志级别枚举"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


@dataclass
class LogConfig:
    """日志配置"""

    enable_debug: bool = True
    enable_timestamps: bool = True
    enable_colors: bool = True
    log_to_file: bool = False
    log_file_path: str = "logs/creatpartner.log"


class Logger:
    """统一日志管理器"""

    # ANSI颜色代码
    COLORS = {
        LogLevel.DEBUG: "\033[36m",  # 青色
        LogLevel.INFO: "\033[32m",  # 绿色
        LogLevel.WARNING: "\033[33m",  # 黄色
        LogLevel.ERROR: "\033[31m",  # 红色
        LogLevel.SUCCESS: "\033[92m",  # 亮绿色
    }

    # Emoji图标
    ICONS = {
        LogLevel.DEBUG: "🔧",
        LogLevel.INFO: "ℹ️",
        LogLevel.WARNING: "⚠️",
        LogLevel.ERROR: "❌",
        LogLevel.SUCCESS: "✅",
    }

    RESET = "\033[0m"

    def __init__(self, config: Optional[LogConfig] = None):
        self.config = config or LogConfig()

        # 从环境变量读取配置
        self.config.enable_debug = os.getenv("DEBUG_MODE", "true").lower() == "true"

        # 确保日志目录存在
        if self.config.log_to_file:
            os.makedirs(os.path.dirname(self.config.log_file_path), exist_ok=True)

    def _format_message(self, level: LogLevel, message: str, **kwargs) -> str:
        """格式化日志消息"""
        parts = []

        # 添加时间戳
        if self.config.enable_timestamps:
            timestamp = datetime.now().strftime("%H:%M:%S")
            parts.append(f"[{timestamp}]")

        # 添加图标
        parts.append(self.ICONS.get(level, ""))

        # 添加级别标识
        if level != LogLevel.INFO:  # INFO级别不显示标识
            parts.append(f"[{level.value}]")

        # 添加消息
        parts.append(message)

        # 如果有额外参数，格式化显示
        if kwargs:
            extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            parts.append(f"({extra_info})")

        return " ".join(parts)

    def _colorize(self, level: LogLevel, message: str) -> str:
        """为消息添加颜色"""
        if not self.config.enable_colors:
            return message

        color = self.COLORS.get(level, "")
        return f"{color}{message}{self.RESET}"

    def _write_log(self, level: LogLevel, message: str, **kwargs):
        """写入日志"""
        formatted_message = self._format_message(level, message, **kwargs)

        # 控制台输出
        if level == LogLevel.DEBUG and not self.config.enable_debug:
            return  # 跳过调试信息

        colored_message = self._colorize(level, formatted_message)
        print(colored_message)

        # 文件输出
        if self.config.log_to_file:
            try:
                with open(self.config.log_file_path, "a", encoding="utf-8") as f:
                    f.write(f"{formatted_message}\n")
            except Exception as e:
                print(f"❌ 日志写入失败: {e}")

        # 广播输出（如果设置了广播函数）
        if _broadcast_function:
            try:
                _broadcast_function(
                    level=level.value, message=formatted_message, **kwargs
                )
            except Exception as e:
                print(f"❌ 日志广播失败: {e}")

    def debug(self, message: str, **kwargs):
        """调试信息"""
        self._write_log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """普通信息"""
        self._write_log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """警告信息"""
        self._write_log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """错误信息"""
        self._write_log(LogLevel.ERROR, message, **kwargs)

    def success(self, message: str, **kwargs):
        """成功信息"""
        self._write_log(LogLevel.SUCCESS, message, **kwargs)

    # 便利方法：兼容原有的print风格
    def print_debug(self, *args, **kwargs):
        """调试打印"""
        message = " ".join(str(arg) for arg in args)
        self.debug(message)

    def print_info(self, *args, **kwargs):
        """信息打印"""
        message = " ".join(str(arg) for arg in args)
        self.info(message)

    def print_warning(self, *args, **kwargs):
        """警告打印"""
        message = " ".join(str(arg) for arg in args)
        self.warning(message)

    def print_error(self, *args, **kwargs):
        """错误打印"""
        message = " ".join(str(arg) for arg in args)
        self.error(message)

    def print_success(self, *args, **kwargs):
        """成功打印"""
        message = " ".join(str(arg) for arg in args)
        self.success(message)

    # 特殊方法：性能监控
    def performance(self, operation_name: str, duration: float, **kwargs):
        """性能监控日志"""
        if duration > 3.0:  # 超过3秒的操作记录为警告
            self.warning(
                f"慢操作: {operation_name}", duration=f"{duration:.2f}秒", **kwargs
            )
        else:
            self.info(
                f"操作完成: {operation_name}", duration=f"{duration:.2f}秒", **kwargs
            )

    # 特殊方法：步骤跟踪
    def step(self, step_name: str, status: str = "开始", **kwargs):
        """步骤跟踪"""
        if status == "开始":
            self.info(f"🕐 开始 {step_name}", **kwargs)
        elif status == "完成":
            self.success(f"⏱️ 完成 {step_name}", **kwargs)
        elif status == "失败":
            self.error(f"💥 失败 {step_name}", **kwargs)
        else:
            self.info(f"📝 {status} {step_name}", **kwargs)

    # 特殊方法：搜索操作
    def search_operation(
        self, query: str, results_count: int, search_type: str = "搜索", **kwargs
    ):
        """搜索操作日志"""
        self.info(
            f"{search_type}: {query[:50]}{'...' if len(query) > 50 else ''}",
            results=results_count,
            **kwargs,
        )

    # 特殊方法：Agent操作
    def agent_operation(
        self, agent_name: str, operation: str, status: str = "执行", **kwargs
    ):
        """Agent操作日志"""
        if status == "初始化":
            self.debug(f"🔧 初始化{agent_name}...", **kwargs)
        elif status == "完成":
            self.success(f"✅ {agent_name}{operation}完成", **kwargs)
        elif status == "失败":
            self.error(f"❌ {agent_name}{operation}失败", **kwargs)
        else:
            self.info(f"🤖 {agent_name}: {operation}", **kwargs)


# 创建全局日志实例
_global_logger = None
_broadcast_function = None


def get_logger() -> Logger:
    """获取全局日志实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger()
    return _global_logger


def set_broadcast_function(func):
    """设置广播函数，用于将日志发送到外部系统"""
    global _broadcast_function
    _broadcast_function = func


def create_logger(config: Optional[LogConfig] = None) -> Logger:
    """创建新的日志实例"""
    return Logger(config)


# 便利函数：直接使用
def debug(message: str, **kwargs):
    """调试日志"""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs):
    """信息日志"""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs):
    """警告日志"""
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs):
    """错误日志"""
    get_logger().error(message, **kwargs)


def success(message: str, **kwargs):
    """成功日志"""
    get_logger().success(message, **kwargs)


def step(step_name: str, status: str = "开始", **kwargs):
    """步骤跟踪"""
    get_logger().step(step_name, status, **kwargs)


def performance(operation_name: str, duration: float, **kwargs):
    """性能监控"""
    get_logger().performance(operation_name, duration, **kwargs)


def search_operation(
    query: str, results_count: int, search_type: str = "搜索", **kwargs
):
    """搜索操作"""
    get_logger().search_operation(query, results_count, search_type, **kwargs)


def agent_operation(agent_name: str, operation: str, status: str = "执行", **kwargs):
    """Agent操作"""
    get_logger().agent_operation(agent_name, operation, status, **kwargs)


# 使用示例
if __name__ == "__main__":
    print("📝 Logger测试")

    logger = get_logger()

    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    logger.success("这是成功信息")

    logger.step("测试步骤", "开始")
    logger.step("测试步骤", "完成")

    logger.performance("测试操作", 2.5)
    logger.search_operation("AI测试", 5, "Jina搜索")
    logger.agent_operation("搜索代理", "搜索任务", "完成")
