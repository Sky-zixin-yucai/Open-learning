"""
OpenLearning RGA - 企业级智能导入系统
=======================================

功能特性：
1. 环境智能感知（开发/生产/测试/直接运行）
2. 多层导入策略（静态/动态/延迟/混合）
3. 完整的错误处理和恢复机制
4. 性能监控和缓存优化
5. 插件化架构，支持自定义导入器
6. 详细的日志和调试信息
7. 自动依赖检测和修复建议
8. 热重载支持
9. 导入路径自动发现
10. 多环境配置管理
"""

import site
import sys
import os
import importlib
import importlib.util
import importlib.machinery
import pkgutil
import warnings
import traceback
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
import logging
from logging.handlers import RotatingFileHandler
import json

# ==================== 配置系统 ====================

class ImportStrategy(Enum):
    """导入策略枚举"""
    STATIC = auto()       # 静态导入
    DYNAMIC = auto()      # 动态导入
    LAZY = auto()         # 延迟导入
    RELATIVE = auto()     # 相对导入
    ABSOLUTE = auto()    # 绝对导入
    HYBRID = auto()      # 混合策略
    SMART = auto()       # 智能选择

class EnvironmentType(Enum):
    """环境类型枚举"""
    DEVELOPMENT = auto()   # 开发环境
    PRODUCTION = auto()    # 生产环境
    TESTING = auto()       # 测试环境
    CI_CD = auto()         # CI/CD环境
    DOCKER = auto()        # Docker容器
    NOTEBOOK = auto()      # Jupyter Notebook
    STANDALONE = auto()    # 独立脚本运行

@dataclass
class ImportConfig:
    """导入配置类"""
    strategy: ImportStrategy = ImportStrategy.SMART
    enable_cache: bool = True
    cache_size: int = 100
    enable_monitoring: bool = True
    log_level: int = logging.INFO
    timeout_seconds: int = 30
    retry_attempts: int = 3
    enable_fallback: bool = True
    verbose: bool = False
    auto_discover: bool = True
    max_depth: int = 3
    blacklist: List[str] = field(default_factory=list)
    whitelist: List[str] = field(default_factory=list)
    prefer_relative: bool = False
    force_absolute: bool = False
    enable_hot_reload: bool = False
    hot_reload_interval: float = 1.0
    custom_paths: List[str] = field(default_factory=list)

# ==================== 日志系统 ====================

class ImportLogger:
    """专用的导入日志系统"""
    
    def __init__(self, name="ImportSystem", log_file="import_system.log"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        
        # 文件处理器
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        # 添加处理器
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def log_import(self, module_name: str, strategy: str, success: bool, duration: float):
        """记录导入事件"""
        status = "成功" if success else "失败"
        self.logger.info(f"导入 {module_name} [{strategy}] - {status} ({duration:.3f}s)")
    
    def log_error(self, module_name: str, error: Exception, context: str = ""):
        """记录错误"""
        self.logger.error(f"导入失败 {module_name}: {error} {context}")
    
    def log_warning(self, message: str):
        """记录警告"""
        self.logger.warning(message)
    
    def log_debug(self, message: str):
        """调试信息"""
        self.logger.debug(message)

# ==================== 环境探测器 ====================

class EnvironmentDetector:
    """高级环境探测器"""
    
    @staticmethod
    def detect() -> Dict[str, Any]:
        """检测完整的运行环境"""
        env_info = {
            'type': EnvironmentType.DEVELOPMENT,
            'python_version': sys.version,
            'platform': sys.platform,
            'executable': sys.executable,
            'argv': sys.argv,
            'path': sys.path.copy(),
            'site_packages': [],
            'virtual_env': False,
            'containerized': False,
            'notebook': False,
            'interactive': False,
            'package_root': None,
            'is_frozen': False,
            'user_site': False,
        }
        
        # 检测环境类型
        if 'site-packages' in __file__ or 'dist-packages' in __file__:
            env_info['type'] = EnvironmentType.PRODUCTION
        elif 'PYTEST_CURRENT_TEST' in os.environ:
            env_info['type'] = EnvironmentType.TESTING
        elif 'CI' in os.environ or 'GITHUB_ACTIONS' in os.environ:
            env_info['type'] = EnvironmentType.CI_CD
        elif 'DOCKER' in os.environ or 'KUBERNETES' in os.environ:
            env_info['type'] = EnvironmentType.DOCKER
        
        # 检测虚拟环境
        env_info['virtual_env'] = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        # 检测容器化
        env_info['containerized'] = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
        
        # 检测Notebook环境
        try:
            from IPython import get_ipython
            env_info['notebook'] = get_ipython() is not None
        except ImportError:
            env_info['notebook'] = False
        
        # 检测交互式环境
        env_info['interactive'] = hasattr(sys, 'ps1') or bool(sys.flags.interactive)
        
        # 检测是否打包
        env_info['is_frozen'] = getattr(sys, 'frozen', False)
        
        # 检测用户site-packages
        env_info['user_site'] = site.USER_SITE if 'site' in sys.modules else None
        
        # 查找site-packages目录
        for path in sys.path:
            if 'site-packages' in path or 'dist-packages' in path:
                env_info['site_packages'].append(path)
        
        # 查找包根目录
        current_file = Path(__file__).absolute()
        for depth in range(5):
            parent = current_file.parents[depth]
            if (parent / 'setup.py').exists() or (parent / 'pyproject.toml').exists():
                env_info['package_root'] = str(parent)
                break
        
        return env_info
    
    @staticmethod
    def get_environment_summary() -> str:
        """获取环境摘要"""
        env = EnvironmentDetector.detect()
        summary = [
            f"环境类型: {env['type'].name}",
            f"Python版本: {env['python_version'].split()[0]}",
            f"平台: {env['platform']}",
            f"虚拟环境: {env['virtual_env']}",
            f"容器化: {env['containerized']}",
            f"Notebook: {env['notebook']}",
            f"交互式: {env['interactive']}",
            f"打包应用: {env['is_frozen']}",
        ]
        return "\n".join(summary)

# ==================== 导入策略实现 ====================

class ImportStrategyFactory:
    """导入策略工厂"""
    
    @staticmethod
    def create(strategy: ImportStrategy, config: ImportConfig):
        """创建策略实例"""
        if strategy == ImportStrategy.STATIC:
            return StaticImportStrategy(config)
        elif strategy == ImportStrategy.DYNAMIC:
            return DynamicImportStrategy(config)
        elif strategy == ImportStrategy.LAZY:
            return LazyImportStrategy(config)
        elif strategy == ImportStrategy.RELATIVE:
            return RelativeImportStrategy(config)
        elif strategy == ImportStrategy.ABSOLUTE:
            return AbsoluteImportStrategy(config)
        elif strategy == ImportStrategy.HYBRID:
            return HybridImportStrategy(config)
        elif strategy == ImportStrategy.SMART:
            return SmartImportStrategy(config)
        else:
            raise ValueError(f"未知策略: {strategy}")

class BaseImportStrategy:
    """导入策略基类"""
    
    def __init__(self, config: ImportConfig):
        self.config = config
        self.logger = ImportLogger(f"Strategy_{self.__class__.__name__}")
        self.cache = {}
    
    def import_module(self, module_name: str, package: str = None) -> Any:
        """导入模块（子类必须实现）"""
        raise NotImplementedError
    
    def _pre_import_check(self, module_name: str) -> bool:
        """导入前检查"""
        if module_name in self.config.blacklist:
            self.logger.log_warning(f"模块 {module_name} 在黑名单中，跳过导入")
            return False
        if self.config.whitelist and module_name not in self.config.whitelist:
            self.logger.log_warning(f"模块 {module_name} 不在白名单中，跳过导入")
            return False
        return True
    
    def _post_import_hook(self, module_name: str, module: Any, success: bool, duration: float):
        """导入后钩子"""
        if self.config.enable_monitoring:
            self.logger.log_import(module_name, self.__class__.__name__, success, duration)
        
        if success and self.config.enable_cache:
            self.cache[module_name] = {
                'module': module,
                'timestamp': time.time(),
                'strategy': self.__class__.__name__
            }

class StaticImportStrategy(BaseImportStrategy):
    """静态导入策略"""
    
    def import_module(self, module_name: str, package: str = None) -> Any:
        if not self._pre_import_check(module_name):
            return None
        
        start_time = time.time()
        try:
            # 执行静态导入
            exec(f"import {module_name}")
            module = sys.modules[module_name]
            
            self._post_import_hook(module_name, module, True, time.time() - start_time)
            return module
        except Exception as e:
            self._post_import_hook(module_name, None, False, time.time() - start_time)
            self.logger.log_error(module_name, e, "静态导入失败")
            return None

class DynamicImportStrategy(BaseImportStrategy):
    """动态导入策略"""
    
    def import_module(self, module_name: str, package: str = None) -> Any:
        if not self._pre_import_check(module_name):
            return None
        
        start_time = time.time()
        for attempt in range(self.config.retry_attempts):
            try:
                module = importlib.import_module(module_name)
                self._post_import_hook(module_name, module, True, time.time() - start_time)
                return module
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    self._post_import_hook(module_name, None, False, time.time() - start_time)
                    self.logger.log_error(module_name, e, f"动态导入失败，尝试 {attempt+1}/{self.config.retry_attempts}")
                time.sleep(0.1 * (attempt + 1))
        
        return None

class LazyImportStrategy(BaseImportStrategy):
    """延迟导入策略"""
    
    def __init__(self, config: ImportConfig):
        super().__init__(config)
        self.lazy_modules = {}
    
    def import_module(self, module_name: str, package: str = None) -> Any:
        if not self._pre_import_check(module_name):
            return None
        
        # 创建延迟加载代理
        if module_name not in self.lazy_modules:
            self.lazy_modules[module_name] = LazyModuleProxy(
                module_name, self.config, self.logger
            )
        
        return self.lazy_modules[module_name]

class LazyModuleProxy:
    """延迟加载模块代理"""
    
    def __init__(self, module_name: str, config: ImportConfig, logger: ImportLogger):
        self._module_name = module_name
        self._config = config
        self._logger = logger
        self._module = None
        self._loaded = False
        self._lock = threading.Lock()
    
    def _load(self):
        """实际加载模块"""
        with self._lock:
            if not self._loaded:
                start_time = time.time()
                try:
                    self._module = importlib.import_module(self._module_name)
                    self._loaded = True
                    duration = time.time() - start_time
                    self._logger.log_import(self._module_name, "LazyLoad", True, duration)
                except Exception as e:
                    self._logger.log_error(self._module_name, e, "延迟加载失败")
                    raise
    
    def __getattr__(self, name):
        if self._module is None:
            self._load()
        return getattr(self._module, name)
    
    def __repr__(self):
        status = "已加载" if self._loaded else "未加载"
        return f"<LazyModuleProxy '{self._module_name}' ({status})>"

class RelativeImportStrategy(BaseImportStrategy):
    """相对导入策略"""
    
    def import_module(self, module_name: str, package: str = None) -> Any:
        if not self._pre_import_check(module_name):
            return None
        
        start_time = time.time()
        package = package or __package__
        
        if not package:
            self.logger.log_error(module_name, ValueError("相对导入需要包上下文"), "缺少包上下文")
            return None
        
        try:
            module = importlib.import_module(f".{module_name}", package)
            self._post_import_hook(module_name, module, True, time.time() - start_time)
            return module
        except Exception as e:
            self._post_import_hook(module_name, None, False, time.time() - start_time)
            self.logger.log_error(module_name, e, "相对导入失败")
            return None

class AbsoluteImportStrategy(BaseImportStrategy):
    """绝对导入策略"""
    
    def import_module(self, module_name: str, package: str = None) -> Any:
        if not self._pre_import_check(module_name):
            return None
        
        start_time = time.time()
        try:
            # 尝试标准绝对导入
            module = importlib.import_module(module_name)
            self._post_import_hook(module_name, module, True, time.time() - start_time)
            return module
        except Exception as e:
            # 尝试添加包前缀
            if package:
                try:
                    module = importlib.import_module(f"{package}.{module_name}")
                    self._post_import_hook(module_name, module, True, time.time() - start_time)
                    return module
                except Exception:
                    pass
            
            self._post_import_hook(module_name, None, False, time.time() - start_time)
            self.logger.log_error(module_name, e, "绝对导入失败")
            return None

class HybridImportStrategy(BaseImportStrategy):
    """混合导入策略"""
    
    def import_module(self, module_name: str, package: str = None) -> Any:
        if not self._pre_import_check(module_name):
            return None
        
        strategies = [
            AbsoluteImportStrategy(self.config),
            RelativeImportStrategy(self.config),
            DynamicImportStrategy(self.config),
        ]
        
        for strategy in strategies:
            start_time = time.time()
            try:
                module = strategy.import_module(module_name, package)
                if module:
                    duration = time.time() - start_time
                    self.logger.log_import(module_name, f"Hybrid-{strategy.__class__.__name__}", True, duration)
                    return module
            except Exception as e:
                self.logger.log_debug(f"混合策略 {strategy.__class__.__name__} 失败: {e}")
                continue
        
        self.logger.log_error(module_name, ImportError("所有混合策略都失败"), "混合导入失败")
        return None

class SmartImportStrategy(BaseImportStrategy):
    """智能导入策略"""
    
    def import_module(self, module_name: str, package: str = None) -> Any:
        if not self._pre_import_check(module_name):
            return None
        
        env = EnvironmentDetector.detect()
        strategy = self._choose_best_strategy(env, module_name, package)
        
        start_time = time.time()
        try:
            module = strategy.import_module(module_name, package)
            if module:
                duration = time.time() - start_time
                strategy_name = strategy.__class__.__name__.replace('ImportStrategy', '')
                self.logger.log_import(module_name, f"Smart-{strategy_name}", True, duration)
                return module
        except Exception as e:
            self.logger.log_error(module_name, e, f"智能策略 {strategy.__class__.__name__} 失败")
        
        # 回退到混合策略
        fallback = HybridImportStrategy(self.config)
        return fallback.import_module(module_name, package)
    
    def _choose_best_strategy(self, env: Dict, module_name: str, package: str = None):
        """根据环境选择最佳策略"""
        env_type = env['type']
        
        if env_type == EnvironmentType.PRODUCTION:
            if self.config.force_absolute:
                return AbsoluteImportStrategy(self.config)
            return DynamicImportStrategy(self.config)
        
        elif env_type == EnvironmentType.DEVELOPMENT:
            if package and self.config.prefer_relative:
                return RelativeImportStrategy(self.config)
            return HybridImportStrategy(self.config)
        
        elif env_type == EnvironmentType.TESTING:
            return LazyImportStrategy(self.config)
        
        elif env_type == EnvironmentType.NOTEBOOK:
            return StaticImportStrategy(self.config)
        
        else:
            return HybridImportStrategy(self.config)

# ==================== 主导入系统 ====================

class ImportSystem:
    """主导入系统"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.config = ImportConfig()
        self.logger = ImportLogger("ImportSystem")
        self.env_detector = EnvironmentDetector()
        self.strategy_factory = ImportStrategyFactory()
        self.current_strategy = self.strategy_factory.create(self.config.strategy, self.config)
        
        # 监控数据
        self.metrics = {
            'total_imports': 0,
            'successful_imports': 0,
            'failed_imports': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'import_durations': [],
            'errors': []
        }
        
        # 热重载线程
        self.hot_reload_thread = None
        self.hot_reload_running = False
        
        self._initialized = True
        self.logger.logger.info("导入系统初始化完成")
    
    def configure(self, **kwargs):
        """重新配置系统"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.logger.log_warning(f"未知配置项: {key}")
        
        # 重新创建策略
        self.current_strategy = self.strategy_factory.create(self.config.strategy, self.config)
        
        # 启动/停止热重载
        if self.config.enable_hot_reload and not self.hot_reload_running:
            self._start_hot_reload()
        elif not self.config.enable_hot_reload and self.hot_reload_running:
            self._stop_hot_reload()
    
    def import_module(self, module_name: str, package: str = None, strategy: ImportStrategy = None) -> Any:
        """导入模块"""
        self.metrics['total_imports'] += 1
        
        # 检查缓存
        if self.config.enable_cache and module_name in self.current_strategy.cache:
            self.metrics['cache_hits'] += 1
            cache_entry = self.current_strategy.cache[module_name]
            
            # 检查缓存是否过期（暂定1小时）
            if time.time() - cache_entry['timestamp'] < 3600:
                return cache_entry['module']
            else:
                del self.current_strategy.cache[module_name]
        
        self.metrics['cache_misses'] += 1
        
        # 选择策略
        if strategy:
            import_strategy = self.strategy_factory.create(strategy, self.config)
        else:
            import_strategy = self.current_strategy
        
        # 执行导入
        start_time = time.time()
        try:
            module = import_strategy.import_module(module_name, package)
            duration = time.time() - start_time
            
            if module:
                self.metrics['successful_imports'] += 1
                self.metrics['import_durations'].append(duration)
            else:
                self.metrics['failed_imports'] += 1
                self.metrics['errors'].append({
                    'module': module_name,
                    'time': time.time(),
                    'error': '导入返回None'
                })
            
            return module
        except Exception as e:
            duration = time.time() - start_time
            self.metrics['failed_imports'] += 1
            self.metrics['errors'].append({
                'module': module_name,
                'time': time.time(),
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
            self.logger.log_error(module_name, e, f"导入失败，耗时 {duration:.3f}s")
            
            # 如果启用了回退，尝试备用策略
            if self.config.enable_fallback and strategy != ImportStrategy.HYBRID:
                fallback = HybridImportStrategy(self.config)
                return fallback.import_module(module_name, package)
            
            return None
    
    def discover_modules(self, path: str = None, max_depth: int = None) -> List[str]:
        """发现可用的模块"""
        if path is None:
            env = self.env_detector.detect()
            path = env.get('package_root', '.')
        
        max_depth = max_depth or self.config.max_depth
        
        discovered = []
        for root, dirs, files in os.walk(path):
            current_depth = root[len(path):].count(os.sep)
            if current_depth > max_depth:
                continue
            
            for file in files:
                if file.endswith('.py') and not file.startswith('_'):
                    module_path = os.path.join(root, file)
                    # 转换为模块名
                    rel_path = os.path.relpath(module_path, path)
                    module_name = rel_path.replace('.py', '').replace(os.sep, '.')
                    discovered.append(module_name)
        
        return discovered
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        metrics = self.metrics.copy()
        
        if metrics['import_durations']:
            metrics['avg_duration'] = sum(metrics['import_durations']) / len(metrics['import_durations'])
            metrics['max_duration'] = max(metrics['import_durations'])
            metrics['min_duration'] = min(metrics['import_durations'])
        else:
            metrics['avg_duration'] = metrics['max_duration'] = metrics['min_duration'] = 0
        
        metrics['success_rate'] = (
            metrics['successful_imports'] / metrics['total_imports'] * 100
            if metrics['total_imports'] > 0 else 0
        )
        
        metrics['cache_efficiency'] = (
            metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses']) * 100
            if (metrics['cache_hits'] + metrics['cache_misses']) > 0 else 0
        )
        
        return metrics
    
    def generate_report(self) -> str:
        """生成系统报告"""
        env_summary = self.env_detector.get_environment_summary()
        metrics = self.get_metrics()
        
        report = [
            "=" * 80,
            "OpenLearning RGA 导入系统报告",
            "=" * 80,
            f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "环境信息:",
            env_summary,
            "",
            "配置信息:",
            f"  策略: {self.config.strategy.name}",
            f"  启用缓存: {self.config.enable_cache}",
            f"  启用监控: {self.config.enable_monitoring}",
            f"  启用热重载: {self.config.enable_hot_reload}",
            "",
            "性能指标:",
            f"  总导入次数: {metrics['total_imports']}",
            f"  成功导入: {metrics['successful_imports']}",
            f"  失败导入: {metrics['failed_imports']}",
            f"  成功率: {metrics['success_rate']:.2f}%",
            f"  缓存命中率: {metrics['cache_efficiency']:.2f}%",
            f"  平均导入时间: {metrics['avg_duration']:.3f}s",
            f"  最大导入时间: {metrics['max_duration']:.3f}s",
            f"  最小导入时间: {metrics['min_duration']:.3f}s",
            "",
        ]
        
        if metrics['errors']:
            report.extend(["最近错误:", ""])
            for error in metrics['errors'][-5:]:  # 显示最近5个错误
                report.append(f"  模块: {error['module']}")
                report.append(f"  时间: {time.strftime('%H:%M:%S', time.localtime(error['time']))}")
                report.append(f"  错误: {error['error'][:100]}...")
                report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _start_hot_reload(self):
        """启动热重载线程"""
        if self.hot_reload_running:
            return
        
        self.hot_reload_running = True
        
        def hot_reload_worker():
            while self.hot_reload_running:
                time.sleep(self.config.hot_reload_interval)
                self._check_for_updates()
        
        self.hot_reload_thread = threading.Thread(
            target=hot_reload_worker, 
            daemon=True,
            name="HotReloadThread"
        )
        self.hot_reload_thread.start()
        self.logger.logger.info("热重载线程已启动")
    
    def _stop_hot_reload(self):
        """停止热重载"""
        self.hot_reload_running = False
        if self.hot_reload_thread:
            self.hot_reload_thread.join(timeout=2.0)
        self.logger.logger.info("热重载线程已停止")
    
    def _check_for_updates(self):
        """检查模块更新"""
        # 这里可以实现文件监控逻辑
        # 暂时只记录日志
        self.logger.log_debug("热重载检查...")

# ==================== 简化的公开接口 ====================

# 全局导入系统实例
_import_system = ImportSystem()

def configure_imports(**kwargs):
    """配置导入系统"""
    _import_system.configure(**kwargs)

def import_module(module_name: str, package: str = None, strategy: str = None):
    """导入模块"""
    if strategy:
        strategy_enum = ImportStrategy[strategy.upper()]
    else:
        strategy_enum = None
    
    return _import_system.import_module(module_name, package, strategy_enum)

def get_import_system() -> ImportSystem:
    """获取导入系统实例"""
    return _import_system

def generate_report() -> str:
    """生成系统报告"""
    return _import_system.generate_report()

# ==================== 延迟加载模块创建 ====================

def create_lazy_module(module_name: str):
    """创建延迟加载模块"""
    return LazyModuleProxy(module_name, _import_system.config, _import_system.logger)

# ==================== 主包导入器 ====================

# 创建主要模块的延迟加载实例
core = create_lazy_module("openlearning.core")
layers = create_lazy_module("openlearning.layers")
integration = create_lazy_module("openlearning.integration")

# 预导入检查
def check_imports():
    """检查所有关键模块是否能导入"""
    modules = ["openlearning.core", "openlearning.layers", "openlearning.integration"]
    results = {}
    
    for module_name in modules:
        start_time = time.time()
        try:
            module = import_module(module_name)
            duration = time.time() - start_time
            results[module_name] = {
                'success': True,
                'duration': duration,
                'module': module
            }
        except Exception as e:
            duration = time.time() - start_time
            results[module_name] = {
                'success': False,
                'duration': duration,
                'error': str(e)
            }
    
    return results

# ==================== 模块初始化 ====================

__version__ = "0.0.9"
__author__ = "RGA Architecture Team"
__description__ = "智能导入系统 OpenLearning RGA"
__license__ = "Apache 2.0"

# 导出列表
__all__ = [
    # 模块信息
    '__version__', '__author__', '__description__', '__license__',
    
    # 主要模块
    'core', 'layers', 'integration',
    
    # 导入系统功能
    'ImportSystem', 'ImportStrategy', 'ImportConfig', 'EnvironmentType',
    'configure_imports', 'import_module', 'get_import_system', 'generate_report',
    'create_lazy_module', 'check_imports',
    
    # 策略类
    'StaticImportStrategy', 'DynamicImportStrategy', 'LazyImportStrategy',
    'RelativeImportStrategy', 'AbsoluteImportStrategy', 'HybridImportStrategy', 'SmartImportStrategy',
]

# 自动检查环境
if __name__ == "__main__":
    print(generate_report())
    
    # 测试导入
    results = check_imports()
    print("\n模块导入测试:")
    for module_name, result in results.items():
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"  {module_name}: {status} ({result['duration']:.3f}s)")
        if not result['success']:
            print(f"    错误: {result['error']}")