"""
Enhanced Configuration System for Cradle + Cortex Integration

此模块提供增强配置系统，用于控制新增特性（SA-KG、双脑系统、ViF 等）。
设计原则：
1. 不修改 Cradle 原有的 Config 类
2. 所有新特性默认 disabled
3. 通过 feature flags 控制每个特性的开关
4. 支持热重载配置
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import yaml

from cradle.utils import Singleton
from cradle.utils.file_utils import assemble_project_path


def _ts_print(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"{timestamp} - {message}")


@dataclass
class SAKGConfig:
    """SA-KG 状态-动作知识图谱配置"""
    enabled: bool = False

    # 嵌入配置
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # 检索配置
    similarity_threshold: float = 0.85
    top_k: int = 5
    as_reference_only: bool = True  # 作为参考提供给 LLM，而非直接执行

    # 存储配置
    storage_backend: str = "chromadb"
    persistence_path: str = "./cache/sa_kg/"
    max_entries: int = 10000

    # 写入条件
    require_reflection_success: bool = True
    min_confidence: float = 0.7


@dataclass
class LittleBrainConfig:
    """Little Brain (System 1) 配置"""
    enabled: bool = True
    retrieval_first: bool = True
    fallback_to_llm: bool = True
    max_latency_ms: int = 500


@dataclass
class BigBrainGatingConfig:
    """Big Brain 门控激活条件"""
    trigger_on_consecutive_failures: int = 3
    trigger_on_low_confidence: float = 0.5
    trigger_on_new_environment: bool = True
    trigger_on_explicit_request: bool = True


@dataclass
class BigBrainConfig:
    """Big Brain (System 2) 配置"""
    enabled: bool = True
    gating: BigBrainGatingConfig = field(default_factory=BigBrainGatingConfig)


@dataclass
class DualBrainConfig:
    """双脑系统配置"""
    enabled: bool = False
    little_brain: LittleBrainConfig = field(default_factory=LittleBrainConfig)
    big_brain: BigBrainConfig = field(default_factory=BigBrainConfig)


@dataclass
class ViFlowConfig:
    """ViF 视觉流防幻觉配置"""
    enabled: bool = False
    inject_detections: bool = True
    detection_format: str = "text"
    max_detections: int = 10
    cross_brain_transfer_enabled: bool = False
    include_bboxes: bool = True
    include_confidence: bool = True


@dataclass
class RTCConfig:
    """RTC 异步执行配置（暂时搁置）"""
    enabled: bool = False
    chunk_duration_ms: int = 200
    target_fps: int = 30


@dataclass
class SkillSynthesisConfig:
    """技能合成配置"""
    enabled: bool = False
    candidate_threshold: int = 3
    promoted_threshold: int = 10
    sandbox_enabled: bool = False
    sandbox_max_retries: int = 3


@dataclass
class LoggingConfig:
    """日志配置"""
    enhanced_features: bool = True
    sa_kg_queries: bool = True
    gating_decisions: bool = True
    performance_metrics: bool = True


@dataclass
class BenchmarkingConfig:
    """性能基准配置"""
    enabled: bool = False
    step_latency: bool = True
    llm_call_rate: bool = True
    sa_kg_hit_rate: bool = True
    task_success_rate: bool = True
    output_path: str = "./cache/benchmarks/"


class EnhancedConfig(metaclass=Singleton):
    """
    增强配置管理器 (单例)

    用法:
        from cradle.config.enhanced_config import EnhancedConfig

        # 获取实例（自动加载默认配置）
        enhanced_config = EnhancedConfig()

        # 检查特性是否启用
        if enhanced_config.sa_kg.enabled:
            # 使用 SA-KG 特性
            pass

        # 热重载配置
        enhanced_config.reload()
    """

    DEFAULT_CONFIG_PATH = "./conf/enhanced_config.yaml"

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化增强配置

        Args:
            config_path: 配置文件路径，默认使用 DEFAULT_CONFIG_PATH
        """
        self._config_path = (
            config_path
            or os.getenv("STARDOJO_ENHANCED_CONFIG", "").strip()
            or self.DEFAULT_CONFIG_PATH
        )
        self._raw_config: Dict[str, Any] = {}

        # 初始化各模块配置为默认值
        self.sa_kg = SAKGConfig()
        self.dual_brain = DualBrainConfig()
        self.vif = ViFlowConfig()
        self.rtc = RTCConfig()
        self.skill_synthesis = SkillSynthesisConfig()
        self.logging = LoggingConfig()
        self.benchmarking = BenchmarkingConfig()

        # 加载配置文件
        self._load_config()

    def _load_config(self) -> None:
        """加载配置文件"""
        try:
            path = assemble_project_path(self._config_path)
            if not os.path.exists(path):
                _ts_print(f"[EnhancedConfig] 配置文件不存在: {path}，使用默认配置")
                return

            with open(path, 'r', encoding='utf-8') as f:
                self._raw_config = yaml.safe_load(f) or {}

            self._parse_config()
            _ts_print(f"[EnhancedConfig] 配置已加载: {path}")

        except Exception as e:
            _ts_print(f"[EnhancedConfig] 加载配置失败: {e}，使用默认配置")

    def _parse_config(self) -> None:
        """解析配置文件到数据类"""

        # 解析 SA-KG 配置
        sa_kg_cfg = self._raw_config.get('sa_kg', {})
        if sa_kg_cfg:
            embedding = sa_kg_cfg.get('embedding', {})
            retrieval = sa_kg_cfg.get('retrieval', {})
            storage = sa_kg_cfg.get('storage', {})
            write_conditions = sa_kg_cfg.get('write_conditions', {})

            self.sa_kg = SAKGConfig(
                enabled=sa_kg_cfg.get('enabled', False),
                embedding_provider=embedding.get('provider', 'local'),
                embedding_model=embedding.get('model', 'BAAI/bge-base-en-v1.5'),
                embedding_dim=embedding.get('dim', 768),
                similarity_threshold=retrieval.get('similarity_threshold', 0.85),
                top_k=retrieval.get('top_k', 5),
                as_reference_only=retrieval.get('as_reference_only', True),
                storage_backend=storage.get('backend', 'chromadb'),
                persistence_path=storage.get('persistence_path', './cache/sa_kg/'),
                max_entries=storage.get('max_entries', 10000),
                require_reflection_success=write_conditions.get('require_reflection_success', True),
                min_confidence=write_conditions.get('min_confidence', 0.7),
            )

        # 解析双脑系统配置
        dual_brain_cfg = self._raw_config.get('dual_brain', {})
        if dual_brain_cfg:
            little_brain = dual_brain_cfg.get('little_brain', {})
            big_brain = dual_brain_cfg.get('big_brain', {})
            gating = big_brain.get('gating', {})

            self.dual_brain = DualBrainConfig(
                enabled=dual_brain_cfg.get('enabled', False),
                little_brain=LittleBrainConfig(
                    enabled=little_brain.get('enabled', True),
                    retrieval_first=little_brain.get('retrieval_first', True),
                    fallback_to_llm=little_brain.get('fallback_to_llm', True),
                    max_latency_ms=little_brain.get('max_latency_ms', 500),
                ),
                big_brain=BigBrainConfig(
                    enabled=big_brain.get('enabled', True),
                    gating=BigBrainGatingConfig(
                        trigger_on_consecutive_failures=gating.get('trigger_on_consecutive_failures', 3),
                        trigger_on_low_confidence=gating.get('trigger_on_low_confidence', 0.5),
                        trigger_on_new_environment=gating.get('trigger_on_new_environment', True),
                        trigger_on_explicit_request=gating.get('trigger_on_explicit_request', True),
                    ),
                ),
            )

        # 解析 ViF 配置
        vif_cfg = self._raw_config.get('vif', {})
        if vif_cfg:
            cross_brain = vif_cfg.get('cross_brain_transfer', {})
            self.vif = ViFlowConfig(
                enabled=vif_cfg.get('enabled', False),
                inject_detections=vif_cfg.get('inject_detections', True),
                detection_format=vif_cfg.get('detection_format', 'text'),
                max_detections=vif_cfg.get('max_detections', 10),
                cross_brain_transfer_enabled=cross_brain.get('enabled', False),
                include_bboxes=cross_brain.get('include_bboxes', True),
                include_confidence=cross_brain.get('include_confidence', True),
            )

        # 解析 RTC 配置
        rtc_cfg = self._raw_config.get('rtc', {})
        if rtc_cfg:
            chunking = rtc_cfg.get('chunking', {})
            scheduling = rtc_cfg.get('scheduling', {})
            self.rtc = RTCConfig(
                enabled=rtc_cfg.get('enabled', False),
                chunk_duration_ms=chunking.get('chunk_duration_ms', 200),
                target_fps=scheduling.get('target_fps', 30),
            )

        # 解析技能合成配置
        skill_cfg = self._raw_config.get('skill_synthesis', {})
        if skill_cfg:
            promotion = skill_cfg.get('promotion', {})
            sandbox = skill_cfg.get('sandbox', {})
            self.skill_synthesis = SkillSynthesisConfig(
                enabled=skill_cfg.get('enabled', False),
                candidate_threshold=promotion.get('candidate_threshold', 3),
                promoted_threshold=promotion.get('promoted_threshold', 10),
                sandbox_enabled=sandbox.get('enabled', False),
                sandbox_max_retries=sandbox.get('max_retries', 3),
            )

        # 解析日志配置
        logging_cfg = self._raw_config.get('logging', {})
        if logging_cfg:
            self.logging = LoggingConfig(
                enhanced_features=logging_cfg.get('enhanced_features', True),
                sa_kg_queries=logging_cfg.get('sa_kg_queries', True),
                gating_decisions=logging_cfg.get('gating_decisions', True),
                performance_metrics=logging_cfg.get('performance_metrics', True),
            )

        # 解析性能基准配置
        bench_cfg = self._raw_config.get('benchmarking', {})
        if bench_cfg:
            metrics = bench_cfg.get('metrics', {})
            self.benchmarking = BenchmarkingConfig(
                enabled=bench_cfg.get('enabled', False),
                step_latency=metrics.get('step_latency', True),
                llm_call_rate=metrics.get('llm_call_rate', True),
                sa_kg_hit_rate=metrics.get('sa_kg_hit_rate', True),
                task_success_rate=metrics.get('task_success_rate', True),
                output_path=bench_cfg.get('output_path', './cache/benchmarks/'),
            )

    def reload(self, config_path: Optional[str] = None) -> None:
        """
        热重载配置文件

        Args:
            config_path: 可选的新配置文件路径
        """
        if config_path:
            self._config_path = config_path
        self._load_config()

    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        检查指定特性是否启用

        Args:
            feature_name: 特性名称 (sa_kg, dual_brain, vif, rtc, skill_synthesis)

        Returns:
            bool: 特性是否启用
        """
        feature_map = {
            'sa_kg': self.sa_kg.enabled,
            'dual_brain': self.dual_brain.enabled,
            'vif': self.vif.enabled,
            'rtc': self.rtc.enabled,
            'skill_synthesis': self.skill_synthesis.enabled,
        }
        return feature_map.get(feature_name, False)

    def get_enabled_features(self) -> list:
        """获取所有已启用的特性列表"""
        features = []
        if self.sa_kg.enabled:
            features.append('sa_kg')
        if self.dual_brain.enabled:
            features.append('dual_brain')
        if self.vif.enabled:
            features.append('vif')
        if self.rtc.enabled:
            features.append('rtc')
        if self.skill_synthesis.enabled:
            features.append('skill_synthesis')
        return features

    def print_status(self) -> None:
        """打印配置状态摘要"""
        print("\n" + "=" * 60)
        print("Enhanced Configuration Status")
        print("=" * 60)
        print(f"  SA-KG:            {'[ON]  ENABLED' if self.sa_kg.enabled else '[OFF] disabled'}")
        print(f"  Dual Brain:       {'[ON]  ENABLED' if self.dual_brain.enabled else '[OFF] disabled'}")
        print(f"  ViF:              {'[ON]  ENABLED' if self.vif.enabled else '[OFF] disabled'}")
        print(f"  RTC:              {'[ON]  ENABLED' if self.rtc.enabled else '[OFF] disabled'}")
        print(f"  Skill Synthesis:  {'[ON]  ENABLED' if self.skill_synthesis.enabled else '[OFF] disabled'}")
        print(f"  Benchmarking:     {'[ON]  ENABLED' if self.benchmarking.enabled else '[OFF] disabled'}")
        print("=" * 60 + "\n")


# 便捷函数：获取增强配置实例
def get_enhanced_config() -> EnhancedConfig:
    """获取增强配置单例实例"""
    return EnhancedConfig()
