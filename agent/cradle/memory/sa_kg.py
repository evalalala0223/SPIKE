"""
SA-KG: State-Action Knowledge Graph

状态-动作知识图谱，用于存储和检索历史经验。

核心功能：
1. 将游戏状态嵌入为向量 (text-embedding-v4)
2. 存储状态→动作的转移关系
3. 基于相似度检索历史经验
4. 作为参考提供给 LLM，不直接执行

设计原则：
- 与 LocalMemory 并行工作（补充层）
- 检索结果作为 LLM 的参考，保留 LLM 决策权
- 需要 SelfReflection 验证成功后才写入
- 支持时间衰减和成功率统计

Author: Cortex Integration Team
Date: 2026-01-27
"""

import os
import json
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

try:
    import chromadb
    from chromadb.config import Settings  # type: ignore[import-untyped]
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    Settings = None  # type: ignore[assignment]

from cradle.config import Config
from cradle.config.enhanced_config import EnhancedConfig
from cradle.log import Logger
from cradle.memory.base import BaseMemory
from cradle.utils import Singleton


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class StateNode:
    """游戏状态节点"""
    state_id: str                    # 唯一标识
    screenshot_path: str             # 截图路径
    description: str                 # 状态描述（文本）
    timestamp: float                 # 时间戳
    metadata: Dict[str, Any]         # 额外元数据（HP, 位置等）
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ActionEdge:
    """状态转移边（动作）"""
    edge_id: str                     # 唯一标识
    from_state_id: str               # 源状态
    to_state_id: str                 # 目标状态
    action: str                      # 采取的动作
    action_params: Dict[str, Any]    # 动作参数
    success: bool                    # 是否成功
    reward: float                    # 奖励值 (1.0 成功, -0.5 失败)
    timestamp: float                 # 执行时间
    execution_count: int             # 执行次数
    success_count: int               # 成功次数
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count


# =============================================================================
# SA-KG Core
# =============================================================================

class SAKG(BaseMemory, metaclass=Singleton):
    """
    State-Action Knowledge Graph
    
    状态-动作知识图谱，用于存储和检索游戏 Agent 的历史经验。
    
    特点：
    - 使用向量数据库存储状态嵌入
    - 支持基于相似度的经验检索
    - 记录动作的成功率和时间衰减
    - 与 LocalMemory 并行工作
    
    Usage:
        sa_kg = SAKG()
        sa_kg.initialize()
        
        # 检索相似状态
        results = sa_kg.retrieve_similar_states(current_state_desc, top_k=3)
        
        # 添加新经验
        sa_kg.add_experience(state_desc, action, success=True)
    """
    
    def __init__(self):
        """初始化 SA-KG"""
        self.config = Config()
        self.enhanced_config = EnhancedConfig()
        self.logger = Logger()
        
        # 配置
        self.enabled = False
        self.similarity_threshold = 0.85
        self.top_k = 5
        self.as_reference_only = True
        
        # 存储
        self.states: Dict[str, StateNode] = {}
        self.actions: Dict[str, ActionEdge] = {}
        self.client = None
        self.collection = None
        self.embedding_provider = None
        self.namespace = "default"
        self.persistence_path = self.enhanced_config.sa_kg.persistence_path
        
        self._is_initialized = False
    
    
    @staticmethod
    def _sanitize_namespace(namespace: Optional[str]) -> str:
        raw = (namespace or "default").strip().lower()
        sanitized = re.sub(r"[^a-z0-9._-]+", "_", raw)
        sanitized = sanitized.strip("._-")
        return sanitized or "default"


    def _resolve_persistence_path(self, namespace: str) -> str:
        base_path = self.enhanced_config.sa_kg.persistence_path
        return os.path.join(base_path, namespace)


    def initialize(self, embedding_provider: Any = None, namespace: Optional[str] = None) -> None:
        """
        初始化 SA-KG 系统
        
        Args:
            embedding_provider: 嵌入模型提供者（可选，从 Cradle 获取）
        """
        target_namespace = self._sanitize_namespace(
            namespace or getattr(self.config, 'env_short_name', None) or getattr(self.config, 'env_name', None)
        )

        if self._is_initialized and self.namespace == target_namespace:
            if embedding_provider is not None:
                self.embedding_provider = embedding_provider
            return

        if self._is_initialized and self.namespace != target_namespace:
            self.logger.write(f"SA-KG switching namespace: {self.namespace} -> {target_namespace}")
            self.states = {}
            self.actions = {}
            self.client = None
            self.collection = None
            self._is_initialized = False
        
        # EnhancedConfig 在 __init__ 时已自动加载配置，不需要再调用 load()
        
        if not self.enhanced_config.sa_kg.enabled:
            self.logger.write("SA-KG is disabled in config")
            return
        
        self.enabled = True
        self.similarity_threshold = self.enhanced_config.sa_kg.similarity_threshold
        self.top_k = self.enhanced_config.sa_kg.top_k
        self.as_reference_only = self.enhanced_config.sa_kg.as_reference_only
        self.namespace = target_namespace
        self.persistence_path = self._resolve_persistence_path(self.namespace)

        # 设置嵌入提供者（在 chromadb 检查之前，确保降级模式也能使用）
        self.embedding_provider = embedding_provider

        # 检查依赖
        if not CHROMADB_AVAILABLE:
            self.logger.warn("Chromadb not available, SA-KG entering degraded text mode")
            # Keep SA-KG enabled in degraded mode so it can still serve as feature layer.
            self.enabled = True
            self._load_metadata()
            self._is_initialized = True
            return
        
        if Settings is None:
            self.logger.warn("Chromadb Settings is unavailable, SA-KG entering degraded text mode")
            self.enabled = True
            self._load_metadata()
            self._is_initialized = True
            return
        
        # 初始化向量数据库
        try:
            persist_dir = self.persistence_path
            self.logger.write(f"[SA-KG Init] Preparing persistence dir: {persist_dir}")
            os.makedirs(persist_dir, exist_ok=True)

            self.logger.write("[SA-KG Init] Creating chromadb PersistentClient...")
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )

            self.logger.write("[SA-KG Init] PersistentClient ready, opening collection...")
            # 获取或创建 collection
            self.collection = self.client.get_or_create_collection(
                name="state_action_kg",
                metadata={"description": "State-Action Knowledge Graph for game agent"}
            )

            self.logger.write("[SA-KG Init] Collection ready, counting existing states...")
            self.logger.write(f"SA-KG initialized with {self.collection.count()} existing states (namespace={self.namespace})")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SA-KG: {e}")
            self.enabled = False
            return

        # 加载持久化数据
        self.logger.write("[SA-KG Init] Loading metadata...")
        self._load_metadata()
        self.logger.write("[SA-KG Init] Metadata loaded")
        
        self._is_initialized = True
        self.logger.write("SA-KG initialized successfully")
    
    
    def retrieve_similar_states(
        self,
        state_description: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        检索相似的历史状态
        
        Args:
            state_description: 当前状态描述
            top_k: 返回前 K 个结果
            threshold: 相似度阈值
        
        Returns:
            相似状态列表，每项包含：
            - state: StateNode
            - action: ActionEdge
            - similarity: float
        """
        if not self.enabled or not self._is_initialized:
            return []
        
        top_k = top_k or self.top_k
        threshold = threshold or self.similarity_threshold
        
        try:
            # Vector search requires both embedding provider and collection
            if self.embedding_provider and self.collection is not None:
                query_embedding = self.embedding_provider.embed_query(state_description)
            else:
                # Fallback to text search (no vector DB or no embedding provider)
                if not self.embedding_provider:
                    self.logger.debug("No embedding provider, using text search")
                return self._text_based_search(state_description, top_k)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['metadatas', 'distances']
            )
            
            similar_states = []
            
            if results['ids'] and len(results['ids'][0]) > 0 and results['distances']:
                for idx, state_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][idx]
                    similarity = 1.0 - distance  # 转换为相似度
                    
                    # 过滤低于阈值的结果
                    if similarity < threshold:
                        continue
                    
                    # 获取状态和动作
                    if state_id in self.states:
                        state = self.states[state_id]
                        # 查找从该状态出发的最佳动作
                        best_action = self._get_best_action_from_state(state_id)
                        
                        if best_action:
                            similar_states.append({
                                'state': state,
                                'action': best_action,
                                'similarity': similarity,
                                'success_rate': best_action.success_rate
                            })
            
            # 按相似度排序
            similar_states.sort(key=lambda x: x['similarity'], reverse=True)
            
            if self.enhanced_config.logging.sa_kg_queries:
                self.logger.write(f"SA-KG retrieved {len(similar_states)} similar states")
            
            return similar_states
            
        except Exception as e:
            self.logger.error(f"SA-KG retrieval error: {e}")
            return []
    
    
    def add_experience(
        self,
        state_description: str,
        screenshot_path: str,
        action: str,
        action_params: Dict[str, Any],
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        添加新的经验（状态→动作）
        
        Args:
            state_description: 状态描述
            screenshot_path: 截图路径
            action: 执行的动作
            action_params: 动作参数
            success: 是否成功
            metadata: 额外元数据
        """
        if not self.enabled or not self._is_initialized:
            return
        
        # 检查写入条件
        if self.enhanced_config.sa_kg.require_reflection_success and not success:
            self.logger.debug("SA-KG: Skip failed action (require_reflection_success=True)")
            return
        
        try:
            # 创建状态节点
            state_id = f"state_{int(time.time() * 1000)}"
            state = StateNode(
                state_id=state_id,
                screenshot_path=screenshot_path,
                description=state_description,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            # 生成嵌入并添加到 Chromadb（可选）
            if self.embedding_provider and self.collection is not None:
                embedding = self.embedding_provider.embed_query(state_description)
                chroma_metadata = {
                    "state_id": state.state_id,
                    "screenshot_path": state.screenshot_path,
                    "description": state.description,
                    "timestamp": state.timestamp,
                    "metadata_json": json.dumps(state.metadata, ensure_ascii=False, sort_keys=True)
                }
                self.collection.add(
                    ids=[state_id],
                    embeddings=[embedding],
                    metadatas=[chroma_metadata]
                )
            else:
                self.logger.debug("SA-KG add_experience running without embeddings/chromadb")
            
            # 添加到内存
            self.states[state_id] = state
            
            # 创建动作边
            edge_id = f"edge_{int(time.time() * 1000)}"
            reward = 1.0 if success else -0.5
            
            action_edge = ActionEdge(
                edge_id=edge_id,
                from_state_id=state_id,
                to_state_id="",  # 暂时留空
                action=action,
                action_params=action_params,
                success=success,
                reward=reward,
                timestamp=time.time(),
                execution_count=1,
                success_count=1 if success else 0
            )
            
            self.actions[edge_id] = action_edge
            
            # 保存元数据
            self._save_metadata()
            
            if self.enhanced_config.logging.sa_kg_queries:
                self.logger.write(f"SA-KG added experience: {action} ({'success' if success else 'failed'})")
            
        except Exception as e:
            self.logger.error(f"SA-KG add experience error: {e}")
    
    
    def _get_best_action_from_state(self, state_id: str) -> Optional[ActionEdge]:
        """获取从给定状态出发的最佳动作"""
        actions = [a for a in self.actions.values() if a.from_state_id == state_id]
        
        if not actions:
            return None
        
        # 按成功率和执行次数排序
        actions.sort(key=lambda a: (a.success_rate, a.execution_count), reverse=True)
        return actions[0]
    
    
    def _text_based_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """基于文本的简单搜索（回退方案）"""
        # 简单的关键词匹配
        results = []
        for state in self.states.values():
            if query.lower() in state.description.lower():
                action = self._get_best_action_from_state(state.state_id)
                if action:
                    results.append({
                        'state': state,
                        'action': action,
                        'similarity': 0.8,  # 固定相似度
                        'success_rate': action.success_rate
                    })
        
        return results[:top_k]
    
    
    def _load_metadata(self) -> None:
        """加载持久化的元数据"""
        metadata_path = os.path.join(
            self.persistence_path,
            "sa_kg_metadata.json"
        )
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 恢复状态
                for state_dict in data.get('states', []):
                    state = StateNode(**state_dict)
                    self.states[state.state_id] = state
                
                # 恢复动作
                for action_dict in data.get('actions', []):
                    action = ActionEdge(**action_dict)
                    self.actions[action.edge_id] = action
                
                self.logger.write(f"Loaded {len(self.states)} states and {len(self.actions)} actions")
                
            except Exception as e:
                self.logger.error(f"Failed to load SA-KG metadata: {e}")
    
    
    def _save_metadata(self) -> None:
        """保存元数据到磁盘"""
        metadata_path = os.path.join(
            self.persistence_path,
            "sa_kg_metadata.json"
        )
        
        try:
            data = {
                'states': [s.to_dict() for s in self.states.values()],
                'actions': [a.to_dict() for a in self.actions.values()],
                'saved_at': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"Failed to save SA-KG metadata: {e}")
    
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.enabled:
            return {'enabled': False}
        
        total_executions = sum(a.execution_count for a in self.actions.values())
        total_successes = sum(a.success_count for a in self.actions.values())
        
        return {
            'enabled': True,
            'total_states': len(self.states),
            'total_actions': len(self.actions),
            'total_executions': total_executions,
            'total_successes': total_successes,
            'overall_success_rate': total_successes / total_executions if total_executions > 0 else 0.0,
            'similarity_threshold': self.similarity_threshold,
            'as_reference_only': self.as_reference_only,
        }
    
    
    # =============================================================================
    # BaseMemory Interface (保持兼容)
    # =============================================================================
    
    def add(self, **kwargs) -> None:
        """BaseMemory 接口兼容"""
        pass
    
    def similarity_search(self, data: str, top_k: int, **kwargs) -> List[Any]:
        """BaseMemory 接口兼容"""
        return self.retrieve_similar_states(data, top_k=top_k)
    
    def add_recent_history(self, key: str, info: Any) -> None:
        """BaseMemory 接口兼容（不实现）"""
        pass
    
    def get_recent_history(self, key: str, k: int = 1) -> List[Any]:
        """BaseMemory 接口兼容（不实现）"""
        return []
    
    def add_summarization(self, hidden_state: str) -> None:
        """BaseMemory 接口兼容（不实现）"""
        pass
    
    def get_summarization(self) -> str:
        """BaseMemory 接口兼容（不实现）"""
        return ""
    
    def load(self) -> None:
        """加载持久化数据"""
        self._load_metadata()
    
    def save(self) -> None:
        """保存数据到持久化"""
        self._save_metadata()
