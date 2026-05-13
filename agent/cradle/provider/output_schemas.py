"""
Pydantic schemas for structured LLM outputs - Phase 2.1 Streaming Support

这些schemas用于流式LLM输出的解析，确保类型安全和提前终止。
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class InfoGatheringSchema(BaseModel):
    """Information Gathering输出结构"""
    image_description: str = Field(description="当前截图的详细描述")
    object_locations: Optional[List[Dict[str, Any]]] = Field(default=None, description="识别的对象位置")
    
    class Config:
        extra = "allow"  # 允许额外字段，保持向后兼容


class SelfReflectionSchema(BaseModel):
    """Self Reflection输出结构"""
    self_reflection: str = Field(description="对当前情况的反思和分析")
    success: Optional[bool] = Field(default=None, description="当前任务是否成功")
    errors: Optional[List[str]] = Field(default=None, description="检测到的错误")
    
    class Config:
        extra = "allow"


class TaskInferenceSchema(BaseModel):
    """Task Inference输出结构"""
    task_guidance: str = Field(description="任务指导和策略")
    sub_tasks: Optional[List[str]] = Field(default=None, description="子任务列表")
    
    class Config:
        extra = "allow"


class ActionPlanningSchema(BaseModel):
    """Action Planning输出结构"""
    reasoning: str = Field(description="行动推理过程")
    action_type: str = Field(description="行动类型")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="行动参数")
    
    class Config:
        extra = "allow"


class SkillExecuteSchema(BaseModel):
    """Skill Execute输出结构"""
    skill_name: str = Field(description="技能名称")
    skill_params: Optional[Dict[str, Any]] = Field(default=None, description="技能参数")
    execution_result: Optional[str] = Field(default=None, description="执行结果")
    
    class Config:
        extra = "allow"
