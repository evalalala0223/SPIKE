"""
LangChain integration for Cradle.

This package provides LangChain-based implementations of LLM providers,
replacing the legacy regex-based parsing with type-safe structured outputs.
"""

from cradle.provider.langchain.schemas import (
    ActionPlanningOutput,
    SelfReflectionOutput,
    InformationGatheringOutput,
    TaskInferenceOutput,
    SkillCurationOutput,
    OutputSchema,
    PROVIDER_SCHEMA_MAP,
    get_schema_for_provider,
)

from cradle.provider.langchain.langchain_provider import LangChainLLMProvider

__all__ = [
    # Schemas
    "ActionPlanningOutput",
    "SelfReflectionOutput",
    "InformationGatheringOutput",
    "TaskInferenceOutput",
    "SkillCurationOutput",
    "OutputSchema",
    "PROVIDER_SCHEMA_MAP",
    "get_schema_for_provider",
    # Providers
    "LangChainLLMProvider",
]
