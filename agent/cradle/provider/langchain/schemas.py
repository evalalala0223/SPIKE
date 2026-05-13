"""
Pydantic schemas for structured LLM outputs.

This module defines type-safe output schemas for all LLM-based providers,
replacing the fragile regex-based parsing in parse_semi_formatted_text().

Design principles:
1. Explicit field descriptions to guide LLM output generation
2. Validators for automatic data cleaning (e.g., removing code blocks)
3. Support for both English and Chinese field names via aliases
4. Backward compatible with legacy output format
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import re


class ActionPlanningOutput(BaseModel):
    """
    Output schema for action planning providers.

    Used by:
    - ActionPlanningProvider
    - RDR2ActionPlanningProvider
    - StardewActionPlanningProvider

    Example output:
    {
        "reasoning": "1. The toolbar shows...\n2. The character is at...",
        "actions": ["move(x=1, y=0)", "use(direction=\"down\")"]
    }
    """

    reasoning: str = Field(
        description=(
            "Step-by-step reasoning for the next action. "
            "Analyze: 1) Toolbar status, 2) Current position, 3) Previous action results, "
            "4) Whether blocked, 5) Most suitable next action. "
            "Must be detailed and answer all analysis questions."
        ),
        min_length=10,
    )

    actions: List[str] = Field(
        description=(
            "Python function calls to execute in the game. "
            "Format: function_name(param1=value1, param2=value2). "
            "Max 2 actions. Must exist in valid action set. "
            "Examples: ['move(x=1, y=0)', 'use(direction=\"down\")']"
        ),
        min_length=1,
        max_length=2,
    )

    @field_validator('actions', mode='before')
    @classmethod
    def clean_actions(cls, v):
        """
        Clean actions field to handle various LLM output formats.

        Handles:
        - Code blocks: ```python move(x=-1, y=0) ```
        - Multiple actions on one line: move(x=-1, y=0); use(direction="left")
        - Comments: move(x=-1, y=0)  # Move to the left
        - Empty strings
        """
        if isinstance(v, str):
            # Remove code block markers
            v = re.sub(r'```(?:python)?\s*', '', v)
            v = re.sub(r'```', '', v)

            # Split by newlines or semicolons
            actions = re.split(r'[\n;]+', v)
        elif isinstance(v, list):
            actions = v
        else:
            raise ValueError(f"Actions must be string or list, got {type(v)}")

        # Clean each action
        cleaned_actions = []
        for action in actions:
            # Remove comments
            action = action.split('#')[0].strip()

            # Skip empty
            if not action:
                continue

            # Validate it looks like a function call
            if '(' not in action or ')' not in action:
                # Try to fix common issues
                if action and not action.endswith(')'):
                    action = f"{action}()"

            cleaned_actions.append(action)

        if not cleaned_actions:
            raise ValueError("No valid actions found after cleaning")

        return cleaned_actions

    @model_validator(mode='after')
    def validate_action_format(self):
        """Validate that actions follow correct Python function call format."""
        for action in self.actions:
            # Check for basic function call pattern: name(...)
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)$', action):
                raise ValueError(
                    f"Invalid action format: '{action}'. "
                    f"Expected: function_name(param=value)"
                )
        return self

    class Config:
        # Allow aliases for Chinese field names
        populate_by_name = True


class SelfReflectionOutput(BaseModel):
    """
    Output schema for self-reflection providers.

    Used by:
    - SelfReflectionProvider
    - RDR2SelfReflectionProvider
    - StardewSelfReflectionProvider

    Example output:
    {
        "reasoning": "The last action was move(x=1, y=0)...",
        "success": true
    }
    """

    reasoning: str = Field(
        description=(
            "Reflection on the last action's outcome. "
            "Analyze: 1) What was attempted, 2) What actually happened (from screenshot), "
            "3) Why it succeeded or failed, 4) Lessons learned for next time."
        ),
        min_length=10,
    )

    success: bool = Field(
        description=(
            "Whether the last action successfully achieved its intended goal. "
            "true = action completed as expected. "
            "false = action failed or was blocked."
        )
    )

    @field_validator('success', mode='before')
    @classmethod
    def parse_success(cls, v):
        """
        Parse success field from various formats.

        Handles:
        - Boolean: true, false
        - String: "true", "True", "yes", "successful", "1"
        - Case insensitive
        """
        if isinstance(v, bool):
            return v

        if isinstance(v, str):
            v_lower = v.lower().strip()
            return v_lower in ("true", "yes", "1", "successful", "succeeded", "success")

        if isinstance(v, int):
            return v == 1

        # Default to False if unclear
        return False

    class Config:
        populate_by_name = True


class InformationGatheringOutput(BaseModel):
    """
    Output schema for information gathering providers.

    Used by:
    - InformationGatheringProvider

    Example output:
    {
        "image_description": "The screenshot shows the character standing in front of a tree...",
        "toolbar_information": "Slot 1: Axe (selected), Slot 2: Hoe, Slot 3: Empty"
    }
    """

    image_description: str = Field(
        description=(
            "Detailed description of the game screenshot. "
            "Include: 1) Character position and orientation, 2) Nearby objects and NPCs, "
            "3) UI elements (health, stamina, time), 4) Environment (indoor/outdoor, weather)."
        ),
        min_length=20,
    )

    toolbar_information: Optional[str] = Field(
        default="",
        description=(
            "Contents of the toolbar/hotbar. "
            "Format: Slot N: Item Name (selected/not selected). "
            "Example: 'Slot 1: Axe (selected), Slot 2: Seeds, Slot 3: Empty'"
        )
    )

    class Config:
        populate_by_name = True


class TaskInferenceOutput(BaseModel):
    """
    Output schema for task inference providers.

    Used by:
    - TaskInferenceProvider

    Example output:
    {
        "reasoning": "Based on the current situation...",
        "task_description": "Cut down the tree blocking the path",
        "subtask_description": "Equip the axe and move to the tree"
    }
    """

    reasoning: str = Field(
        description="Analysis of current situation and task decomposition logic.",
        min_length=10,
    )

    task_description: str = Field(
        description="High-level task to achieve the overall goal.",
        min_length=5,
    )

    subtask_description: str = Field(
        description="Immediate subtask to work on next.",
        min_length=5,
    )

    class Config:
        populate_by_name = True


class SkillCurationOutput(BaseModel):
    """
    Output schema for skill curation providers.

    Used by:
    - SkillCurationProvider

    Example output:
    {
        "reasoning": "The task requires tree cutting, which needs the axe skill...",
        "skills": ["use_tool", "move_to_object"]
    }
    """

    reasoning: str = Field(
        description="Reasoning for selecting these specific skills from the skill library.",
        min_length=10,
    )

    skills: List[str] = Field(
        description="List of skill names relevant to the current task.",
        min_length=1,
    )

    class Config:
        populate_by_name = True


# Type alias for all output schemas
OutputSchema = Union[
    ActionPlanningOutput,
    SelfReflectionOutput,
    InformationGatheringOutput,
    TaskInferenceOutput,
    SkillCurationOutput
]


# Mapping from provider type to schema
PROVIDER_SCHEMA_MAP = {
    "action_planning": ActionPlanningOutput,
    "self_reflection": SelfReflectionOutput,
    "information_gathering": InformationGatheringOutput,
    "task_inference": TaskInferenceOutput,
    "skill_curation": SkillCurationOutput,
}


def get_schema_for_provider(provider_type: str) -> type[BaseModel]:
    """
    Get the appropriate Pydantic schema for a provider type.

    Args:
        provider_type: Type of provider (e.g., "action_planning")

    Returns:
        Pydantic schema class

    Raises:
        ValueError: If provider type is unknown

    Example:
        >>> schema = get_schema_for_provider("action_planning")
        >>> schema
        <class 'ActionPlanningOutput'>
    """
    if provider_type not in PROVIDER_SCHEMA_MAP:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Valid types: {list(PROVIDER_SCHEMA_MAP.keys())}"
        )
    return PROVIDER_SCHEMA_MAP[provider_type]
