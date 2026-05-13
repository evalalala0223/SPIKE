from importlib import import_module
from typing import Any

from .base import BaseProvider
from .base.base_provider import BaseModuleProvider
from .base.base_embedding import EmbeddingProvider
from .base.base_llm import LLMProvider


_LAZY_EXPORTS = {
    "OpenAIProvider": ".llm.openai",
    "CircleDetectProvider": ".circle_detector",
    "SamProvider": ".sam_provider",
    "GdProvider": ".object_detect.gd_provider",
    "VideoOCRExtractorProvider": ".video.video_ocr_extractor",
    "VideoRecordProvider": ".video.video_recorder",
    "VideoFrameExtractorProvider": ".video.video_frame_extractor",
    "VideoClipProvider": ".video.video_clip",
    "ActionPlanningPreprocessProvider": ".process.action_planning",
    "ActionPlanningPostprocessProvider": ".process.action_planning",
    "RDR2ActionPlanningPreprocessProvider": ".process.action_planning",
    "RDR2ActionPlanningPostprocessProvider": ".process.action_planning",
    "StardewActionPlanningPreprocessProvider": ".process.action_planning",
    "StardewActionPlanningPostprocessProvider": ".process.action_planning",
    "SkylinesActionPlanningPreprocessProvider": ".process.action_planning",
    "InformationGatheringPreprocessProvider": ".process.information_gathering",
    "InformationGatheringPostprocessProvider": ".process.information_gathering",
    "RDR2InformationGatheringPreprocessProvider": ".process.information_gathering",
    "RDR2InformationGatheringPostprocessProvider": ".process.information_gathering",
    "StardewInformationGatheringPreprocessProvider": ".process.information_gathering",
    "StardewInformationGatheringPostprocessProvider": ".process.information_gathering",
    "SelfReflectionPreprocessProvider": ".process.self_reflection",
    "SelfReflectionPostprocessProvider": ".process.self_reflection",
    "RDR2SelfReflectionPreprocessProvider": ".process.self_reflection",
    "RDR2SelfReflectionPostprocessProvider": ".process.self_reflection",
    "StardewSelfReflectionPreprocessProvider": ".process.self_reflection",
    "StardewSelfReflectionPostprocessProvider": ".process.self_reflection",
    "SkylinesSelfReflectionPreprocessProvider": ".process.self_reflection",
    "TaskInferencePreprocessProvider": ".process.task_inference",
    "TaskInferencePostprocessProvider": ".process.task_inference",
    "RDR2TaskInferencePreprocessProvider": ".process.task_inference",
    "RDR2TaskInferencePostprocessProvider": ".process.task_inference",
    "StardewTaskInferencePreprocessProvider": ".process.task_inference",
    "StardewTaskInferencePostprocessProvider": ".process.task_inference",
    "SkylinesTaskInferencePreprocessProvider": ".process.task_inference",
    "RDR2InformationGatheringProvider": ".module.information_gathering",
    "InformationGatheringProvider": ".module.information_gathering",
    "StardewInformationGatheringProvider": ".module.information_gathering",
    "RDR2SelfReflectionProvider": ".module.self_reflection",
    "SelfReflectionProvider": ".module.self_reflection",
    "StardewSelfReflectionProvider": ".module.self_reflection",
    "RDR2ActionPlanningProvider": ".module.action_planning",
    "ActionPlanningProvider": ".module.action_planning",
    "StardewActionPlanningProvider": ".module.action_planning",
    "RDR2TaskInferenceProvider": ".module.task_inference",
    "TaskInferenceProvider": ".module.task_inference",
    "StardewTaskInferenceProvider": ".module.task_inference",
    "RDR2SkillCurationProvider": ".module.skill_curation",
    "SkillCurationProvider": ".module.skill_curation",
    "SkillExecuteProvider": ".execute.skill_execute",
    "AugmentProvider": ".augment.augment",
    "CoordinatesProvider": ".others.coordinates",
    "TaskGuidanceProvider": ".others.task_guidance",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "BaseProvider",
    "BaseModuleProvider",
    "LLMProvider",
    "EmbeddingProvider",
    *_LAZY_EXPORTS.keys(),
]
