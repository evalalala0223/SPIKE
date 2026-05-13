import base64
import time
import json
import importlib
import inspect
from typing import Type, AnyStr, Any

import numpy as np
import dill
from dataclasses import dataclass
from dataclass_wizard import JSONWizard
from dataclass_wizard.abstractions import W
from dataclass_wizard.type_def import JSONObject, Encoder

from stardojo.config import Config

config = Config()


@dataclass
class Skill(JSONWizard):

    skill_name: str
    skill_function: Any
    skill_embedding: np.ndarray
    skill_code: str
    skill_code_base64: str


    def __call__(self, *args, **kwargs):
        return self.skill_function(*args, **kwargs)


    @classmethod
    def from_dict(cls: Type[W], o: JSONObject) -> W:
        skill_function = None
        skill_module = str(o.get('skill_module', '') or '').strip()
        skill_function_name = str(
            o.get('skill_function_name', '')
            or o.get('skill_name', '')
            or ''
        ).strip()
        if skill_module and skill_function_name:
            try:
                resolved = getattr(importlib.import_module(skill_module), skill_function_name)
                if hasattr(resolved, "skill_function"):
                    resolved = getattr(resolved, "skill_function")
                skill_function = resolved
            except Exception:
                skill_function = None

        if skill_function is None:
            skill_function_hex = str(o.get('skill_function', '') or '').strip()
            if skill_function_hex:
                skill_function = dill.loads(bytes.fromhex(skill_function_hex))
            else:
                namespace = {}
                exec(str(o.get('skill_code', '') or ''), namespace)
                skill_function = namespace.get(skill_function_name)
                if skill_function is None:
                    raise ValueError(f"Unable to reconstruct skill function for {skill_function_name}")

        skill_embedding = np.frombuffer(base64.b64decode(o['skill_embedding']), dtype=np.float64)

        return cls(
            skill_name=o['skill_name'],
            skill_function=skill_function,
            skill_embedding=skill_embedding,
            skill_code=o['skill_code'],
            skill_code_base64=o['skill_code_base64']
        )


    def to_dict(self) -> JSONObject:
        skill_function_hex = ""
        skill_module = ""
        skill_function_name = str(self.skill_name or "").strip()

        if inspect.isfunction(self.skill_function):
            skill_module = str(getattr(self.skill_function, "__module__", "") or "").strip()
            fallback_name = str(getattr(self.skill_function, "__name__", "") or "").strip()
            if fallback_name:
                skill_function_name = fallback_name

            if skill_module and skill_function_name:
                try:
                    resolved = getattr(importlib.import_module(skill_module), skill_function_name, None)
                    if hasattr(resolved, "skill_function"):
                        resolved = getattr(resolved, "skill_function")
                    if resolved is not self.skill_function:
                        skill_function_hex = dill.dumps(self.skill_function).hex()
                except Exception:
                    skill_function_hex = dill.dumps(self.skill_function).hex()
            else:
                skill_function_hex = dill.dumps(self.skill_function).hex()
        else:
            skill_function_hex = dill.dumps(self.skill_function).hex()

        skill_embedding_base64 = base64.b64encode(self.skill_embedding).decode('utf-8')

        return {
            'skill_name': self.skill_name,
            'skill_function': skill_function_hex,
            'skill_module': skill_module,
            'skill_function_name': skill_function_name,
            'skill_embedding': skill_embedding_base64,
            'skill_code': self.skill_code,
            'skill_code_base64': self.skill_code_base64
        }


    def to_json(self: W, *,
                encoder: Encoder = json.dumps,
                **encoder_kwargs) -> AnyStr:
        return json.dumps(self.to_dict(), **encoder_kwargs)


    @classmethod
    def from_json(cls: Type[W], s: AnyStr, *,
                  decoder: Any = json.loads,
                  **decoder_kwargs) -> W:
        return cls.from_dict(json.loads(s, **decoder_kwargs))


def post_skill_wait(wait_time = config.DEFAULT_POST_ACTION_WAIT_TIME):
    """Wait for skill to finish. Like if there is an animation"""
    time.sleep(wait_time)
