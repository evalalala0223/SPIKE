import json
import re
from typing import Optional, Tuple, Dict
from collections import OrderedDict
from collections.abc import Mapping, Iterable
from datetime import datetime

import torch

from cradle import constants
from cradle.utils.string_utils import contains_punctuation, is_numbered_bullet_list_item


def load_json(file_path):
    with open(file_path, mode='r', encoding='utf8') as fp:
        json_dict = json.load(fp)
        return json_dict


def serialize_data(item):
    """Recursively convert non-serializable items in the dictionary."""

    if isinstance(item, (str, int, float, bool)):
        return item
    elif isinstance(item, torch.Tensor):
        # Check if the tensor is 0-d (a scalar)
        if item.dim() == 0:
            # Convert scalar tensor to a Python number
            return item.item()
        else:
            # Check if tensor is on a GPU, move to CPU first
            if item.is_cuda:
                item = item.cpu()
            # Convert tensor to a list
            return item.numpy().tolist()
    elif isinstance(item, datetime):
        return item.isoformat()

    if isinstance(item, Mapping):
        return {key: serialize_data(value) for key, value in item.items()}
    elif isinstance(item, Iterable):
        return [serialize_data(element) for element in item]
    elif isinstance(item, JsonFrameStructure):  # Assuming JSONStructure needs to be handled
        return item.to_dict()  # Assuming JSONStructure objects have a to_dict method or similar
    return item


def save_json(file_path, json_dict, indent=-1):
    processed_data = serialize_data(json_dict)
    with open(file_path, mode='w', encoding='utf8') as fp:
        if indent == -1:
            json.dump(processed_data, fp, ensure_ascii=False)
        else:
            json.dump(processed_data, fp, ensure_ascii=False, indent=indent)


def check_json(json_string):
    try:
        json.loads(json_string)
    except (json.JSONDecodeError, ValueError, TypeError):
        return False
    return True


def refine_json(json_string):
    patterns = [
        r"^`+json(.*?)`+", # ```json content```, ```json content``, ...
        r"^json(.*?)", # json content
        r"^json(.*?)\." # json content.
    ]

    for pattern in patterns:
        match = re.search(pattern, json_string, re.DOTALL)
        if match:
            json_string = match.group(1)
            if check_json(json_string):
                return json_string
    return json_string


def parse_semi_formatted_json(json_string):

    obj = None

    try:
        response = refine_json(json_string)
        obj = json.loads(response)

    except Exception as e:
        raise ValueError(f"Error in processing json: {e}. Object was: {json_string}.") from e

    return obj


# Key aliases to normalize common variations in LLM responses
_KEY_ALIASES = {
    "action": "actions",
    "actions": "actions",
    "next_action": "actions",
    "next_actions": "actions",
    "selected_action": "actions",
    "执行动作": "actions",      # Chinese support
    "动作": "actions",          # Chinese support
    "行动": "actions",          # Chinese support
    "reason": "reasoning",
    "reasoning": "reasoning",
    "analysis": "reasoning",
    "thought": "reasoning",
    "thoughts": "reasoning",
    "thinking": "reasoning",
    "推理": "reasoning",        # Chinese support
    "分析": "reasoning",        # Chinese support
    "思考": "reasoning",        # Chinese support
    "answer": "actions",
}


def _extract_actions_from_code_blocks(text: str) -> list:
    """
    Fallback: Extract function calls from ```python code blocks.
    Used when normal parsing fails to find actions.
    """
    actions = []

    # Pattern to match ```python ... ``` code blocks
    code_pattern = r'```(?:python)?\s*([\s\S]*?)```'
    matches = re.findall(code_pattern, text, re.IGNORECASE)

    for match in matches:
        lines = [l.strip() for l in match.strip().split('\n') if l.strip()]
        for line in lines:
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            # Check if it looks like a function call (contains parentheses)
            if '(' in line and ')' in line:
                # Remove inline comments
                line = line.split('#')[0].strip()
                if line:
                    actions.append(line)

    return actions


def _extract_actions_from_reasoning(text: str, skill_keywords: list = None) -> list:
    """
    Last resort fallback: Extract function calls from reasoning text.
    Only extracts calls that match known skill patterns.
    """
    if skill_keywords is None:
        skill_keywords = [
            'move_', 'use_', 'select_', 'get_out', 'open_', 'close_',
            'click_', 'press_', 'hold_', 'release_', 'wait_', 'do_action',
            'nop', 'interact_',
            # Stardew Valley actions (without trailing underscore)
            'move(', 'use(', 'interact(', 'choose_item(', 'choose_option(',
            'attach_item(', 'unattach_item(', 'craft(', 'menu(',
        ]

    actions = []

    # Pattern to match function calls like func_name(args)
    func_pattern = r'(\w+\([^)]*\))'
    matches = re.findall(func_pattern, text)

    for match in matches:
        # Check if it matches known skill patterns
        if any(kw in match.lower() for kw in skill_keywords):
            actions.append(match)

    return actions


def _extract_actions_from_natural_language(text: str) -> list:
    """
    Fallback parser for free-form reasoning that never emitted an explicit
    `Actions:` block but still described a concrete next action.

    Only extracts highly constrained action forms to avoid inventing actions:
    - choose/select/equip/switch to ... slot N  -> choose_item(slot_index=N)
    - move up/down/left/right [by N]            -> move(x=?, y=?)
    """
    source = str(text or "")
    if not source:
        return []

    candidate_regions: list[str] = []
    decision_pattern = re.compile(
        r"(?is)(?:therefore|thus|so|hence|the immediate (?:grounded )?action is|the immediate next action is|the next action is|the logical next step is|the most logical next step is|let me start by|i should first|i should|i will|my next action is|the action is)\b.*$"
    )
    decision_matches = list(decision_pattern.finditer(source))
    if decision_matches:
        candidate_regions.append(decision_matches[-1].group(0))

    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", source) if segment.strip()]
    if sentences:
        candidate_regions.append(" ".join(sentences[-3:]))
        candidate_regions.append(sentences[-1])

    if not candidate_regions:
        candidate_regions.append(source)

    candidates: list[tuple[int, str]] = []

    def _append(position: int, action: str) -> None:
        if not action:
            return
        candidates.append((position, action))

    slot_pattern = re.compile(
        r"\b(?:choose|choosing|select|selecting|equip|equipping|switch(?:ing)?\s+to|pick|picking)\b[^\n\r.]{0,120}?\bslot(?:_index)?\s*(\d+)\b",
        re.IGNORECASE,
    )
    move_pattern = re.compile(
        r"\bmove\s+(up|down|left|right)\b(?:[^\n\r.]{0,40}?\bby\s+(\d+))?",
        re.IGNORECASE,
    )

    for region in candidate_regions:
        for match in slot_pattern.finditer(region):
            slot_index = int(match.group(1))
            _append(match.start(), f"choose_item(slot_index={slot_index})")

        for match in move_pattern.finditer(region):
            prefix = region[max(0, match.start() - 24):match.start()].lower()
            if re.search(r"(?:do\s+not|don't|not|avoid)\s+$", prefix):
                continue
            direction = match.group(1).lower()
            magnitude = int(match.group(2) or 1)
            magnitude = max(1, magnitude)
            if direction == "up":
                action = f"move(x=0, y=-{magnitude})"
            elif direction == "down":
                action = f"move(x=0, y={magnitude})"
            elif direction == "left":
                action = f"move(x=-{magnitude}, y=0)"
            else:
                action = f"move(x={magnitude}, y=0)"
            _append(match.start(), action)

        if candidates:
            break

    candidates.sort(key=lambda item: item[0])
    return [action for _, action in candidates]


def _is_line_key_candidate(line: str) -> Tuple[bool, Optional[str]]:

    result = False
    likely_key = None

    # Strip common markdown formatting that might wrap the key
    cleaned_line = line.strip()
    cleaned_line = cleaned_line.lstrip('#').strip()  # Remove heading markers

    if cleaned_line.endswith(':'):

        # Cannot have other previous punctuation, except if it's a numbered bullet list item
        num_idx = is_numbered_bullet_list_item(cleaned_line)

        post_num_idx = 0
        if num_idx > -1:
            post_num_idx = num_idx

        likely_key = cleaned_line[post_num_idx:-1].strip()
        result = not contains_punctuation(likely_key)

        # Normalize key using aliases
        if result and likely_key:
            normalized_key = likely_key.replace(" ", "_").lower()
            if normalized_key in _KEY_ALIASES:
                likely_key = _KEY_ALIASES[normalized_key]

    return result, likely_key


### Parses the semi-formatted text from model response
def parse_semi_formatted_text(text):

    lines = text.split('\n')

    lines = [line.rstrip() for line in lines if line.rstrip()]
    result_dict = {}
    current_key = None
    current_value = []
    parsed_data = []
    in_code_flag = False

    for line in lines:

        line = line.replace("**", "").replace("###", "").replace("##", "") # Remove unnecessary in Markdown formatting

        is_key, key_candidate = _is_line_key_candidate(line)

        # Check if the line indicates a new key
        if  is_key and in_code_flag == False:

            # If there's a previous key, process its values
            if current_key and current_key == constants.ACTION_GUIDANCE:
                result_dict[current_key] = parsed_data
            elif current_key:
                result_dict[current_key] = '\n'.join(current_value).strip()

            try:
                current_key = key_candidate.replace(" ", "_").lower()
            except Exception as e:
                # logger.error(f"Response is not in the correct format: {e}\nReceived text was: {text}")
                raise

            current_value = []
            parsed_data = []
        else:
            if current_key == constants.ACTION_GUIDANCE:
                in_code_flag = True
                if line.strip() == '```':
                    if current_value:  # Process previous code block and description
                        entry = {"code": '\n'.join(current_value[1:])}
                        parsed_data.append(entry)
                        current_value = []
                    in_code_flag = False
                else:
                    current_value.append(line)
                    if line.strip().lower() == 'null':
                        in_code_flag = False
            else:
                in_code_flag = False
                line = line.strip()
                current_value.append(line)

    # Process the last key
    if current_key == constants.ACTION_GUIDANCE:
        if current_value:  # Process the last code block and description
            entry = {"code": '\n'.join(current_value[:-1]).strip()}
            parsed_data.append(entry)
        result_dict[current_key] = parsed_data
    elif current_key is not None:
        result_dict[current_key] = '\n'.join(current_value).strip()

    if "actions" in result_dict:
        actions = result_dict["actions"]
        if isinstance(actions, str):
            actions = actions.replace('```python', '').replace('```', '')
            actions = actions.split('\n')
        elif not isinstance(actions, list):
            actions = [str(actions)]

        normalized_actions = []
        for action in actions:
            action_text = str(action).split('#', 1)[0].strip()
            action_text = re.sub(r"^(?:[-*]\s*|\d+[\.)]\s*)", "", action_text)
            if action_text and re.match(r"^[A-Za-z_]\w*\s*\(.*\)$", action_text):
                normalized_actions.append(action_text)

        result_dict["actions"] = normalized_actions

    # Fallback 1: If no actions found, try to extract from code blocks in the original text
    if "actions" not in result_dict or not result_dict.get("actions"):
        fallback_actions = _extract_actions_from_code_blocks(text)
        if fallback_actions:
            result_dict["actions"] = fallback_actions
            # Log the fallback (using print since we don't have logger here)
            # print(f"[json_utils] Fallback: Extracted actions from code blocks: {fallback_actions}")

    # Fallback 2: If still no actions, try to extract from reasoning or full text
    if "actions" not in result_dict or not result_dict.get("actions"):
        # Try reasoning field first, then fall back to the full original text
        # (covers cases where the LLM outputs a thinking chain with no structure)
        reasoning_text = result_dict.get("reasoning", "") or text
        if reasoning_text:
            fallback_actions = _extract_actions_from_reasoning(reasoning_text)
            if fallback_actions:
                # Only take the last action mentioned (most likely the final decision)
                result_dict["actions"] = fallback_actions[-1:]

    # Fallback 3: If the model only described the next action in prose,
    # recover narrowly-scoped Stardew actions from natural language.
    if "actions" not in result_dict or not result_dict.get("actions"):
        reasoning_text = result_dict.get("reasoning", "") or text
        if reasoning_text:
            fallback_actions = _extract_actions_from_natural_language(reasoning_text)
            if fallback_actions:
                result_dict["actions"] = fallback_actions[-1:]

    if "success" in result_dict:
        success_value = result_dict["success"]
        if isinstance(success_value, bool):
            pass  # Already a boolean
        elif isinstance(success_value, str):
            success_lower = success_value.lower().strip()
            result_dict["success"] = success_lower in ("true", "yes", "1", "successful", "succeeded")
        else:
            result_dict["success"] = False
    else:
        # Try to find success-related keys with different names
        for key in ["succeed", "successful", "succeeded", "is_success", "result"]:
            if key in result_dict:
                success_value = result_dict[key]
                if isinstance(success_value, str):
                    success_lower = success_value.lower().strip()
                    result_dict["success"] = success_lower in ("true", "yes", "1", "successful", "succeeded")
                elif isinstance(success_value, bool):
                    result_dict["success"] = success_value
                break

    return result_dict


class JsonFrameStructure():

    def __init__(self):
        self.data_structure: Dict[int, Dict[str, list[Dict[str, any]]]] = {}
        self.end_index: int = -1


    def add_instance(self, timestamp: str, instance: dict[str, any]) -> None:
        # Check if the timestamp already exists across all indices
        exists = False
        for index_data in self.data_structure.values():
            if timestamp in index_data:
                # Timestamp already exists, append the instance to the existing timestamp
                index_data[timestamp].append(instance)
                exists = True
                break

        if not exists:
            # Timestamp is new, create a new entry and increment the end_index
            self.end_index += 1
            self.data_structure.setdefault(self.end_index, {}).setdefault(timestamp, []).append(instance)


    def sort_index_by_timestamp(self) -> None:
        extracted_data = [(key, value) for entry in self.data_structure.values() for key, value in entry.items()]
        sorted_data = sorted(extracted_data, key=lambda x: x[0])

        # Reconstructing the JSON structure with sorted data
        self.data_structure = OrderedDict({index: {key: value} for index, (key, value) in enumerate(sorted_data)})


    def search_type_across_all_indices(self, search_type: str) -> list[dict[str, any]]:

        results = []

        # Sort the keys in ascending order
        for index, index_data in sorted(self.data_structure.items()):
            for object_id, instances in index_data.items():
                for instance in instances:
                    for type, values in instance.items():
                        if type == search_type and values != "" and values != []:
                            results.append({"index": index, "object_id": object_id, "values":values})

        return results


    def to_dict(self):
        return {
            "data_structure": self.data_structure,
            "end_index": self.end_index
        }


    @classmethod
    def from_dict(cls, data_dict):
        instance = cls()
        instance.data_structure = data_dict.get("data_structure", {})
        instance.end_index = data_dict.get("end_index", -1)
        return instance
