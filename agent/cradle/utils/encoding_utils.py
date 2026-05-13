import base64
import os
from typing import Any, List
import io

import numpy as np
import cv2
from PIL import Image

from cradle.log.logger import Logger
from cradle.utils.file_utils import assemble_project_path
from cradle.utils.string_utils import hash_text_sha256

logger = Logger()

# LLM vision 最大分辨率 (从 enhanced_config.yaml 加载, 默认 960)
_LLM_MAX_RESOLUTION: int = 0  # 0 = 未加载


def _get_llm_max_resolution() -> int:
    """Lazy-load max resolution from config. Returns 0 to disable resize."""
    global _LLM_MAX_RESOLUTION
    if _LLM_MAX_RESOLUTION != 0:
        return _LLM_MAX_RESOLUTION
    try:
        import yaml
        from cradle.utils.file_utils import assemble_project_path as _asp
        cfg_path = os.getenv("STARDOJO_ENHANCED_CONFIG", "").strip()
        cfg_path = _asp(cfg_path) if cfg_path else _asp('./conf/enhanced_config.yaml')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
                _LLM_MAX_RESOLUTION = int(
                    cfg.get('performance', {}).get('vision', {})
                    .get('llm_max_resolution', 960)
                )
    except Exception:
        _LLM_MAX_RESOLUTION = 960
    return _LLM_MAX_RESOLUTION


def _resize_for_llm(image: Image.Image) -> Image.Image:
    """Resize image so its longest side <= llm_max_resolution.

    Preserves aspect ratio. Returns original if already small enough
    or if llm_max_resolution <= 0 (disabled).
    """
    max_res = _get_llm_max_resolution()
    if max_res <= 0:
        return image
    w, h = image.size
    longest = max(w, h)
    if longest <= max_res:
        return image
    scale = max_res / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    logger.debug(f"[ImageEncode] Resized {w}x{h} → {new_w}x{new_h} (max={max_res})")
    return resized


def encode_base64(payload):

    if payload is None:
        raise ValueError("Payload cannot be None.")

    return base64.b64encode(payload).decode('utf-8')


def decode_base64(payload):

    if payload is None:
        raise ValueError("Payload cannot be None.")

    return base64.b64decode(payload)


def encode_image_path(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = encode_image_binary(image_file.read(), image_path)
        return encoded_image


def encode_image_binary(image_binary, image_path=None):
    encoded_image = encode_base64(image_binary)
    if image_path is None:
        image_path = '<$bin_placeholder$>'

    logger.debug(f'|>. img_hash {hash_text_sha256(encoded_image)}, path {image_path} .<|')
    return encoded_image


def decode_image(base64_encoded_image):
    return decode_base64(base64_encoded_image)


def encode_data_to_base64_path(data: Any) -> List[str]:
    encoded_images = []
    
    # 🚀 P3 Fix: 去重优化 - 跟踪已处理的路径
    seen_paths = set()
    dedup_count = 0

    # Handle different input types
    if isinstance(data, (str, Image.Image, np.ndarray, bytes)):
        data = [data]
    elif isinstance(data, dict):
        # ✅ FIX: dict should be treated as a single item, not iterated over its keys
        data = [data]

    for item in data:
        logger.debug(f"[ImageEncode] Processing item type: {type(item).__name__}, value: {str(item)[:100]}...")
        buffered = None  # Initialize buffered to avoid reference errors
        
        if isinstance(item, str):
            # 🚀 P3 Fix: 检查路径去重
            if item in seen_paths:
                dedup_count += 1
                logger.write(f"[ImageEncode] ⏩ DUPLICATE SKIPPED: {item}")
                continue
            seen_paths.add(item)
            
            # Try to assemble project path first
            path = assemble_project_path(item)
            
            # If assembled path doesn't exist, try the original path
            if not os.path.exists(path):
                path = item
            
            # If path exists, encode it (with resize for LLM)
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    img = _resize_for_llm(img)
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    encoded_image = encode_base64(buffered.getvalue())
                    logger.debug(f'|>. img_hash {hash_text_sha256(encoded_image)}, path {path} .<|')
                except Exception:
                    # Fallback: encode raw file if PIL fails
                    encoded_image = encode_image_path(path)
                image_type = "jpeg"
                encoded_image = f"data:image/{image_type};base64,{encoded_image}"
                encoded_images.append(encoded_image)
            else:
                # Path doesn't exist - log error and skip
                logger.error(f"[ImageEncode] Image path does not exist: {item} (assembled: {path})")
                # Don't append invalid paths - this causes 400 errors!
                # encoded_images.append(item)  # ❌ BAD: sends file path to LLM
                continue  # Skip missing images entirely

            continue

        elif isinstance(item, bytes):  # raw bytes - try to decode as image
            buffered = io.BytesIO(item)
            try:
                image = Image.open(buffered)
                image = _resize_for_llm(image)
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG", quality=85)
            except Exception as e:
                logger.error(f"Failed to decode bytes as image: {e}")
                continue
        elif isinstance(item, Image.Image):  # PIL image
            item = _resize_for_llm(item)
            buffered = io.BytesIO()
            item.save(buffered, format="JPEG", quality=85)
        elif isinstance(item, np.ndarray):  # cv2 image array
            item = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)  # convert to RGB
            image = Image.fromarray(item)
            image = _resize_for_llm(image)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
        elif item is None:
            logger.error("Trying to encode None image! Skipping it.")
            continue
        elif isinstance(item, dict):
            # Handle dict-based image structures (e.g., from augment methods)
            # Try to extract the actual image path or data
            if 'path' in item:
                # Recursively process the path value
                nested_result = encode_data_to_base64_path(item['path'])
                encoded_images.extend(nested_result)
            elif 'image' in item:
                nested_result = encode_data_to_base64_path(item['image'])
                encoded_images.extend(nested_result)
            elif 'data' in item:
                nested_result = encode_data_to_base64_path(item['data'])
                encoded_images.extend(nested_result)
            elif 'augmented_image' in item:
                nested_result = encode_data_to_base64_path(item['augmented_image'])
                encoded_images.extend(nested_result)
            else:
                logger.warn(f"Dict item has no known image key (path/image/data/augmented_image): {list(item.keys())}. Skipping.")
            continue
        else:
            logger.error(f"Unknown image type: {type(item)}. Skipping it.")
            continue

        # Only proceed if buffered was successfully created
        if buffered is not None:
            encoded_image = encode_image_binary(buffered.getvalue())
            encoded_image = f"data:image/jpeg;base64,{encoded_image}"
            encoded_images.append(encoded_image)
    
    # 🚀 P3 Fix: 输出去重统计
    if dedup_count > 0:
        logger.write(f"[ImageEncode] ✅ Removed {dedup_count} duplicate image(s)")
        logger.write(f"[ImageEncode] Final count: {len(encoded_images)} unique images")

    return encoded_images
