import base64
import os
from typing import Any, List, Protocol, Tuple, cast
import io

import numpy as np
import cv2
from PIL import Image

from stardojo.log.logger import Logger
from stardojo.utils.file_utils import assemble_project_path
from stardojo.utils.string_utils import hash_text_sha256

logger = Logger()


class MSSScreenShot(Protocol):
    size: Tuple[int, int]
    bgra: bytes


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

    if isinstance(data, (str, Image.Image, np.ndarray, bytes)) or (hasattr(data, "size") and hasattr(data, "bgra")):
        data = [data]

    for item in data:
        if isinstance(item, str):
            path = assemble_project_path(item)
            if not os.path.exists(path):
                path = item

            if os.path.exists(path):
                encoded_image = encode_image_path(path)
                image_type = path.split(".")[-1].lower()
                encoded_image = f"data:image/{image_type};base64,{encoded_image}"
                encoded_images.append(encoded_image)
            else:
                logger.error(f"[ImageEncode] Image path does not exist: {item} (resolved: {path})")

            continue

        if item is None:
            logger.error("Tring to encode None image! Skipping it.")
            continue

        buffered = io.BytesIO()

        if isinstance(item, bytes):  # raw image bytes
            encoded_image = encode_image_binary(item)
            encoded_image = f"data:image/jpeg;base64,{encoded_image}"
            encoded_images.append(encoded_image)
            continue
        elif hasattr(item, "size") and hasattr(item, "bgra"):  # mss grab screenshot
            screenshot = cast(MSSScreenShot, item)
            image = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
            image.save(buffered, format="JPEG")
        elif isinstance(item, Image.Image):  # PIL image
            item.save(buffered, format="JPEG")
        elif isinstance(item, np.ndarray):  # cv2 image array
            item = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)  # convert to RGB
            image = Image.fromarray(item)
            image.save(buffered, format="JPEG")
        else:
            logger.error(f"[ImageEncode] Unsupported image type: {type(item)}")
            continue

        encoded_image = encode_image_binary(buffered.getvalue())
        encoded_image = f"data:image/jpeg;base64,{encoded_image}"
        encoded_images.append(encoded_image)

    return encoded_images
