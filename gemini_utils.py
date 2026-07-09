#!/usr/bin/env python3
"""Small Gemini API helpers for Interact3D."""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_GEMINI_TEXT_MODEL = os.getenv("INTERACT3D_GEMINI_TEXT_MODEL", "gemini-3.5-flash")
DEFAULT_GEMINI_VISION_MODEL = os.getenv("INTERACT3D_GEMINI_VISION_MODEL", DEFAULT_GEMINI_TEXT_MODEL)
DEFAULT_GEMINI_IMAGE_MODEL = os.getenv("INTERACT3D_GEMINI_IMAGE_MODEL", "gemini-3-pro-image")
DEFAULT_GEMINI_RETRIES = int(os.getenv("INTERACT3D_GEMINI_RETRIES", "3"))
DEFAULT_GEMINI_TIMEOUT = int(os.getenv("INTERACT3D_GEMINI_TIMEOUT", "300"))
DEFAULT_GEMINI_IMAGE_TIMEOUT = int(os.getenv("INTERACT3D_GEMINI_IMAGE_TIMEOUT", "900"))


def log(message: str) -> None:
    print(f"[gemini] {message}", file=sys.stderr, flush=True)


def parse_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in Gemini response: {text}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError(f"Gemini JSON response is not an object: {text}")
    return data


def _api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before calling the Gemini API.")
    return key


def _client(timeout: int | None = None) -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("Install google-genai in the active environment: pip install google-genai") from error

    if timeout:
        try:
            return genai.Client(api_key=_api_key(), http_options=types.HttpOptions(timeout=timeout * 1000)), types
        except TypeError:
            pass
    return genai.Client(api_key=_api_key()), types


def _config(types: Any, **kwargs: Any) -> Any:
    try:
        return types.GenerateContentConfig(**kwargs)
    except Exception:
        return kwargs


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _image_part(types: Any, path: Path) -> Any:
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=_mime_type(path))


def _response_parts(response: Any) -> list[Any]:
    parts: list[Any] = list(getattr(response, "parts", []) or [])
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        parts.extend(getattr(content, "parts", []) or [])
    return parts


def _response_text(response: Any) -> str:
    try:
        text = getattr(response, "text", None)
    except Exception:
        text = None
    if text:
        return str(text)
    chunks: list[str] = []
    for part in _response_parts(response):
        part_text = getattr(part, "text", None)
        if part_text:
            chunks.append(str(part_text))
    return "\n".join(chunks)


def _inline_image_bytes(response: Any) -> bytes | None:
    for part in _response_parts(response):
        inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
        data = getattr(inline, "data", None) if inline is not None else None
        if data:
            return base64.b64decode(data) if isinstance(data, str) else bytes(data)
    return None


def generate_json(
    *,
    prompt: str,
    model: str,
    label: str,
    retries: int,
    timeout: int,
    image_paths: Path | list[Path] | None = None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = [] if image_paths is None else (image_paths if isinstance(image_paths, list) else [image_paths])
    schema_text = f"\n\nJSON schema:\n{json.dumps(schema, ensure_ascii=False)}" if schema else ""
    full_prompt = f"{prompt}{schema_text}\n\nReturn only one valid JSON object. Do not include markdown."
    client, types = _client(timeout)
    contents: list[Any] = [full_prompt, *[_image_part(types, path) for path in paths]]
    config = _config(types, response_mime_type="application/json")
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        suffix = f" attempt {attempt}/{retries}" if retries > 1 else ""
        log(f"{label}: start{suffix}")
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            return parse_json(_response_text(response))
        except Exception as error:
            last_error = error
            log(f"{label}: {type(error).__name__}: {error}")
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"Gemini {label} failed after {retries} attempt(s)") from last_error


def generate_image(
    *,
    prompt: str,
    output_path: Path,
    model: str,
    label: str,
    retries: int,
    timeout: int,
    source: Path | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client, types = _client(timeout)
    instruction = f"""
Create exactly one PNG image for Interact3D image-to-3D reconstruction.
Instruction: {prompt}
Requirements:
- Return an actual image, not only text.
- Pure white background (#FFFFFF), complete object geometry, centered and fully visible.
- No text, watermark, UI, border, or extra objects unless explicitly requested.
""".strip()
    contents: list[Any] = [instruction]
    if source is not None:
        contents.append(_image_part(types, source))
    config = _config(types, response_modalities=["TEXT", "IMAGE"])
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        suffix = f" attempt {attempt}/{retries}" if retries > 1 else ""
        log(f"{label}: start{suffix}")
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            data = _inline_image_bytes(response)
            if not data:
                raise RuntimeError(f"Gemini returned no image. Text response: {_response_text(response)[-2000:]}")
            image = Image.open(BytesIO(data))
            image.save(output_path)
            validate_png(output_path)
            return
        except Exception as error:
            last_error = error
            log(f"{label}: {type(error).__name__}: {error}")
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"Gemini {label} failed after {retries} attempt(s)") from last_error


def validate_png(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Gemini did not create expected image: {path}")
    with Image.open(path) as image:
        image.verify()
