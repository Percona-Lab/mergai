import importlib.util
from pathlib import Path

_SPEC = __spec__ or importlib.util.find_spec(__name__)
if _SPEC is None or _SPEC.loader is None or _SPEC.origin is None:
    raise ImportError("Unable to initialize prompt loader")
if not hasattr(_SPEC.loader, "get_data"):
    raise ImportError("Loader does not support data access")
_PROMPTS_DIR = Path(_SPEC.origin).resolve().parent / "prompts"
_LOADER = _SPEC.loader


def load_prompt(prompt_name: str) -> str:
    return _LOADER.get_data(str(_PROMPTS_DIR / prompt_name)).decode("utf-8")


def load_system_prompt() -> str:
    return load_prompt("system_prompt.md")
