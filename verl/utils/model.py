import importlib.util
from pathlib import Path

import transformers


if not hasattr(transformers, "AutoModelForVision2Seq"):
    replacement = getattr(transformers, "AutoModelForImageTextToText", None)
    if replacement is None:
        raise ImportError(
            "Transformers v5 compatibility patch could not find AutoModelForImageTextToText."
        )
    transformers.AutoModelForVision2Seq = replacement


def _load_original_module():
    import verl.utils as verl_utils_pkg

    current_dir = Path(__file__).resolve().parent
    for package_path in map(Path, verl_utils_pkg.__path__):
        candidate = package_path / "model.py"
        if not candidate.exists() or candidate.parent == current_dir:
            continue
        spec = importlib.util.spec_from_file_location(
            "_apa_original_verl_utils_model", candidate
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ImportError("Could not locate installed verl.utils.model for delegation.")


_original_module = _load_original_module()

for name, value in vars(_original_module).items():
    if name.startswith("__") and name not in {"__all__", "__doc__"}:
        continue
    globals()[name] = value
