import importlib.util
from pathlib import Path
from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)

current_dir = Path(__file__).resolve().parent
for package_path in map(Path, __path__):
    candidate = package_path / "__init__.py"
    if not candidate.exists() or candidate.parent == current_dir:
        continue
    spec = importlib.util.spec_from_file_location("_apa_original_verl", candidate)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in vars(module).items():
        if name.startswith("__") and name not in {"__all__", "__doc__"}:
            continue
        globals().setdefault(name, value)
    break
