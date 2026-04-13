from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


root_sitecustomize = Path(__file__).resolve().parent.parent / "sitecustomize.py"
spec = spec_from_file_location("_apa_root_sitecustomize", root_sitecustomize)
assert spec is not None and spec.loader is not None
module = module_from_spec(spec)
spec.loader.exec_module(module)
