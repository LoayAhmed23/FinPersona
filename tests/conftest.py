import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_MODULE_NAMES = {
    "app",
    "config",
    "data_loader",
    "evaluation",
    "feature_engineering",
    "feature_importance",
    "model",
    "pipeline",
    "preprocessing",
}


def load_project_module(relative_path: str, module_name: str):
    """Load a project file that uses sibling imports such as ``import config``."""
    module_path = PROJECT_ROOT / relative_path
    module_dir = str(module_path.parent)

    for name in LOCAL_MODULE_NAMES:
        sys.modules.pop(name, None)

    sys.path.insert(0, module_dir)
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if sys.path and sys.path[0] == module_dir:
            sys.path.pop(0)
