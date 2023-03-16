"""
Copy from stable diffusion
"""
import importlib


def instantiate_from_config(config:dict, **args_from_code):
    """Util funciton to decompose differenct modules using config

    Args:
        config (dict): with key of "target" and "params", better from yaml
        static 
        args_from_code: additional con


    Returns:
        a validation/training pipeline, a module
    """
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **args_from_code)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
