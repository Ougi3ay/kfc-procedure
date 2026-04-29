

from typing import Any, Dict, List, Union

from kfc_procedure.core.clustering.bregman import BregmanKMeans


def resolve_bregman(cfg: Union[str, Dict[str, Any], BregmanKMeans]) -> BregmanKMeans:
    if isinstance(cfg, BregmanKMeans):
        return cfg

    if isinstance(cfg, str):
        return BregmanKMeans(divergence=cfg)

    if isinstance(cfg, dict):
        if "name" not in cfg:
            raise ValueError(f"Missing 'name' in config: {cfg}")
        
        name = cfg["name"]
        params = cfg.get("params", {})

        # merge top level key
        extra = {
            key : value for key, value in cfg.items()
            if key not in {"name", "params"}
        }
        merged = {**params, **extra}
        return BregmanKMeans(divergence=name, **merged)

    raise TypeError(
        f"Unsupported divergence type: {type(cfg)}"
    )

def resolve_kstep(cfgs: List[Union[str, Dict[str, Any], BregmanKMeans]]) -> Dict[str, BregmanKMeans]:
    models = {}
    for idx, cfg in enumerate(cfgs):
        model = resolve_bregman(cfg)
        key = f"DB{idx}"
        models[key] = model
    return models

