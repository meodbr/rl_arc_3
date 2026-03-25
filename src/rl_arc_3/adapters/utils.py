from rl_arc_3.base.env import EnvSignature
from rl_arc_3.base.model import ModelSignature
from rl_arc_3.base.model_adapter import ModelAdapter

from rl_arc_3.adapters.full import FullModelAdapter
from rl_arc_3.adapters.keyboard_only import KeyboardOnlyModelAdapter


def get_model_adapter(
    name: str,
    env_signature: EnvSignature,
    model_signature: ModelSignature | None = None,
) -> ModelAdapter:
    if name == "full":
        return FullModelAdapter(env_signature, model_signature)
    elif name == "keyboard_only":
        return KeyboardOnlyModelAdapter(env_signature, model_signature)
    else:
        raise ValueError(f"Unknown model adapter name: {name}")
