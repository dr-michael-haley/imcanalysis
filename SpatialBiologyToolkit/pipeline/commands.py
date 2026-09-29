"""Scientific command descriptions reusable by local and future workflow backends."""

from .models import StageCommand
from .registry import get_stage


# A stage's environment_keys is a set of requirements, not an invocation order.
MULTI_ENVIRONMENT_COMMANDS = {
    "cellpose": ("analysis", "cellposesam"),
    "dnqc": ("tensorflow", "analysis"),
    "cellvision-full": ("scportrait", "scportrait", "analysis", "scportrait"),
}


def stage_commands(stage_name: str) -> list[StageCommand]:
    stage = get_stage(stage_name)
    if not stage.python_modules or stage_name == "slogs":
        raise ValueError(
            f"Stage '{stage_name}' has no local scientific command. "
            "Its shell/SLURM utility is available through the SLURM backend."
        )
    keys = MULTI_ENVIRONMENT_COMMANDS.get(stage_name)
    if keys is None:
        if len(stage.environment_keys) != 1:
            raise ValueError(
                f"Stage '{stage_name}' needs an explicit command/environment mapping."
            )
        keys = tuple(stage.environment_keys * len(stage.python_modules))
    if len(keys) != len(stage.python_modules):
        raise ValueError(f"Incomplete command/environment mapping for '{stage_name}'.")
    return [
        StageCommand(module=module, environment_key=key)
        for module, key in zip(stage.python_modules, keys)
    ]
