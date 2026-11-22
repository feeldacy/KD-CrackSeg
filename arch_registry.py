"""Shared registry of the model architectures this repository can train.

Every architecture family declares a teacher-side class, a student-side class
and the input resolution its models expect. The training, evaluation and
complexity scripts all resolve their models through this registry, so adding
a new architecture only has to be done here.
"""

from student_lactnet import LACTNet as StudentLACTNet
from teacher_lactnet import LACTNet as TeacherLACTNet


ARCH_REGISTRY = {
    # The original LACTNet pair, trained on 256x256 tiles.
    "lactnet": {
        "teacher": TeacherLACTNet,
        "student": StudentLACTNet,
        "size": (256, 256),
    },
}


def get_arch(name):
    """Return the registry entry of an architecture family."""
    if name not in ARCH_REGISTRY:
        raise KeyError(
            f"Unknown architecture '{name}'. Available: {list(ARCH_REGISTRY)}"
        )
    return ARCH_REGISTRY[name]


def get_model_class(name, role="student"):
    """Return the model class of an architecture family.

    Args:
        name: architecture family, e.g. ``lactnet``.
        role: ``student`` or ``teacher``.
    """
    entry = get_arch(name)
    if role not in ("student", "teacher"):
        raise ValueError(f"role must be 'student' or 'teacher', got '{role}'")
    return entry[role]


def get_input_size(name):
    """Return the (width, height) an architecture family expects."""
    return get_arch(name)["size"]


def resolve_model_spec(name):
    """Return (model_class, input_size) for an ``evaluate.py --model`` value.

    The legacy names ``teacher`` and ``student`` select the two LACTNet
    variants; every other name is an architecture family whose teacher and
    student classes are the same model.
    """
    if name in ("teacher", "student"):
        role = "teacher" if name == "teacher" else "student"
        return get_model_class("lactnet", role), get_input_size("lactnet")
    return get_model_class(name, "student"), get_input_size(name)


def detect_arch(checkpoint, state_dict=None):
    """Infer the architecture of a checkpoint.

    Args:
        checkpoint: the loaded checkpoint, either a bare state dict or a dict
            with a ``model`` key.
        state_dict: the model weights, when they are nested under ``model``.

    Returns an ``evaluate.py --model`` compatible value: ``teacher`` or
    ``student`` for the two LACTNet variants. Returns None when the
    architecture cannot be determined.
    """
    if isinstance(checkpoint, dict):
        recorded = checkpoint.get("arch") or checkpoint.get("model_type")
        if recorded in ARCH_REGISTRY:
            # The LACTNet family needs the role to pick the right class.
            if recorded == "lactnet":
                role = checkpoint.get("role")
                return role if role in ("teacher", "student") else "teacher"
            return recorded
        if recorded in ("teacher", "student"):
            return recorded

    # Fall back to inspecting the parameter names of the weights.
    weights = state_dict if state_dict is not None else checkpoint
    return detect_arch_from_keys(weights)


def detect_arch_from_keys(checkpoint):
    """Infer the architecture from the parameter names of a state dict."""
    if not isinstance(checkpoint, dict):
        return None
    keys = checkpoint.keys()
    if not keys:
        return None

    # The LACTNet teacher carries the transformer encoder (patch_embed, swin)
    # and the AFFM blocks the student lacks; the student starts at "conv1".
    if any(key.startswith(("affm", "patch_embed", "swin")) for key in keys):
        return "teacher"
    if any(key.startswith("conv1.") for key in keys):
        return "student"

    return None
