"""Step-recognition model definitions (thesis Ch. 3, step recognition).

The step classifier is a Temporal Shift Module (Lin et al., 2019) on a ResNet-18 backbone. The network
class is imported lazily through :func:`load_tsm_resnet18`, i.e. only when a step-recognition model is
actually built, so every other part of the package imports without PyTorch or the model module.
"""
import importlib

_MODEL_MODULE = "ws_racr.perception.step_recognition.models.tsm_resnet"


def load_tsm_resnet18():
    """Return the ``TSMResNet18`` class (``TSMResNet18(num_classes, n_segment, pretrained, dropout)``).

    Raises:
        RuntimeError: if the TSM-ResNet18 model module is not configured in this installation.
    """
    try:
        return importlib.import_module(_MODEL_MODULE).TSMResNet18
    except ImportError as exc:
        raise RuntimeError(
            "Step recognition requires the TSM-ResNet18 model module "
            f"'{_MODEL_MODULE}' to be installed in this environment."
        ) from exc
