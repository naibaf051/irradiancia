"""Kernel ML toolkit: kernel functions (Belanche), KSVC, KANNC and KRidgeClassifier."""
from .KernelUtilities import make_kernel, gram_matrix, KERNEL_NAMES
from .KSVM import KSVC
from .KANN import KANNC
from .KRidge import KRidgeClassifier
from .discretizers import TargetDiscretizer
from .metrics import class_scores, evaluate_classifier

__all__ = ["make_kernel", "gram_matrix", "KERNEL_NAMES", "KSVC", "KANNC",
           "KRidgeClassifier", "TargetDiscretizer", "class_scores", "evaluate_classifier"]
