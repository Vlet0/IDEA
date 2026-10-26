"""
defend/
=======
Defender-side analysis of the trained victim model.  Everything here is run by the *defender*,
who owns the model weights and a small held-out **clean** dataset but does NOT know the trigger,
the poisoned subset, or the attack target.  Nothing in this package is allowed to touch
``attack/`` or to use trigger / poison-mask information as a defense input: the detectors receive
the poison mask only to *score* their own output (AUROC / recall), never to produce it.

Two families:

* **Detection** (:mod:`defend.detection`) — does the poison leave a separable trace in feature
  space?  Spectral Signatures (Tran et al. 2018) and Activation Clustering (Chen et al. 2018)
  on the training features, plus a clean-vs-triggered linear-probe separability at test time.
  The proposed latent attack *trains* ``u_k^T z`` to separate, so a defender that looks at the
  latent geometry is expected to succeed — these numbers quantify that cost.

* **Model repair** (:mod:`defend.model_repair`) — Fine-Pruning (Liu et al. 2018) and clean
  Fine-Tuning: standard post-hoc removal that only uses the defender's clean data.  Reported as
  ASR-vs-utility curves so an attack is only "robust" if ASR survives at a clean-utility cost the
  defender would refuse to pay.
"""

from .detection import (
    activation_clustering_audit,
    feature_space_report,
    linear_probe_separability,
)
from .model_repair import fine_pruning_curve, fine_tuning_curve

__all__ = [
    "activation_clustering_audit",
    "linear_probe_separability",
    "feature_space_report",
    "fine_pruning_curve",
    "fine_tuning_curve",
]
