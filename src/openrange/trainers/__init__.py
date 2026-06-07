"""Trainer adapters: the consumer side of the training-integration standard.

Each module here adapts one external trainer to the pack-agnostic seam in
``openrange.training`` (``EpisodeReport → (trajectory, reward)``). The adapters
are deliberately import-light: ``openrange.trainers.trl`` is torch-free and
unit-testable without a model — only the gated ``tests/test_trl_live.py`` and the
``examples/trl_grpo_lora.ipynb`` notebook construct a real ``GRPOTrainer``. See
``DESIGN.md`` for the shape and the bet.
"""

from __future__ import annotations
