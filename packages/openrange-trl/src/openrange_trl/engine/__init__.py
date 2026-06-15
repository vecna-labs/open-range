"""Async concurrent rollout engine — run many episodes at once against a model server.

The synchronous TRL path generates in-process, one episode at a time. This engine
runs N episodes concurrently (overlapping the model-wait), drives each with a
domain-agnostic ReAct loop against a configurable OpenAI-style endpoint, and reuses
the existing ``EpisodeEnv`` / sandbox / grading / trajectory pieces. Trainer-side only
— core OpenRange is untouched. Importing it pulls in no torch/transformers/httpx.
"""

from openrange_trl.engine.backend import InferBackend, OpenAIBackend
from openrange_trl.engine.dispatch import batch
from openrange_trl.engine.protocol import Action, Finish, ToolCall
from openrange_trl.engine.react import Policy, run_react
from openrange_trl.engine.rollout import AsyncRollout
from openrange_trl.engine.schema import tool_schema

__all__ = [
    "Action",
    "AsyncRollout",
    "Finish",
    "InferBackend",
    "OpenAIBackend",
    "Policy",
    "ToolCall",
    "batch",
    "run_react",
    "tool_schema",
]
