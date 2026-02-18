"""Prompt templates and builders for the Oxmaint agent."""

from app.prompts.templates import (
    GROQ_MANUAL_STRUCTURING_SYSTEM_PROMPT,
    build_manual_structuring_user_prompt,
)

__all__ = [
    "GROQ_MANUAL_STRUCTURING_SYSTEM_PROMPT",
    "build_manual_structuring_user_prompt",
]
