"""
ArchitectAI prompt parser package.

Public API::

    from backend.core.prompt_parser import parse_prompt

    arch = parse_prompt("React frontend, FastAPI backend, Redis cache")
"""

from backend.core.prompt_parser.parser import parse_prompt

__all__ = ["parse_prompt"]
