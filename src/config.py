"""
GodDev 3.0 — Runtime Config Loader
=====================================

Agents load this at startup so the self-improver can tune their behaviour
across runs without touching source code.

Default values are conservative and safe. The self-improver overwrites
memory/runtime_config.json after each successful analysis run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_DEFAULT: dict[str, Any] = {
    "version": 1,
    "last_updated": "never",
    "total_improvements": 0,
    "agent_config": {
        "critic_score_threshold": 7.0,
        "min_critic_score": 5.0,
        "worker_max_retries": 1,
        "max_files_per_squad": 8,
        "cto_prompt_additions": "",
        "squad_leader_prompt_additions": "",
        "worker_prompt_additions": "",
        "preferred_agents": {
            "python_code": "openai",
            "javascript_code": "openai",
            "typescript_code": "openai",
            "tests": "deepseek",
            "algorithms": "deepseek",
            "documentation": "gemini",
            "config_files": "claude",
            "architecture": "claude",
            "docker": "deepseek",
            "ci_cd": "deepseek"
        }
    },
    "performance_history": {
        "total_runs": 0,
        "avg_critic_score": 0.0,
        "avg_verification_rate": 0.0,
        "avg_wall_time_s": 0.0,
        "critic_first_pass_rate": 0.0
    },
    "cost_effectiveness": {
        "cost_per_point": {},  # agent → cost per quality point
        "efficiency_by_agent": {},  # agent → {tokens_per_file, verification_rate, quality_score}
        "estimated_cost_per_run": 0.0,
        "cost_optimization_enabled": True,
        "preferred_agents_by_cost": {}  # task_type → cost-optimized agent ranking
    }
}


def _config_path() -> Path:
    memory_dir = os.getenv("MEMORY_DIR", "/opt/goddev/memory")
    return Path(memory_dir) / "runtime_config.json"


def _prune_prompt_additions(prompt_str: str, max_chars: int = 2000, num_blocks: int = 2) -> str:
    """
    Prunes a long prompt string to a maximum number of characters by keeping the most recent blocks.
    Blocks are defined by `\\n\\n` separators. This function aggressively removes duplicate blocks,
    prioritizing the most recent occurrence of any given block. If, after keeping the specified
    number of blocks, the string still exceeds `max_chars`, it will be further truncated to
    `max_chars` (minus the truncation marker length) with a `\\n\\n[…truncated]` marker appended.

    Args:
        prompt_str: The prompt string to prune.
        max_chars: The maximum desired length for the prompt string, including the truncation marker.
        num_blocks: The number of recent blocks to keep initially.

    Returns:
        The pruned prompt string.
    """
    if len(prompt_str) <= max_chars:
        return prompt_str

    blocks = prompt_str.split('\n\n')
    
    # Aggressively remove duplicate blocks by processing from most recent to oldest.
    # This ensures that if a block appears multiple times, only its latest occurrence
    # is preserved, preventing prompt bloat from repeated identical instructions/context.
    seen_blocks: set[str] = set()
    reverse_unique_blocks_with_latest: list[str] = []
    
    for block in reversed(blocks):
        if block not in seen_blocks:
            reverse_unique_blocks_with_latest.append(block)
            seen_blocks.add(block)
    
    # Re-reverse to get them in their original relative chronological order,
    # but with older duplicate instances removed and only the latest content of a block preserved.
    unique_and_chronological_blocks = list(reversed(reverse_unique_blocks_with_latest))

    # Keep the last `num_blocks` blocks from this refined list.
    if len(unique_and_chronological_blocks) > num_blocks:
        pruned_blocks = unique_and_chronological_blocks[-num_blocks:]
    else:
        pruned_blocks = unique_and_chronological_blocks
        
    pruned_str = '\n\n'.join(pruned_blocks)

    # Apply hard ceiling: if after keeping last num_blocks the result still exceeds max_chars,
    # truncate to max_chars and append '\n\n[…truncated]' marker.
    truncation_marker = '\n\n[…truncated]'
    if len(pruned_str) > max_chars:
        # Ensure enough space for the truncation marker
        if max_chars <= len(truncation_marker):
            # If max_chars is too small to even fit the marker, return as much of the marker as possible.
            return truncation_marker[:max_chars]
        
        return pruned_str[:max_chars - len(truncation_marker)] + truncation_marker
    
    return pruned_str


def prune_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Prunes specific prompt addition fields within the runtime configuration to cap their length.
    This helps reduce token usage by preventing excessively long prompt additions from being
    injected into every agent call. The pruning strategy involves keeping the most recent
    appended blocks (split by '\\n\\n') up to a certain character limit, while aggressively
    removing redundant/duplicate blocks.

    Args:
        config: The runtime configuration dictionary to be pruned. This function modifies
                a copy of the relevant parts of the dictionary and returns it.

    Returns:
        The pruned runtime configuration dictionary.
    """
    pruned_config = config.copy()
    agent_config = pruned_config.get("agent_config")

    if agent_config and isinstance(agent_config, dict):
        for key in ["cto_prompt_additions", "squad_leader_prompt_additions", "worker_prompt_additions"]:
            if key in agent_config and isinstance(agent_config[key], str):
                agent_config[key] = _prune_prompt_additions(agent_config[key])
    return pruned_config


def load_runtime_config() -> dict[str, Any]:
    """
    Load the current runtime config, falling back to defaults.
    Applies pruning to large prompt addition fields before returning.
    """
    p = _config_path()
    merged_config: dict[str, Any]
    if p.exists():
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
            # Deep merge with defaults (so new default keys are always present)
            merged_config = _deep_merge(_DEFAULT, loaded)
        except (json.JSONDecodeError, IOError):
            # If the file is corrupt or unreadable, fall back to defaults
            merged_config = json.loads(json.dumps(_DEFAULT))  # deep copy of default
    else:
        merged_config = json.loads(json.dumps(_DEFAULT))  # deep copy of default

    # Apply pruning to the in-memory config before returning
    pruned_config = prune_runtime_config(merged_config)
    return pruned_config


def save_runtime_config(config: dict[str, Any]) -> None:
    """
    Persist the runtime config to disk.
    The configuration is pruned before saving to ensure the on-disk file size is bounded.
    """
    p = _config_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Prune the config before writing to disk to keep file size bounded
        pruned_config = prune_runtime_config(config)
        p.write_text(json.dumps(pruned_config, indent=2), encoding="utf-8")
    except IOError as e:
        print(f"Error saving runtime config to {p}: {e}")
        # Optionally re-raise or handle more gracefully based on system needs


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively, keeping all base keys."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result