"""Shared prompt formatting for turn-based robot dialogue.

The OmniAction-LIBERO data provides user speech separately from the robot's
textual replies.  Audio is supplied through the audio-conditioning path, while
this module formats only the robot-side textual history that is fed back to the
VLA model on later turns.
"""

from __future__ import annotations

from collections.abc import Sequence


ACT_TOKEN = "[ACT]"
TALKER_SYSTEM_PROMPT = "User audio is provided. Respond as the robot."
ACTION_SYSTEM_PROMPT = "User audio is provided. Execute the confirmed request."


def _clean_turn(turn: str) -> str:
    return " ".join(str(turn).strip().split())


def build_talker_prompt(previous_robot_turns: Sequence[str] = ()) -> str:
    """Format the text context used before generating the next robot reply."""
    lines = [TALKER_SYSTEM_PROMPT]
    cleaned_turns = [_clean_turn(turn) for turn in previous_robot_turns if _clean_turn(turn)]
    if cleaned_turns:
        lines.append("Previous robot turns:")
        lines.extend(cleaned_turns)
    lines.append("Robot:")
    return "\n".join(lines)


def build_action_prompt(robot_turns: Sequence[str] = ()) -> str:
    """Format the text context provided to the continuous-action executor."""
    lines = [ACTION_SYSTEM_PROMPT]
    cleaned_turns = [_clean_turn(turn) for turn in robot_turns if _clean_turn(turn)]
    if cleaned_turns:
        lines.append("Robot dialogue:")
        lines.extend(cleaned_turns)
    return "\n".join(lines)
