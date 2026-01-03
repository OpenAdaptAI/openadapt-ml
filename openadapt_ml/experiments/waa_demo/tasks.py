"""Task definitions for WAA demo experiment.

10 carefully selected tasks across 4 enterprise-relevant domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Domain(Enum):
    BROWSER = "msedge"
    OFFICE_CALC = "libreoffice_calc"
    OFFICE_WRITER = "libreoffice_writer"
    SETTINGS = "settings"
    FILE_EXPLORER = "file_explorer"


@dataclass
class WATask:
    """A Windows Agent Arena task definition."""

    task_id: str
    instruction: str
    domain: Domain
    difficulty: Difficulty
    first_action_hint: str
    demo_method: str  # "manual" or "recorded"
    json_path: str  # Path in WAA repo


TASKS: dict[str, WATask] = {
    "1": WATask(
        task_id="004587f8-6028-4656-94c1-681481abbc9c-wos",
        instruction="Enable the 'Do Not Track' feature in Edge",
        domain=Domain.BROWSER,
        difficulty=Difficulty.MEDIUM,
        first_action_hint="Click three-dot menu in Edge",
        demo_method="manual",
        json_path="examples/msedge/004587f8-6028-4656-94c1-681481abbc9c-wos.json",
    ),
    "2": WATask(
        task_id="049d3788-c979-4ea6-934d-3a35c4630faf-WOS",
        instruction="Save this webpage to bookmarks bar",
        domain=Domain.BROWSER,
        difficulty=Difficulty.EASY,
        first_action_hint="Click star/bookmark icon or Ctrl+D",
        demo_method="manual",
        json_path="examples/msedge/049d3788-c979-4ea6-934d-3a35c4630faf-WOS.json",
    ),
    "3": WATask(
        task_id="2acd62b4-a2ab-44a7-a7e3-f5227bbd8324-wos",
        instruction="Set default font size to largest for grandmother",
        domain=Domain.BROWSER,
        difficulty=Difficulty.MEDIUM,
        first_action_hint="Open Settings > Appearance",
        demo_method="manual",
        json_path="examples/msedge/2acd62b4-a2ab-44a7-a7e3-f5227bbd8324-wos.json",
    ),
    "4": WATask(
        task_id="01b269ae-2111-4a07-81fd-3fcd711993b0-WOS",
        instruction="Fill all blank cells with value from cell above",
        domain=Domain.OFFICE_CALC,
        difficulty=Difficulty.HARD,
        first_action_hint="Select cells, use Go To Special > Blanks",
        demo_method="recorded",
        json_path="examples/libreoffice_calc/01b269ae-2111-4a07-81fd-3fcd711993b0-WOS.json",
    ),
    "5": WATask(
        task_id="0a2e43bf-b26c-4631-a966-af9dfa12c9e5-WOS",
        instruction="Calculate monthly totals and create line chart",
        domain=Domain.OFFICE_CALC,
        difficulty=Difficulty.HARD,
        first_action_hint="Click cell for SUM formula",
        demo_method="recorded",
        json_path="examples/libreoffice_calc/0a2e43bf-b26c-4631-a966-af9dfa12c9e5-WOS.json",
    ),
    "6": WATask(
        task_id="3ef2b351-8a84-4ff2-8724-d86eae9b842e-WOS",
        instruction="Center align the heading in LibreOffice Writer",
        domain=Domain.OFFICE_WRITER,
        difficulty=Difficulty.EASY,
        first_action_hint="Select text, click center align button",
        demo_method="manual",
        json_path="examples/libreoffice_writer/3ef2b351-8a84-4ff2-8724-d86eae9b842e-WOS.json",
    ),
    "7": WATask(
        task_id="37e10fc4-b4c5-4b02-a65c-bfae8bc51d3f-wos",
        instruction="Turn off notifications for system",
        domain=Domain.SETTINGS,
        difficulty=Difficulty.MEDIUM,
        first_action_hint="Open Settings > System > Notifications",
        demo_method="manual",
        json_path="examples/settings/37e10fc4-b4c5-4b02-a65c-bfae8bc51d3f-wos.json",
    ),
    "8": WATask(
        task_id="46adf721-2949-4426-b069-010b7c128d8f-wos",
        instruction="Enable Night Light: on at 7PM, off at 7AM",
        domain=Domain.SETTINGS,
        difficulty=Difficulty.MEDIUM,
        first_action_hint="Open Settings > Display > Night Light",
        demo_method="manual",
        json_path="examples/settings/46adf721-2949-4426-b069-010b7c128d8f-wos.json",
    ),
    "9": WATask(
        task_id="0c9dda13-428c-492b-900b-f48562111f93-WOS",
        instruction="Create Archive folder and move all .docx files",
        domain=Domain.FILE_EXPLORER,
        difficulty=Difficulty.MEDIUM,
        first_action_hint="Right-click > New Folder, then select and move files",
        demo_method="recorded",
        json_path="examples/file_explorer/0c9dda13-428c-492b-900b-f48562111f93-WOS.json",
    ),
    "10": WATask(
        task_id="34a4fee9-e52e-4a4a-96d2-68d35091504a-WOS",
        instruction="Change view to Details view",
        domain=Domain.FILE_EXPLORER,
        difficulty=Difficulty.EASY,
        first_action_hint="Click View menu or dropdown",
        demo_method="manual",
        json_path="examples/file_explorer/34a4fee9-e52e-4a4a-96d2-68d35091504a-WOS.json",
    ),
}


def get_task(task_num: str | int) -> Optional[WATask]:
    """Get a task by its number (1-10)."""
    return TASKS.get(str(task_num))


def get_tasks_by_method(method: str) -> list[WATask]:
    """Get all tasks that use a specific demo method."""
    return [t for t in TASKS.values() if t.demo_method == method]


def get_manual_tasks() -> list[WATask]:
    """Get tasks requiring manual demo writing."""
    return get_tasks_by_method("manual")


def get_recorded_tasks() -> list[WATask]:
    """Get tasks requiring recorded demos."""
    return get_tasks_by_method("recorded")
