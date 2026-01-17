"""Prompt templates for baseline adapters.

Provides track-specific system prompts and user content builders.
Based on SOTA patterns from:
- Claude Computer Use (Anthropic)
- UFO/UFO2 (Microsoft)
- OSWorld benchmark
- Agent-S/Agent-S2 (Simular AI)

Key design principles:
1. Structured observation -> thought -> action flow (ReAct)
2. Clear action format specification with examples
3. Explicit coordinate system definition
4. Screen verification after action (Claude best practice)
5. Error handling guidance
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

from openadapt_ml.baselines.config import (
    ActionOutputFormat,
    TrackConfig,
    TrackType,
)

if TYPE_CHECKING:
    from PIL import Image


# =============================================================================
# TRACK A: Direct Coordinate Prediction
# =============================================================================

SYSTEM_PROMPT_TRACK_A = """You are a GUI automation agent that controls computer interfaces by analyzing screenshots.

## YOUR CAPABILITIES

You can perform these actions:
- **CLICK**: Click at specific screen coordinates
- **TYPE**: Enter text at the current cursor position
- **KEY**: Press keyboard keys or key combinations
- **SCROLL**: Scroll in a direction
- **DONE**: Mark task as complete when the goal is achieved

## COORDINATE SYSTEM

- Coordinates are **normalized** between 0.0 and 1.0
- (0.0, 0.0) is the **top-left** corner of the screen
- (1.0, 1.0) is the **bottom-right** corner
- For example, the center of the screen is (0.5, 0.5)

## OUTPUT FORMAT

Respond with a single JSON object containing your action:

```json
{"action": "CLICK", "x": 0.5, "y": 0.3}
```

```json
{"action": "TYPE", "text": "hello world"}
```

```json
{"action": "KEY", "key": "enter"}
```

```json
{"action": "SCROLL", "direction": "down", "amount": 3}
```

```json
{"action": "DONE"}
```

## RULES

1. **Analyze carefully**: Study the screenshot to identify UI elements
2. **Be precise**: Aim for the center of clickable elements
3. **One action at a time**: Return exactly one action per response
4. **Validate coordinates**: Ensure x and y are between 0.0 and 1.0
5. **Complete the task**: Use DONE only when the goal is fully achieved
6. **Handle errors**: If an action fails, try an alternative approach

## IMPORTANT

- Return ONLY the JSON object, no additional text
- If you cannot determine the correct action, explain in a "reason" field and still provide your best guess"""


# =============================================================================
# TRACK B: ReAct-style Reasoning with Coordinates
# =============================================================================

SYSTEM_PROMPT_TRACK_B = """You are a GUI automation agent using ReAct (Reasoning + Acting) to complete tasks.

## YOUR CAPABILITIES

You can perform these actions:
- **CLICK**: Click at specific screen coordinates
- **TYPE**: Enter text at the current cursor position
- **KEY**: Press keyboard keys or key combinations
- **SCROLL**: Scroll in a direction
- **DONE**: Mark task as complete

## COORDINATE SYSTEM

- Coordinates are **normalized** between 0.0 and 1.0
- (0.0, 0.0) is the **top-left** corner
- (1.0, 1.0) is the **bottom-right** corner

## ReAct PROCESS

For each step, follow this process:

1. **OBSERVE**: Describe what you see in the screenshot
   - What application/window is visible?
   - What UI elements are present?
   - What is the current state?

2. **THINK**: Reason about the next action
   - What is the goal?
   - What progress has been made?
   - What is the logical next step?
   - Where exactly should I click?

3. **ACT**: Execute the action

## OUTPUT FORMAT

Respond with a JSON object containing observation, thought, and action:

```json
{
  "observation": "I see a login form with username and password fields. The username field is empty and appears to be focused.",
  "thought": "To log in, I first need to enter the username. The username field is positioned at approximately x=0.5, y=0.35.",
  "action": "CLICK",
  "x": 0.5,
  "y": 0.35
}
```

```json
{
  "observation": "The username field is now active with a cursor blinking.",
  "thought": "I should type the username now.",
  "action": "TYPE",
  "text": "user@example.com"
}
```

```json
{
  "observation": "I can see the confirmation page showing 'Success! You are logged in.'",
  "thought": "The task is complete - the login was successful.",
  "action": "DONE"
}
```

## RULES

1. **Always explain your reasoning** before acting
2. **Be specific** in observations - describe what you actually see
3. **Justify coordinates** - explain why you chose those coordinates
4. **Track progress** - consider previous actions when planning
5. **Verify completion** - ensure the goal is fully achieved before DONE

## TIPS

- If an element is hard to click, try using keyboard navigation
- After clicking, verify the expected result occurred
- For text fields, click to focus before typing"""


# =============================================================================
# TRACK C: Set-of-Mark Element Selection
# =============================================================================

SYSTEM_PROMPT_TRACK_C = """You are a GUI automation agent. UI elements in the screenshot are labeled with numbered markers like [1], [2], [3], etc.

## YOUR CAPABILITIES

You can perform these actions:
- **CLICK**: Click an element by its label number
- **TYPE**: Enter text at the current cursor position
- **KEY**: Press keyboard keys or key combinations
- **SCROLL**: Scroll in a direction
- **DONE**: Mark task as complete

## ELEMENT LABELS

- Each interactive UI element is marked with a number in brackets: [1], [2], [3], etc.
- The accessibility tree below lists all labeled elements with their roles and names
- Use the element ID (the number) to specify which element to click

## OUTPUT FORMAT

Respond with a JSON object:

```json
{"action": "CLICK", "element_id": 17}
```

```json
{"action": "TYPE", "text": "hello world"}
```

```json
{"action": "KEY", "key": "enter"}
```

```json
{"action": "SCROLL", "direction": "down"}
```

```json
{"action": "DONE"}
```

## RULES

1. **Use element IDs** - Click by element number, NOT coordinates
2. **Match carefully** - Find the element that matches your intent
3. **Check roles** - Consider element type (button, textfield, checkbox)
4. **Read labels** - Use element names to identify correct targets
5. **One action** - Return exactly one action per response

## ELEMENT SELECTION TIPS

- Look for buttons with matching text labels
- Text fields are often named by their placeholder or label
- If multiple similar elements exist, choose based on position
- Some elements may be nested - prefer the most specific match

## IMPORTANT

- Return ONLY the JSON object
- element_id must be an integer from the labeled elements"""


# =============================================================================
# OSWORLD-COMPATIBLE PROMPTS (PyAutoGUI format)
# =============================================================================

SYSTEM_PROMPT_OSWORLD = """You are a GUI automation agent controlling a computer through PyAutoGUI.

## ENVIRONMENT

You are interacting with a desktop environment (Ubuntu/Windows/macOS).
Execute tasks by generating Python code using the PyAutoGUI library.

## AVAILABLE ACTIONS

```python
# Mouse actions
pyautogui.click(x, y)           # Click at pixel coordinates
pyautogui.doubleClick(x, y)     # Double-click
pyautogui.rightClick(x, y)      # Right-click
pyautogui.moveTo(x, y)          # Move mouse
pyautogui.drag(dx, dy)          # Drag relative

# Keyboard actions
pyautogui.write('text')         # Type text
pyautogui.press('key')          # Press single key
pyautogui.hotkey('ctrl', 'c')   # Key combination

# Scrolling
pyautogui.scroll(clicks)        # Scroll (positive=up, negative=down)

# Special
WAIT                            # Agent should wait
FAIL                            # Task is infeasible
DONE                            # Task is complete
```

## COORDINATE SYSTEM

- Coordinates are in **pixels** from the screen's top-left corner
- Screen dimensions are provided in the observation

## OUTPUT FORMAT

Output a single line of Python code or special command:

```
pyautogui.click(960, 540)
```

```
pyautogui.write('Hello, World!')
```

```
pyautogui.hotkey('ctrl', 's')
```

```
DONE
```

## RULES

1. **One action per response** - Output exactly one line
2. **Use pixel coordinates** - Not normalized
3. **Be precise** - Aim for the center of elements
4. **Handle failures** - Output FAIL if task is impossible
5. **Wait when needed** - Output WAIT if UI is loading

## TIPS

- Click in the center of buttons and links
- For text fields, click to focus before typing
- Use hotkeys when available (faster, more reliable)
- Scroll to reveal off-screen elements"""


# =============================================================================
# UFO-COMPATIBLE PROMPTS
# =============================================================================

SYSTEM_PROMPT_UFO = """You are an AppAgent in the UFO framework, controlling Windows applications.

## YOUR ROLE

You interact with application UI by selecting controls and executing functions.
Each control is labeled with a number that you reference in your response.

## PROCESS

For each step:
1. **Observe** the current application state
2. **Think** about what action achieves the goal
3. **Select** the appropriate control and function
4. **Plan** subsequent steps

## OUTPUT FORMAT

Respond with a JSON object:

```json
{
  "Observation": "The Notepad application is open with an empty document.",
  "Thought": "To save the file, I need to use File > Save or Ctrl+S. I'll click the File menu first.",
  "ControlLabel": 3,
  "ControlText": "File",
  "Function": "click",
  "Args": [],
  "Status": "CONTINUE",
  "Plan": ["Click Save in the menu", "Enter filename", "Click Save button"],
  "Comment": "Starting the save workflow"
}
```

## AVAILABLE FUNCTIONS

- **click**: Click the control
- **input_text**: Type text (Args: ["text to type"])
- **select**: Select option from dropdown (Args: ["option"])
- **scroll**: Scroll control (Args: ["up"] or ["down"])
- **hotkey**: Press key combination (Args: ["ctrl", "s"])
- **wait**: Wait for UI update (Args: [seconds])

## STATUS VALUES

- **CONTINUE**: More actions needed
- **FINISH**: Task completed successfully
- **ERROR**: Something went wrong
- **PENDING**: Waiting for user input

## RULES

1. **Always provide Observation and Thought**
2. **ControlLabel must match a labeled element**
3. **Plan should list remaining steps**
4. **Use FINISH only when goal is achieved**"""


# =============================================================================
# System Prompt Registry
# =============================================================================

SYSTEM_PROMPTS = {
    TrackType.TRACK_A: SYSTEM_PROMPT_TRACK_A,
    TrackType.TRACK_B: SYSTEM_PROMPT_TRACK_B,
    TrackType.TRACK_C: SYSTEM_PROMPT_TRACK_C,
}

# Additional format-specific prompts
FORMAT_PROMPTS = {
    ActionOutputFormat.PYAUTOGUI: SYSTEM_PROMPT_OSWORLD,
}


# =============================================================================
# PromptBuilder Class
# =============================================================================


class PromptBuilder:
    """Builds prompts for baseline API calls.

    Constructs system prompts and user content based on track configuration.
    Supports multiple output formats and benchmark compatibility.

    Example:
        builder = PromptBuilder(track_config)
        system = builder.get_system_prompt()
        content = builder.build_user_content(
            goal="Log into the application",
            screenshot=img,
            a11y_tree=tree,
            history=history,
        )
    """

    def __init__(self, track: TrackConfig):
        """Initialize prompt builder.

        Args:
            track: Track configuration.
        """
        self.track = track

    def get_system_prompt(
        self,
        demo: str | None = None,
        custom_instructions: str | None = None,
    ) -> str:
        """Get the system prompt for this track.

        Args:
            demo: Optional demo text to include as an example.
            custom_instructions: Optional custom instructions to append.

        Returns:
            System prompt string.
        """
        # Select base prompt based on format or track
        if self.track.action_format == ActionOutputFormat.PYAUTOGUI:
            base_prompt = SYSTEM_PROMPT_OSWORLD
        else:
            base_prompt = SYSTEM_PROMPTS.get(
                self.track.track_type, SYSTEM_PROMPT_TRACK_A
            )

        parts = [base_prompt]

        # Add demo example if provided
        if demo:
            parts.append(self._format_demo_section(demo))

        # Add screen verification instruction if enabled
        if self.track.verify_after_action:
            parts.append(self._get_verification_instruction())

        # Add custom instructions
        if custom_instructions:
            parts.append(f"\n## ADDITIONAL INSTRUCTIONS\n\n{custom_instructions}")

        return "\n\n".join(parts)

    def _format_demo_section(self, demo: str) -> str:
        """Format demonstration example section."""
        return textwrap.dedent(f"""
        ## EXAMPLE DEMONSTRATION

        Here is an example of successfully completing a similar task:

        {demo}

        Follow a similar pattern for your task.
        """).strip()

    def _get_verification_instruction(self) -> str:
        """Get instruction for post-action verification.

        Based on Claude Computer Use best practices.
        """
        return textwrap.dedent("""
        ## VERIFICATION

        After each action, a new screenshot will be provided. Verify that:
        1. The action was executed correctly
        2. The UI state changed as expected
        3. You are making progress toward the goal

        If something unexpected happened, explain what went wrong and try again.
        """).strip()

    def build_user_content(
        self,
        goal: str,
        screenshot: "Image" | None = None,
        a11y_tree: str | dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        encode_image_fn: Any = None,
        screen_info: dict[str, Any] | None = None,
        window_info: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Build user message content for API call.

        Args:
            goal: Task goal/instruction.
            screenshot: Screenshot image (PIL Image).
            a11y_tree: Accessibility tree (string or dict).
            history: List of previous actions.
            encode_image_fn: Function to encode image for API.
            screen_info: Screen dimensions and other info.
            window_info: Active window information.

        Returns:
            List of content blocks for API message.
        """
        content: list[dict[str, Any]] = []

        # Build text prompt
        text_parts = [self._format_goal(goal)]

        # Add screen info if provided
        if screen_info:
            text_parts.append(self._format_screen_info(screen_info))

        # Add window info if provided
        if window_info:
            text_parts.append(self._format_window_info(window_info))

        # Add accessibility tree if configured
        if self.track.use_a11y_tree and a11y_tree:
            tree_text = self._format_a11y_tree(a11y_tree)
            if tree_text:
                text_parts.append(self._format_a11y_section(tree_text))

        # Add action history if configured
        if self.track.include_history and history:
            history_text = self._format_history(history)
            if history_text:
                text_parts.append(self._format_history_section(history_text))

        # Add instruction based on track
        text_parts.append(self._get_action_instruction())

        # Combine text parts
        content.append({"type": "text", "text": "\n\n".join(text_parts)})

        # Add screenshot if provided
        if screenshot is not None and encode_image_fn is not None:
            content.append(encode_image_fn(screenshot))

        return content

    def _format_goal(self, goal: str) -> str:
        """Format the task goal."""
        return f"## TASK\n\n{goal}"

    def _format_screen_info(self, screen_info: dict[str, Any]) -> str:
        """Format screen information."""
        width = screen_info.get("width", "unknown")
        height = screen_info.get("height", "unknown")
        return f"## SCREEN\n\nResolution: {width} x {height} pixels"

    def _format_window_info(self, window_info: dict[str, Any]) -> str:
        """Format active window information."""
        parts = ["## ACTIVE WINDOW"]

        if "title" in window_info:
            parts.append(f"Title: {window_info['title']}")
        if "app" in window_info:
            parts.append(f"Application: {window_info['app']}")
        if "url" in window_info:
            parts.append(f"URL: {window_info['url']}")

        return "\n".join(parts)

    def _format_a11y_section(self, tree_text: str) -> str:
        """Format accessibility tree section with header."""
        header = "## UI ELEMENTS" if self.track.use_som else "## ACCESSIBILITY TREE"
        return f"{header}\n\n{tree_text}"

    def _format_history_section(self, history_text: str) -> str:
        """Format history section with header."""
        return f"## PREVIOUS ACTIONS\n\n{history_text}"

    def _get_action_instruction(self) -> str:
        """Get instruction for action output based on track."""
        if self.track.track_type == TrackType.TRACK_B:
            return "## YOUR TURN\n\nAnalyze the screenshot, explain your reasoning, and provide the next action."
        elif self.track.track_type == TrackType.TRACK_C:
            return "## YOUR TURN\n\nAnalyze the screenshot and select the appropriate element to interact with."
        else:
            return "## YOUR TURN\n\nAnalyze the screenshot and provide the next action."

    def _format_a11y_tree(self, tree: str | dict[str, Any]) -> str:
        """Format accessibility tree for prompt.

        Args:
            tree: Accessibility tree as string or dict.

        Returns:
            Formatted string (possibly truncated).
        """
        if isinstance(tree, str):
            text = tree
        elif isinstance(tree, dict):
            text = self._dict_to_tree_string(tree)
        else:
            return ""

        # Truncate if needed
        max_lines = self.track.max_a11y_elements
        lines = text.split("\n")
        if len(lines) > max_lines:
            original_count = len(lines)
            lines = lines[:max_lines]
            lines.append(f"... (showing {max_lines} of {original_count} elements)")

        return "\n".join(lines)

    def _dict_to_tree_string(
        self,
        tree: dict[str, Any],
        indent: int = 0,
        max_depth: int = 5,
    ) -> str:
        """Convert dict tree to formatted string.

        Args:
            tree: Dictionary representing accessibility tree.
            indent: Current indentation level.
            max_depth: Maximum recursion depth.

        Returns:
            Formatted tree string.
        """
        if indent > max_depth:
            return ""

        lines = []
        prefix = "  " * indent

        role = tree.get("role", "unknown")
        name = tree.get("name", "")
        node_id = tree.get("id", tree.get("node_id", ""))

        # Format node based on track
        if self.track.use_som and node_id:
            # SoM format: [id] role "name"
            line = f"{prefix}[{node_id}] {role}"
        elif node_id:
            # Non-SoM with ID
            line = f"{prefix}({node_id}) {role}"
        else:
            line = f"{prefix}{role}"

        if name:
            # Truncate long names
            if len(name) > 50:
                name = name[:47] + "..."
            line += f': "{name}"'

        # Add bounding box if available (useful for debugging)
        bbox = tree.get("bbox", tree.get("bounds"))
        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            # Show center point for SoM
            if self.track.use_som:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                line += f" @ ({cx:.2f}, {cy:.2f})"

        lines.append(line)

        # Process children
        children = tree.get("children", [])
        for child in children:
            if isinstance(child, dict):
                child_text = self._dict_to_tree_string(child, indent + 1, max_depth)
                if child_text:
                    lines.append(child_text)

        return "\n".join(lines)

    def _format_history(self, history: list[dict[str, Any]]) -> str:
        """Format action history for prompt.

        Args:
            history: List of action dictionaries.

        Returns:
            Formatted history string.
        """
        if not history:
            return ""

        lines = []
        max_steps = self.track.max_history_steps
        recent = history[-max_steps:] if len(history) > max_steps else history

        for i, action in enumerate(recent, 1):
            action_type = action.get("type", action.get("action", "unknown")).upper()
            line = self._format_single_action(i, action_type, action)
            lines.append(line)

        return "\n".join(lines)

    def _format_single_action(
        self, step: int, action_type: str, action: dict[str, Any]
    ) -> str:
        """Format a single action for history display."""
        if action_type == "CLICK":
            if "element_id" in action:
                return f"{step}. CLICK([{action['element_id']}])"
            elif "x" in action and "y" in action:
                return f"{step}. CLICK({action['x']:.3f}, {action['y']:.3f})"
            else:
                return f"{step}. CLICK()"
        elif action_type == "TYPE":
            text = action.get("text", "")
            # Truncate long text
            if len(text) > 30:
                text = text[:27] + "..."
            return f'{step}. TYPE("{text}")'
        elif action_type == "KEY":
            key = action.get("key", "")
            return f"{step}. KEY({key})"
        elif action_type == "SCROLL":
            direction = action.get("direction", "down")
            amount = action.get("amount", 1)
            return f"{step}. SCROLL({direction}, {amount})"
        elif action_type == "DONE":
            return f"{step}. DONE()"
        elif action_type == "WAIT":
            return f"{step}. WAIT()"
        else:
            return f"{step}. {action_type}()"

    def build_verification_prompt(
        self,
        goal: str,
        previous_action: dict[str, Any],
        screenshot: "Image" | None = None,
        encode_image_fn: Any = None,
    ) -> list[dict[str, Any]]:
        """Build a verification prompt after an action.

        Used to verify action results and decide next steps.
        Based on Claude Computer Use best practices.

        Args:
            goal: Original task goal.
            previous_action: The action that was just executed.
            screenshot: Screenshot after action execution.
            encode_image_fn: Function to encode image.

        Returns:
            List of content blocks.
        """
        content: list[dict[str, Any]] = []

        action_str = self._format_single_action(
            0, previous_action.get("type", ""), previous_action
        )
        action_str = action_str[3:]  # Remove "0. " prefix

        text = textwrap.dedent(f"""
        ## VERIFICATION CHECK

        **Goal**: {goal}

        **Previous Action**: {action_str}

        Analyze the screenshot and verify:
        1. Did the action execute correctly?
        2. Is the UI state as expected?
        3. Are we making progress toward the goal?

        If the goal is achieved, respond with {{"action": "DONE"}}.
        Otherwise, provide the next action.
        """).strip()

        content.append({"type": "text", "text": text})

        if screenshot is not None and encode_image_fn is not None:
            content.append(encode_image_fn(screenshot))

        return content
