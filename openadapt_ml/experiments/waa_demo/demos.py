"""Demonstrations for WAA tasks.

Each demo follows the format:
    DEMONSTRATION:
    Goal: [Task instruction]

    Step N:
      [Screen: UI description]
      [Action: ACTION_TYPE(parameters)]
      [Result: What changed]

7 manual demos + 3 placeholder for recorded demos.
"""

from __future__ import annotations

from typing import Optional

# =============================================================================
# TASK 1: Enable Do Not Track in Edge
# =============================================================================
DEMO_1_DO_NOT_TRACK = """DEMONSTRATION:
Goal: Enable the 'Do Not Track' feature in Edge

Step 1:
  [Screen: Microsoft Edge browser with a webpage open. Three-dot menu icon visible in top-right corner of toolbar.]
  [Action: CLICK(three-dot menu icon "Settings and more" in top-right)]
  [Result: Dropdown menu appears with options like New tab, New window, Settings, etc.]

Step 2:
  [Screen: Edge dropdown menu open showing various options]
  [Action: CLICK("Settings" option in the menu)]
  [Result: Settings page opens in a new tab]

Step 3:
  [Screen: Edge Settings page with left sidebar showing categories: Profiles, Privacy/security/services, Appearance, etc.]
  [Action: CLICK("Privacy, search, and services" in left sidebar)]
  [Result: Privacy settings panel displays on right side]

Step 4:
  [Screen: Privacy settings panel with sections for Tracking prevention, Privacy, Security]
  [Action: SCROLL(down) to find "Privacy" section with "Send 'Do Not Track' requests" toggle]
  [Result: "Send 'Do Not Track' requests" toggle becomes visible]

Step 5:
  [Screen: Privacy section visible with "Send 'Do Not Track' requests" toggle currently OFF]
  [Action: CLICK(toggle switch next to "Send 'Do Not Track' requests")]
  [Result: Toggle switches to ON position, Do Not Track is now enabled]
"""

# =============================================================================
# TASK 2: Save Webpage to Bookmarks Bar
# =============================================================================
DEMO_2_BOOKMARK = """DEMONSTRATION:
Goal: Save this webpage to bookmarks bar

Step 1:
  [Screen: Microsoft Edge browser with a webpage loaded. Star icon visible in the address bar on the right side.]
  [Action: CLICK(star icon in address bar)]
  [Result: "Favorite added" popup appears with name field and folder dropdown]

Step 2:
  [Screen: Bookmark popup showing Name field (auto-filled with page title), Folder dropdown, and Done/Remove buttons]
  [Action: CLICK(Folder dropdown to expand it)]
  [Result: Dropdown shows folder options including "Favorites bar", "Other favorites", etc.]

Step 3:
  [Screen: Folder dropdown expanded showing available bookmark folders]
  [Action: CLICK("Favorites bar" option)]
  [Result: "Favorites bar" is selected as the target folder]

Step 4:
  [Screen: Bookmark popup with "Favorites bar" selected as folder]
  [Action: CLICK("Done" button)]
  [Result: Bookmark is saved to favorites bar, popup closes, page bookmark icon now filled]
"""

# =============================================================================
# TASK 3: Set Font Size to Largest in Edge
# =============================================================================
DEMO_3_FONT_SIZE = """DEMONSTRATION:
Goal: Set default font size to largest for grandmother

Step 1:
  [Screen: Microsoft Edge browser open. Three-dot menu icon in top-right corner.]
  [Action: CLICK(three-dot menu icon "Settings and more")]
  [Result: Dropdown menu appears]

Step 2:
  [Screen: Edge dropdown menu visible]
  [Action: CLICK("Settings" in the menu)]
  [Result: Settings page opens]

Step 3:
  [Screen: Edge Settings page with left sidebar showing categories]
  [Action: CLICK("Appearance" in left sidebar)]
  [Result: Appearance settings panel displays]

Step 4:
  [Screen: Appearance settings showing Fonts section with "Font size" dropdown showing current size (e.g., "Medium")]
  [Action: CLICK(Font size dropdown)]
  [Result: Dropdown expands showing options: Very small, Small, Medium, Large, Very large]

Step 5:
  [Screen: Font size dropdown expanded with size options visible]
  [Action: CLICK("Very large" option)]
  [Result: Font size is set to Very large, dropdown closes, text on pages will now appear larger]
"""

# =============================================================================
# TASK 4: Fill Blank Cells - PLACEHOLDER (requires recorded demo)
# =============================================================================
DEMO_4_FILL_BLANKS = """DEMONSTRATION:
Goal: Fill all blank cells with value from cell above

[PLACEHOLDER - This demo requires recording on Windows]
[Complex task involving: Select range, Go To Special, Blanks, Formula, Ctrl+Enter]

Key steps outline:
1. Select the data range containing blank cells
2. Press Ctrl+G or go to Edit > Go To
3. Click "Special" button
4. Select "Empty cells" option
5. Click OK to select all blank cells
6. Type formula =A1 (or reference to cell above)
7. Press Ctrl+Enter to fill all selected cells
"""

# =============================================================================
# TASK 5: Calculate Totals and Create Chart - PLACEHOLDER (requires recorded demo)
# =============================================================================
DEMO_5_CHART = """DEMONSTRATION:
Goal: Calculate monthly totals and create line chart

[PLACEHOLDER - This demo requires recording on Windows]
[Complex task involving: SUM formulas, data selection, chart wizard, line chart type]

Key steps outline:
1. Navigate to the row below the data
2. Type "Total" in label column
3. Enter =SUM() formula for first data column
4. Copy formula across all columns
5. Select entire data range including headers and totals
6. Go to Insert > Chart
7. Select "Line" chart type
8. Configure chart options and finish
"""

# =============================================================================
# TASK 6: Center Align Heading in LibreOffice Writer
# =============================================================================
DEMO_6_CENTER_ALIGN = """DEMONSTRATION:
Goal: Center align the heading in LibreOffice Writer

Step 1:
  [Screen: LibreOffice Writer document open with text content. Heading text visible at top. Formatting toolbar shows alignment buttons (left, center, right, justify).]
  [Action: CLICK(anywhere in the heading line to place cursor)]
  [Result: Cursor is now in the heading paragraph]

Step 2:
  [Screen: Cursor is in heading line. Center align button (lines centered icon) visible in toolbar.]
  [Action: CLICK(Center align button in formatting toolbar)]
  [Result: Heading text is now centered on the page]

Alternative using keyboard:
Step 1:
  [Action: CLICK(in heading line)]
Step 2:
  [Action: KEY(Ctrl+E)]
  [Result: Heading is centered]
"""

# =============================================================================
# TASK 7: Turn Off System Notifications
# =============================================================================
DEMO_7_NOTIFICATIONS = """DEMONSTRATION:
Goal: Turn off notifications for system

Step 1:
  [Screen: Windows 11 desktop with taskbar at bottom. Start button visible on left side of taskbar.]
  [Action: CLICK(Start button in taskbar)]
  [Result: Start menu opens showing pinned apps and search bar]

Step 2:
  [Screen: Start menu open with search bar at top, pinned apps below]
  [Action: CLICK(Settings gear icon in pinned apps) OR TYPE("Settings") in search]
  [Result: Settings app launches]

Step 3:
  [Screen: Windows Settings app open showing main categories: System, Bluetooth, Network, etc.]
  [Action: CLICK("System" category)]
  [Result: System settings panel opens]

Step 4:
  [Screen: System settings showing options: Display, Sound, Notifications, Focus, Power, etc.]
  [Action: CLICK("Notifications" in the list)]
  [Result: Notifications settings page opens]

Step 5:
  [Screen: Notifications settings showing main toggle "Notifications" at top, followed by per-app settings]
  [Action: CLICK(main "Notifications" toggle switch to turn it OFF)]
  [Result: Toggle switches to Off, all system notifications are now disabled]
"""

# =============================================================================
# TASK 8: Configure Night Light Schedule
# =============================================================================
DEMO_8_NIGHT_LIGHT = """DEMONSTRATION:
Goal: Enable the "Night light" feature and set it to turn on at 7:00 PM and off at 7:00 AM

Step 1:
  [Screen: Windows 11 desktop with taskbar visible]
  [Action: CLICK(Start button) OR press KEY(Windows key)]
  [Result: Start menu opens]

Step 2:
  [Screen: Start menu open]
  [Action: CLICK(Settings gear icon)]
  [Result: Settings app opens]

Step 3:
  [Screen: Windows Settings main page with categories]
  [Action: CLICK("System" on the left)]
  [Result: System settings displayed]

Step 4:
  [Screen: System settings with Display highlighted at top]
  [Action: CLICK("Display" option)]
  [Result: Display settings panel opens]

Step 5:
  [Screen: Display settings showing Night light option with toggle and ">" arrow for more options]
  [Action: CLICK("Night light" row or the arrow to expand settings)]
  [Result: Night light settings panel opens]

Step 6:
  [Screen: Night light settings showing: Turn on now button, Strength slider, Schedule night light toggle]
  [Action: CLICK("Schedule night light" toggle to enable it)]
  [Result: Schedule options appear with Turn on/Turn off time fields]

Step 7:
  [Screen: Schedule section now visible with "Turn on" time field and "Turn off" time field]
  [Action: CLICK("Turn on" time field)]
  [Result: Time picker opens or field becomes editable]

Step 8:
  [Screen: Time picker or editable time field for Turn on time]
  [Action: TYPE("7:00 PM") or use time picker to set 7:00 PM]
  [Result: Turn on time is set to 7:00 PM]

Step 9:
  [Screen: Turn on time set to 7:00 PM, Turn off time field visible]
  [Action: CLICK("Turn off" time field)]
  [Result: Time picker opens for Turn off time]

Step 10:
  [Screen: Time picker for Turn off time]
  [Action: TYPE("7:00 AM") or use time picker to set 7:00 AM]
  [Result: Turn off time is set to 7:00 AM, Night light schedule is fully configured]
"""

# =============================================================================
# TASK 9: Create Archive Folder and Move Files - PLACEHOLDER (requires recorded demo)
# =============================================================================
DEMO_9_ARCHIVE = """DEMONSTRATION:
Goal: Create Archive folder and move all .docx files

[PLACEHOLDER - This demo requires recording on Windows]
[Medium complexity: right-click context menu, folder creation, file selection, drag/move]

Key steps outline:
1. In File Explorer, right-click on empty space in the current folder
2. Select "New" > "Folder" from context menu
3. Type "Archive" as folder name, press Enter
4. Click on a .docx file to start selection
5. Hold Ctrl and click each .docx file to select all
   OR use filter/search to show only .docx files then Ctrl+A
6. Right-click on selection and choose "Cut" (or Ctrl+X)
7. Double-click Archive folder to open it
8. Right-click and "Paste" (or Ctrl+V)
"""

# =============================================================================
# TASK 10: Change to Details View in File Explorer
# =============================================================================
DEMO_10_DETAILS_VIEW = """DEMONSTRATION:
Goal: Change view to Details view

Step 1:
  [Screen: File Explorer window open showing files/folders. Toolbar at top with View button/dropdown visible.]
  [Action: CLICK("View" button or dropdown in the toolbar)]
  [Result: View options appear showing: Extra large icons, Large icons, Medium icons, Small icons, List, Details, Tiles, Content]

Step 2:
  [Screen: View options dropdown/menu expanded showing all view choices]
  [Action: CLICK("Details" option)]
  [Result: File Explorer switches to Details view showing columns: Name, Date modified, Type, Size]

Alternative (Windows 11 compact view):
Step 1:
  [Screen: File Explorer with Layout section in View menu]
  [Action: CLICK(View dropdown arrow in toolbar)]
Step 2:
  [Action: CLICK("Details" in the layout options)]
  [Result: View changes to Details with file information columns]
"""

# =============================================================================
# Demo Registry
# =============================================================================
DEMOS: dict[str, str] = {
    "1": DEMO_1_DO_NOT_TRACK,
    "2": DEMO_2_BOOKMARK,
    "3": DEMO_3_FONT_SIZE,
    "4": DEMO_4_FILL_BLANKS,  # Placeholder
    "5": DEMO_5_CHART,  # Placeholder
    "6": DEMO_6_CENTER_ALIGN,
    "7": DEMO_7_NOTIFICATIONS,
    "8": DEMO_8_NIGHT_LIGHT,
    "9": DEMO_9_ARCHIVE,  # Placeholder
    "10": DEMO_10_DETAILS_VIEW,
}


def get_demo(task_num: str | int) -> Optional[str]:
    """Get a demo by task number (1-10)."""
    return DEMOS.get(str(task_num))


def get_complete_demos() -> dict[str, str]:
    """Get only demos that are fully written (not placeholders)."""
    return {k: v for k, v in DEMOS.items() if "[PLACEHOLDER" not in v}


def get_placeholder_demos() -> dict[str, str]:
    """Get demos that need to be recorded."""
    return {k: v for k, v in DEMOS.items() if "[PLACEHOLDER" in v}


def format_demo_for_prompt(demo: str, task_instruction: str) -> str:
    """Format a demo for inclusion in an LLM prompt.

    Args:
        demo: The demo text
        task_instruction: The current task instruction

    Returns:
        Formatted demo with context header
    """
    return f"""The following demonstration shows how to complete a similar task.
Use it as a reference for understanding the UI navigation pattern.

{demo}

---
Now complete this task: {task_instruction}
"""
