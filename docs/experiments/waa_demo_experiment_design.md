# WAA Demo-Conditioned Experiment Design (Option A: Minimal)

**Date**: January 2026
**Author**: OpenAdapt Team
**Status**: Design Complete - Pending Execution

## Executive Summary

This document defines a minimal demo-conditioned experiment on Windows Agent Arena (WAA) to validate whether demonstrations improve task success on Windows desktop tasks, extending the validated macOS results (+53pp first-action accuracy).

**Experiment scope**: 10 carefully selected tasks across 4 enterprise-relevant domains, evaluated under zero-shot and demo-conditioned conditions.

---

## Task Selection Criteria

Tasks were selected based on:

1. **Diverse first actions** - Not all "click Start menu" or "click Settings"
2. **Enterprise relevance** - Tasks common in business workflows
3. **Clear success criteria** - Programmatic evaluation possible
4. **Varying difficulty** - Mix of simple (2-3 steps) and complex (5+ steps)
5. **Different UI starting states** - Some start from desktop, some from within apps

---

## Selected Tasks (10 Tasks)

### Browser/Edge Domain (3 tasks)

| # | Task ID | Instruction | First Action | Difficulty | Why Selected |
|---|---------|-------------|--------------|------------|--------------|
| 1 | `004587f8-6028-4656-94c1-681481abbc9c-wos` | "Enable the 'Do Not Track' feature in Edge" | Edge is launched; navigate to Settings > Privacy | Medium | Common enterprise privacy requirement; starts from Edge open (not desktop) |
| 2 | `049d3788-c979-4ea6-934d-3a35c4630faf-WOS` | "Save this webpage to bookmarks bar" | Click star/bookmark icon or use Ctrl+D | Easy | Frequent user action; starts with page open |
| 3 | `2acd62b4-a2ab-44a7-a7e3-f5227bbd8324-wos` | "Set default font size to largest for grandmother" | Edge Settings > Appearance > Font size | Medium | Accessibility task; tests navigation depth |

### Office/LibreOffice Domain (3 tasks)

| # | Task ID | Instruction | First Action | Difficulty | Why Selected |
|---|---------|-------------|--------------|------------|--------------|
| 4 | `01b269ae-2111-4a07-81fd-3fcd711993b0-WOS` | "Fill all blank cells with value from cell above" | Select cells, use Go To Special > Blanks | Hard | Complex data operation; document already open |
| 5 | `0a2e43bf-b26c-4631-a966-af9dfa12c9e5-WOS` | "Calculate monthly totals and create line chart" | Click cell for SUM formula | Hard | Multi-step data + visualization; enterprise analytics |
| 6 | `3ef2b351-8a84-4ff2-8724-d86eae9b842e-WOS` | "Center align the heading in LibreOffice Writer" | Select text, click center align button | Easy | Simple formatting; document already open |

### Settings Domain (2 tasks)

| # | Task ID | Instruction | First Action | Difficulty | Why Selected |
|---|---------|-------------|--------------|------------|--------------|
| 7 | `37e10fc4-b4c5-4b02-a65c-bfae8bc51d3f-wos` | "Turn off notifications for system" | Click Start or search for Settings | Medium | IT admin task; starts from desktop |
| 8 | `46adf721-2949-4426-b069-010b7c128d8f-wos` | "Enable Night Light: on at 7PM, off at 7AM" | Click Start > Settings > Display | Medium | Exact match to our validated macOS demo (Night Shift); good transfer test |

### File Explorer Domain (2 tasks)

| # | Task ID | Instruction | First Action | Difficulty | Why Selected |
|---|---------|-------------|--------------|------------|--------------|
| 9 | `0c9dda13-428c-492b-900b-f48562111f93-WOS` | "Create Archive folder and move all .docx files" | Explorer already open; right-click > New Folder | Medium | File management; Explorer already launched |
| 10 | `34a4fee9-e52e-4a4a-96d2-68d35091504a-WOS` | "Change view to Details view" | Click View menu or View dropdown | Easy | UI preference change; Explorer already open |

---

## Task Diversity Analysis

### First Action Diversity

| First Action Category | Tasks | Count |
|-----------------------|-------|-------|
| Navigate to Settings menu | #1, #3 | 2 |
| Click UI button (bookmark/align/view) | #2, #6, #10 | 3 |
| Select cells in spreadsheet | #4, #5 | 2 |
| Open Start menu or search | #7, #8 | 2 |
| Right-click context menu | #9 | 1 |

**Result**: 5 distinct first action categories across 10 tasks.

### Starting State Diversity

| Starting State | Tasks | Count |
|----------------|-------|-------|
| Desktop (no app open) | #7, #8 | 2 |
| Edge browser open | #1, #2, #3 | 3 |
| LibreOffice Calc open with file | #4, #5 | 2 |
| LibreOffice Writer open with file | #6 | 1 |
| File Explorer open | #9, #10 | 2 |

**Result**: 5 distinct starting states.

### Difficulty Distribution

| Difficulty | Tasks | Count |
|------------|-------|-------|
| Easy (1-2 steps) | #2, #6, #10 | 3 |
| Medium (3-5 steps) | #1, #3, #7, #8, #9 | 5 |
| Hard (5+ steps, multi-modal) | #4, #5 | 2 |

---

## Demo Format Specification

Each demo should follow this structured format:

```
DEMONSTRATION:
Goal: [Exact task instruction]

Step 1:
  [Screenshot description: key UI elements visible]
  [Action: ACTION_TYPE(parameters)]
  [Result: what changed after this action]

Step 2:
  ...
```

### Demo Content Requirements

For each of the 10 selected tasks, the demo must include:

1. **Complete action sequence** - All steps from initial state to task completion
2. **Screenshot descriptions** - Text describing key UI elements at each step
3. **Explicit actions** - Using the action format:
   - `CLICK(x, y)` or `CLICK([element_id])` for SoM
   - `TYPE("text")` for keyboard input
   - `KEY(key_name)` for special keys
   - `SCROLL(direction)` for scrolling
4. **Result descriptions** - What changed after each action

### Example Demo Format (Task #8: Night Light)

```
DEMONSTRATION:
Goal: Enable the "Night light" feature and set it to turn on at 7:00 PM and off at 7:00 AM.

Step 1:
  [Screen: Windows 11 desktop with taskbar visible]
  [Action: CLICK(Start button in taskbar)]
  [Result: Start menu opened]

Step 2:
  [Screen: Start menu with search bar and pinned apps]
  [Action: CLICK(Settings gear icon) OR TYPE("Settings")]
  [Result: Settings app launched]

Step 3:
  [Screen: Settings app showing main categories]
  [Action: CLICK("System" category)]
  [Result: System settings panel shown]

Step 4:
  [Screen: System settings with Display highlighted]
  [Action: CLICK("Display" in left sidebar)]
  [Result: Display settings shown with Night light option]

Step 5:
  [Screen: Display settings showing Night light toggle]
  [Action: CLICK("Night light" toggle or settings link)]
  [Result: Night light settings expanded/opened]

Step 6:
  [Screen: Night light settings with schedule options]
  [Action: CLICK("Turn on" checkbox for scheduling)]
  [Result: Schedule inputs become active]

Step 7:
  [Screen: Night light schedule settings visible]
  [Action: CLICK/TYPE to set "Turn on" time to 7:00 PM]
  [Result: Turn on time set to 7:00 PM]

Step 8:
  [Screen: Night light schedule with turn on time set]
  [Action: CLICK/TYPE to set "Turn off" time to 7:00 AM]
  [Result: Turn off time set to 7:00 AM, task complete]
```

---

## Demo Requirements Per Task

### Task 1: Enable Do Not Track (Edge)

**Key steps demo must show**:
1. Click three-dot menu (Settings and more)
2. Click "Settings" from dropdown
3. Click "Privacy, search, and services" in sidebar
4. Scroll to "Privacy" section
5. Toggle "Send 'Do Not Track' requests"

**Critical disambiguation**: Edge has multiple entry points to settings; demo should show the canonical path.

### Task 2: Save to Bookmarks Bar (Edge)

**Key steps demo must show**:
1. Click star icon in address bar OR press Ctrl+D
2. Ensure "Favorites bar" is selected as folder
3. Click "Done"

**Critical disambiguation**: Bookmark vs Favorites terminology; ensure bar not sidebar.

### Task 3: Increase Font Size (Edge)

**Key steps demo must show**:
1. Open Settings (three-dot menu > Settings)
2. Click "Appearance" in sidebar
3. Locate "Font size" dropdown
4. Select "Very large" or largest option

**Critical disambiguation**: Font size vs page zoom - they are different settings.

### Task 4: Fill Blank Cells (LibreOffice Calc)

**Key steps demo must show**:
1. Select the data range (A1:Bx or appropriate range)
2. Go to Edit > Go To (or Ctrl+G)
3. Click "Special" button
4. Select "Empty cells"
5. Type formula `=A1` (or relative reference to cell above)
6. Press Ctrl+Enter to fill all selected

**Critical disambiguation**: This requires knowing the "Go To Special" feature; zero-shot likely fails.

### Task 5: Calculate Totals and Create Chart (LibreOffice Calc)

**Key steps demo must show**:
1. Navigate to row below data
2. Type "Total" label
3. Enter SUM formula for first column
4. Copy formula across columns
5. Select data range including totals
6. Insert > Chart
7. Select "Line" chart type
8. Configure X-axis as months

**Critical disambiguation**: Must create both totals AND chart; two distinct sub-tasks.

### Task 6: Center Align Heading (LibreOffice Writer)

**Key steps demo must show**:
1. Click in heading line (or select heading text)
2. Click center align button in toolbar OR use Ctrl+E

**Critical disambiguation**: Center align paragraph vs text alignment; heading vs selected text.

### Task 7: Turn Off System Notifications (Settings)

**Key steps demo must show**:
1. Open Settings (Start > Settings or Win+I)
2. Click "System" category
3. Click "Notifications" in sidebar
4. Toggle off main "Notifications" switch

**Critical disambiguation**: System-wide toggle vs per-app notifications.

### Task 8: Configure Night Light Schedule (Settings)

**Key steps demo must show**:
1. Open Settings
2. Navigate to System > Display
3. Click "Night light" settings
4. Enable scheduling
5. Set "Turn on" time to 7:00 PM
6. Set "Turn off" time to 7:00 AM

**Critical disambiguation**: Must set BOTH times; schedule mode vs manual toggle.

### Task 9: Create Archive Folder and Move Files (File Explorer)

**Key steps demo must show**:
1. Right-click in Documents folder empty space
2. Select "New" > "Folder"
3. Name it "Archive"
4. Select all .docx files (Ctrl+Click or filter)
5. Cut or drag files to Archive folder

**Critical disambiguation**: Create folder first, then move; not copy.

### Task 10: Change to Details View (File Explorer)

**Key steps demo must show**:
1. Click "View" menu in File Explorer toolbar
2. Select "Details" from view options

**Critical disambiguation**: "Details" vs "List" vs "Tiles" - specific view type matters.

---

## Evaluation Protocol

### Conditions

| Condition | Description |
|-----------|-------------|
| **Zero-shot** | Task instruction + initial screenshot only |
| **Demo-conditioned** | Task instruction + formatted demonstration + initial screenshot |
| **Control (optional)** | Task instruction + length-matched irrelevant text + screenshot |

### Metrics

#### Primary Metric: Episode Success Rate

```
episode_success = 1 if task_evaluator_passes else 0
success_rate = sum(episode_success) / n_tasks
```

WAA provides programmatic evaluators for each task (e.g., `exact_match`, `compare_table`, `is_expected_bookmarks`).

#### Secondary Metrics

1. **First Action Accuracy**: Does the first action match expected first step?
2. **Steps to Completion**: How many steps before correct completion (or timeout)?
3. **Error Categories**: Type of errors (wrong element, wrong sequence, stuck/looping)

### Evaluation Procedure

1. **Setup**: Launch WAA environment with task config
2. **Prompt model**: Provide instruction (+ demo for conditioned) + screenshot
3. **Execute action**: Send predicted action to WAA
4. **Capture result**: New screenshot + task state
5. **Repeat**: Until DONE action or max steps (15)
6. **Evaluate**: Run task-specific evaluator
7. **Record**: Success/failure + action trace

### Max Steps

15 steps maximum per task (WAA default), then timeout failure.

---

## Expected Baseline

### WAA SOTA Context

| Model | Success Rate | Source |
|-------|-------------|--------|
| GPT-4V + OmniParser | ~19.5% | WAA paper (2024) |
| Claude Sonnet 4 | TBD | Our baseline run |
| GPT-5.1 + OmniParser | ~19.5% | WAA leaderboard |

### Expected Results

Based on our macOS experiment (+53pp with demos):

| Condition | Expected Success Rate | Rationale |
|-----------|----------------------|-----------|
| **Zero-shot** | 10-20% (1-2/10) | Consistent with WAA SOTA |
| **Demo-conditioned** | 40-60% (4-6/10) | +30-40pp improvement |
| **Best case** | 70%+ (7/10) | If demo quality is high |

### Why Not 100%?

1. **Multi-step degradation**: Errors compound over 5-10 steps
2. **Coordinate precision**: Windows UI requires precise clicks
3. **Dynamic elements**: Exact UI may differ from demo screenshots
4. **Complex tasks**: Tasks #4, #5 require domain knowledge beyond navigation

---

## Task-Level Predictions

| Task | Zero-shot | Demo-conditioned | Reasoning |
|------|-----------|------------------|-----------|
| #1 Do Not Track | Fail | Pass | Navigation path is learnable |
| #2 Bookmark | Pass | Pass | Simple, visible UI |
| #3 Font Size | Fail | Pass | Hidden setting, demo needed |
| #4 Fill Blanks | Fail | Uncertain | Requires advanced Excel knowledge |
| #5 Chart | Fail | Uncertain | Multi-step complexity |
| #6 Center Align | Pass | Pass | Simple, toolbar visible |
| #7 Notifications | Uncertain | Pass | Navigation learnable |
| #8 Night Light | Fail | Pass | Direct transfer from macOS demo |
| #9 Archive Folder | Uncertain | Pass | Multi-step file ops |
| #10 Details View | Pass | Pass | Simple view menu |

**Zero-shot expected**: 2-4/10 (20-40%)
**Demo-conditioned expected**: 6-8/10 (60-80%)

---

## Implementation Notes

### Files to Create

```
openadapt_ml/experiments/waa_demo/
    __init__.py
    tasks.py           # Task definitions from this doc
    demos.py           # Demo content for each task
    runner.py          # Experiment runner
    results/           # Output directory

scripts/
    run_waa_demo_experiment.py  # CLI entry point
```

### Dependencies

- WAA environment (VM or Azure)
- Task config files from `vendor/WindowsAgentArena/`
- VLM API access (Claude/GPT-4V)

### Estimated Runtime

- Per task: 2-5 minutes (depends on step count)
- 10 tasks x 2 conditions = 20 runs
- Total: ~1-2 hours per complete experiment run

---

## Next Steps

1. **Write demos** for all 10 tasks (behavior-only format)
2. **Validate task selection** - ensure all tasks run correctly in WAA
3. **Run zero-shot baseline** - establish baseline before demos
4. **Run demo-conditioned** - with high-quality demos
5. **Analyze results** - compare to predictions, document findings

---

## Appendix: Full Task JSON Locations

| Task | File Path |
|------|-----------|
| #1 | `vendor/WindowsAgentArena/src/win-arena-container/client/evaluation_examples_windows/examples/msedge/004587f8-6028-4656-94c1-681481abbc9c-wos.json` |
| #2 | `vendor/WindowsAgentArena/src/win-arena-container/client/evaluation_examples_windows/examples/msedge/049d3788-c979-4ea6-934d-3a35c4630faf-WOS.json` |
| #3 | `vendor/WindowsAgentArena/src/win-arena-container/client/evaluation_examples_windows/examples/msedge/2acd62b4-a2ab-44a7-a7e3-f5227bbd8324-wos.json` |
| #4 | `vendor/WindowsAgentArena/src/win-arena-container/client/evaluation_examples_windows/examples/libreoffice_calc/01b269ae-2111-4a07-81fd-3fcd711993b0-WOS.json` |
| #5 | `vendor/WindowsAgentArena/src/win-arena-container/client/evaluation_examples_windows/examples/libreoffice_calc/0a2e43bf-b26c-4631-a966-af9dfa12c9e5-WOS.json` |
| #6 | `vendor/WindowsAgentArena/src/win-arena-container/client/evaluation_examples_windows/examples/libreoffice_writer/3ef2b351-8a84-4ff2-8724-d86eae9b842e-WOS.json` |
| #7 | `vendor/WindowsAgentArena/src/win-arena-container/client/evaluation_examples_windows/examples/settings/37e10fc4-b4c5-4b02-a65c-bfae8bc51d3f-wos.json` |
| #8 | `vendor/WindowsAgentArena/src/win-arena-container/client/evaluation_examples_windows/examples/settings/46adf721-2949-4426-b069-010b7c128d8f-wos.json` |
| #9 | `vendor/WindowsAgentArena/src/win-arena-container/client/evaluation_examples_windows/examples/file_explorer/0c9dda13-428c-492b-900b-f48562111f93-WOS.json` |
| #10 | `vendor/WindowsAgentArena/src/win-arena-container/client/evaluation_examples_windows/examples/file_explorer/34a4fee9-e52e-4a4a-96d2-68d35091504a-WOS.json` |
