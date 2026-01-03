# WAA Demo Recording Guide

This guide explains how to record demonstrations on Windows for the WAA demo-conditioned experiment.

## Overview

We need 3 recorded demos for tasks with complex UI interactions that are difficult to describe accurately in text:

| Task | Instruction | Why Record |
|------|-------------|------------|
| **#4** | Fill all blank cells with value from cell above | Go To Special dialog, formula entry, Ctrl+Enter |
| **#5** | Calculate monthly totals and create line chart | SUM formulas, chart wizard, multi-step |
| **#9** | Create Archive folder and move all .docx files | Context menu, file selection, cut/paste |

## Prerequisites

### 1. Install openadapt-capture

```powershell
# Option A: pip install (if published)
pip install openadapt-capture

# Option B: Clone and install from source
git clone https://github.com/OpenAdaptAI/openadapt-capture.git
cd openadapt-capture
pip install -e .
```

### 2. Install LibreOffice (for tasks #4 and #5)

Download from: https://www.libreoffice.org/download/download/

Tasks #4 and #5 use LibreOffice Calc, not Excel (WAA uses LibreOffice).

## Recording Instructions

### General Tips

1. **Clean desktop**: Close unnecessary windows before recording
2. **Slow and deliberate**: Move mouse slowly, pause briefly before clicking
3. **Visible actions**: Make sure each click/keystroke is distinct
4. **Narrate mentally**: Think "I'm clicking the File menu" as you do it

### Task #4: Fill Blank Cells with Value Above

**Setup before recording**:
1. Open LibreOffice Calc
2. Create a spreadsheet with some data and intentional blank cells:
   ```
   A       B
   --------
   Apple   10

   Banana  20

   Cherry  30
   ```

**Recording steps**:
1. Start openadapt-capture: `openadapt-capture record --name fill-blanks`
2. Select the data range (e.g., A1:B6)
3. Press `Ctrl+G` or go to **Sheet** > **Navigate** > **Go To Cell**
4. Click **Special...** button (or use **Edit** > **Go To** > **Special** in some versions)
5. Select **Empty cells** option
6. Click **OK** - all blank cells are now selected
7. Type the formula `=A1` (or reference to cell directly above)
8. Press `Ctrl+Enter` to fill all selected cells
9. Stop recording: `Ctrl+C` in terminal

**Expected result**: Blank cells now contain the value from the cell above them.

### Task #5: Calculate Monthly Totals and Create Line Chart

**Setup before recording**:
1. Open LibreOffice Calc
2. Create a spreadsheet with monthly data:
   ```
   Month    Sales    Expenses
   Jan      1000     800
   Feb      1200     900
   Mar      1500     1000
   Apr      1100     850
   ```

**Recording steps**:
1. Start openadapt-capture: `openadapt-capture record --name create-chart`
2. Click in the cell below the last data row (e.g., A6)
3. Type "Total" and press Tab
4. Type `=SUM(B2:B5)` and press Tab
5. Type `=SUM(C2:C5)` and press Enter
6. Select the entire data range including headers and totals (A1:C6)
7. Go to **Insert** > **Chart...**
8. In the Chart Wizard:
   - Select **Line** chart type
   - Click **Next** to configure data range
   - Click **Next** to configure data series
   - Click **Next** to add title if desired
   - Click **Finish**
9. Stop recording: `Ctrl+C` in terminal

**Expected result**: A line chart showing Sales and Expenses trends is embedded in the spreadsheet.

### Task #9: Create Archive Folder and Move .docx Files

**Setup before recording**:
1. Open File Explorer
2. Navigate to a folder containing some .docx files (create test files if needed):
   ```
   Documents/
     report.docx
     notes.docx
     draft.docx
     image.png
     data.xlsx
   ```

**Recording steps**:
1. Start openadapt-capture: `openadapt-capture record --name archive-folder`
2. Right-click on empty space in the folder
3. Select **New** > **Folder**
4. Type "Archive" and press Enter
5. Click on the first .docx file
6. Hold `Ctrl` and click each additional .docx file to select all
7. Right-click on the selection
8. Click **Cut** (or press `Ctrl+X`)
9. Double-click the "Archive" folder to open it
10. Right-click in empty space
11. Click **Paste** (or press `Ctrl+V`)
12. Stop recording: `Ctrl+C` in terminal

**Expected result**: Archive folder contains all .docx files, original location has only non-.docx files.

## Exporting Recordings

After recording, export each capture:

```powershell
# List recordings
openadapt-capture list

# Export to a directory
openadapt-capture export fill-blanks --output ./captures/fill-blanks
openadapt-capture export create-chart --output ./captures/create-chart
openadapt-capture export archive-folder --output ./captures/archive-folder
```

## Transferring to Mac

Option A: **Cloud sync** (OneDrive, Dropbox, Google Drive)
```powershell
# Copy to OneDrive
cp -r ./captures/* ~/OneDrive/openadapt-captures/
```

Option B: **Direct transfer** (if on same network)
```powershell
# From Windows, SCP to Mac
scp -r ./captures/* user@mac-ip:~/oa/src/openadapt-capture/waa-demos/
```

Option C: **USB drive**

## Converting to Demo Format

Once captures are on the Mac, I'll convert them to the text demo format:

```bash
# From openadapt-ml directory
uv run python scripts/convert_capture_to_demo.py \
  --capture ~/oa/src/openadapt-capture/waa-demos/fill-blanks \
  --task 4 \
  --output openadapt_ml/experiments/waa_demo/demos.py
```

The conversion extracts:
- Screenshot descriptions from each frame
- Action types and parameters (CLICK, TYPE, KEY)
- UI state changes between steps

## Verification Checklist

Before transferring, verify each recording:

- [ ] Recording starts from the expected initial state
- [ ] All steps are captured (check frame count)
- [ ] No errors during recording
- [ ] Screenshots are clear and readable
- [ ] Actions are correctly logged (check action log)

## Troubleshooting

### Recording doesn't start
- Run as Administrator if permission issues
- Check that screen recording permissions are granted

### Missing keystrokes
- Some special keys may not be captured; use mouse alternatives
- Ctrl+Enter might need to be recorded as separate Ctrl down, Enter, Ctrl up

### Screenshots are black
- Disable hardware acceleration in the target application
- Try windowed mode instead of fullscreen

## Questions?

If you encounter issues, check:
- openadapt-capture GitHub issues
- WAA documentation for expected UI states
- The manual demos in `openadapt_ml/experiments/waa_demo/demos.py` for format reference
