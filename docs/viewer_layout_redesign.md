# Viewer Layout Redesign

## Problem Statement

The current viewer layout has several UX issues:
1. **Controls below the fold** - Playback controls, timeline, and dropdowns require scrolling to access
2. **Transcript naming is ambiguous** - "Transcript" could mean many things; it's specifically audio transcription
3. **No visual context** - Missing auto-generated description of what's happening in the video

## Current Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ [Training] [Viewer]          EXAMPLE ▼  CHECKPOINT ▼   ID: xxx │  <- Unified Header
├─────────────────────────────────────────────────────────────────┤
│ Action Comparison                                               │
│ ┌──────────────┐  ┌──────────────┐                              │
│ │ Human Action │  │ Model Pred   │  [Match indicator]           │
│ └──────────────┘  └──────────────┘                              │
├─────────────────────────────────────────────────────────┬───────┤
│                                                         │Events │
│        [Screenshot]                                     │(21)   │
│                                                         ├───────┤
│ ⏮ ◀ ▶ ⏭  0:00 / 1:00  [Overlay toggle]                │Details│
│ [─────────────────── Timeline ──────────────────]       ├───────┤
│ [Audio controls]                                        │Trans- │
│                                                         │cript  │
└─────────────────────────────────────────────────────────┴───────┘
```

## Proposed Layout

Move controls to header area, add Video Description, rename Transcript:

```
┌─────────────────────────────────────────────────────────────────┐
│ [Training] [Viewer]          EXAMPLE ▼  CHECKPOINT ▼   ID: xxx │  <- Unified Header
├─────────────────────────────────────────────────────────────────┤
│ ⏮ ◀ ▶ ⏭  Step 3/21  0:05.2 / 1:00.3   [Overlay ●]             │  <- Playback Bar
│ [══════════════════════ Timeline ═══════════════════════════]   │
├─────────────────────────────────────────────────────────────────┤
│ Action Comparison                                               │
│ ┌──────────────┐  ┌──────────────┐                              │
│ │ Human Action │  │ Model Pred   │  [Match: ✓ 12px]             │
│ └──────────────┘  └──────────────┘                              │
├─────────────────────────────────────────────────────────┬───────┤
│                                                         │Video  │
│        [Screenshot]                                     │Desc.  │
│                                                         ├───────┤
│                                                         │Audio  │
│                                                         │Trans. │
│                                                         ├───────┤
│                                                         │Events │
│                                                         ├───────┤
│                                                         │Details│
└─────────────────────────────────────────────────────────┴───────┘
```

## Key Changes

### 1. Playback Controls Above Fold
Move playback controls (⏮ ◀ ▶ ⏭) and timeline to a dedicated bar below the header, always visible.

**Benefits:**
- Controls always accessible without scrolling
- Cleaner separation of navigation vs content
- Matches video player UX conventions (YouTube, etc.)

### 2. Rename "Transcript" → "Audio Transcript"
More explicit about what this data represents.

### 3. Add "Video Description" Panel
Auto-generated description of what's happening visually in the current frame/action.

**Implementation options:**
1. **Local Qwen VLM** - Use existing `Qwen3VLAdapter` or `Qwen25VLAdapter`
2. **API VLM** - Use `AnthropicVLMAdapter` or `OpenAIVLMAdapter` for higher quality
3. **Pre-computed** - Generate during capture/training, store in JSON

**Prompt template:**
```
Describe what the user is doing in this screenshot in one sentence.
Focus on: which application, what UI element is being interacted with, what action is being performed.
```

**Example outputs:**
- "User clicks the 'New Document' button in Microsoft Word's ribbon toolbar"
- "User types 'meeting notes' into the filename field of a Save dialog"
- "User scrolls down in a web browser showing a product listing page"

### 4. Sidebar Reordering
New order (top to bottom):
1. **Video Description** (new) - What's happening visually
2. **Audio Transcript** - What was said
3. **Events** - Raw event list
4. **Event Details** - Selected event details

**Rationale:**
- Description + Transcript together tell the "story" of what happened
- Events/Details are more technical, used for debugging

## Implementation Plan

### Phase 1: Layout Restructure
1. Move playback controls to dedicated bar below header
2. Update CSS grid/flexbox for new layout
3. Rename "Transcript" → "Audio Transcript"

### Phase 2: Video Description Feature
1. Add `video_description` field to prediction JSON schema
2. Create VLM description generator utility
3. Add Video Description panel to sidebar
4. Option A: Generate during capture (real-time)
5. Option B: Generate during training/viewer generation (batch)

### Files to Modify

- `openadapt_ml/training/trainer.py` - Viewer HTML generation
- `openadapt_ml/scripts/compare.py` - Comparison viewer generation
- `openadapt_ml/schemas/sessions.py` - Add video_description field (if stored)
- `openadapt_ml/vlm/` - Add description generation utility

## Open Questions

1. **When to generate descriptions?**
   - Real-time during viewing (slow, requires GPU/API)
   - Pre-computed during capture (adds capture time)
   - Pre-computed during viewer generation (batch, one-time cost)

2. **Which VLM adapter to use by default?**
   - Local Qwen (free, requires GPU)
   - Anthropic Claude (fast, costs money)
   - Configurable via settings

3. **Should descriptions be editable?**
   - Allow users to correct/improve descriptions
   - Store as annotations alongside captures
