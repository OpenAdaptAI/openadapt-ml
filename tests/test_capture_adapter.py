#!/usr/bin/env python
"""Test CaptureAdapter directly."""

from pathlib import Path
from openadapt_ml.segmentation.adapters.capture_adapter import CaptureAdapter

# Test on turn-off-nightshift
capture_path = Path("/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift")

print(f"Testing CaptureAdapter on: {capture_path}")
print("=" * 60)

adapter = CaptureAdapter(include_moves=False)

try:
    images, events = adapter.load_recording(capture_path)

    print(f"\n✓ SUCCESS!")
    print(f"  Loaded {len(images)} frames")
    print(f"  Found {len(events)} events")

    # Show first few events
    print(f"\nFirst 5 events:")
    for i, event in enumerate(events[:5]):
        print(f"  {i+1}. [{event['name']}] @ {event['timestamp']:.2f}s")
        if 'mouse_x' in event:
            print(f"     Mouse: ({event['mouse_x']}, {event['mouse_y']})")
        if 'text' in event:
            print(f"     Text: {event['text']}")

    # Show last few events
    print(f"\nLast 5 events:")
    for i, event in enumerate(events[-5:], len(events)-4):
        print(f"  {i}. [{event['name']}] @ {event['timestamp']:.2f}s")
        if 'mouse_x' in event:
            print(f"     Mouse: ({event['mouse_x']}, {event['mouse_y']})")
        if 'text' in event:
            print(f"     Text: {event['text']}")

    # Image dimensions
    if images:
        w, h = images[0].size
        print(f"\nImage dimensions: {w}x{h}")

except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
