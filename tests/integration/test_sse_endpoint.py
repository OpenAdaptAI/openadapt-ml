#!/usr/bin/env python3
"""Test script for SSE benchmark endpoint.

Usage:
    python test_sse_endpoint.py [--interval 5]
"""

import argparse
import requests
import json
import time
from datetime import datetime


def test_sse_endpoint(base_url: str = "http://localhost:8765", interval: int = 5):
    """Test the SSE endpoint by connecting and printing events."""
    url = f"{base_url}/api/benchmark-sse?interval={interval}"

    print(f"Connecting to SSE endpoint: {url}")
    print(f"Poll interval: {interval}s")
    print("-" * 80)

    try:
        response = requests.get(url, stream=True, timeout=None)
        response.raise_for_status()

        print(f"Connected! Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        print("-" * 80)

        # Read events
        event_type = None
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')

                if line.startswith('event:'):
                    event_type = line[6:].strip()
                elif line.startswith('data:'):
                    data = line[5:].strip()
                    timestamp = datetime.now().strftime("%H:%M:%S")

                    # Parse JSON data
                    try:
                        data_obj = json.loads(data)
                        print(f"\n[{timestamp}] Event: {event_type}")
                        print(json.dumps(data_obj, indent=2))
                    except json.JSONDecodeError:
                        print(f"\n[{timestamp}] Event: {event_type}")
                        print(f"Data: {data}")

                    print("-" * 80)

    except KeyboardInterrupt:
        print("\n\nDisconnected by user")
    except requests.exceptions.RequestException as e:
        print(f"\nConnection error: {e}")
    except Exception as e:
        print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test SSE benchmark endpoint")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8765",
        help="Base URL of the server (default: http://localhost:8765)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Poll interval in seconds (default: 5)"
    )

    args = parser.parse_args()

    print("SSE Benchmark Endpoint Test")
    print("=" * 80)
    print(f"Server: {args.base_url}")
    print(f"Interval: {args.interval}s")
    print("\nPress Ctrl+C to stop\n")

    test_sse_endpoint(args.base_url, args.interval)


if __name__ == "__main__":
    main()
