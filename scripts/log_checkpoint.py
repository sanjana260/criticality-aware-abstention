#!/usr/bin/env python3
"""Record a checkpoint: an empty commit tagged [checkpoint], then push.

Commit and push your real work first. This script does not include the
working tree; it only adds a marker commit that the progress log picks up.

Usage:
  python scripts/log_checkpoint.py "Title" "Optional notes"
"""
import subprocess
import sys


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('usage: python scripts/log_checkpoint.py "Title" ["Notes"]')
    title = sys.argv[1]
    notes = sys.argv[2] if len(sys.argv) > 2 else ""
    message = f"[checkpoint] {title}"
    if notes:
        message += f"\n\n{notes}"
    subprocess.run(["git", "commit", "--allow-empty", "-m", message], check=True)
    subprocess.run(["git", "push"], check=True)


if __name__ == "__main__":
    main()
