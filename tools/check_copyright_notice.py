#!/usr/bin/env python3
# Copyright 2025 Irreducible Inc.
# Copyright 2026 The Binius Developers
"""Check and fix copyright headers in Rust files."""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

YEAR = datetime.now().year
COPYRIGHT_HOLDER = "The Binius Developers"
VALID_PATTERN = re.compile(rf"^// Copyright (\d{{4}}-)?{YEAR} {re.escape(COPYRIGHT_HOLDER)}$")


def get_rust_files(dirname):
    """Get all Rust files in crates directory using git ls-files."""
    try:
        result = subprocess.run(
            ["git", "ls-files", f"{dirname}/"],
            capture_output=True,
            text=True,
            check=True,
        )
        rust_files = [f for f in result.stdout.strip().split("\n") if f.endswith(".rs")]
        return sorted(rust_files)
    except subprocess.CalledProcessError:
        return sorted(str(p) for p in Path(dirname).rglob("*.rs"))


def check_file(filepath):
    """Get first line of file, or full lines if return_lines=True."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.readline().strip()
    except Exception:
        return None


def fix_file(filepath, old_header=None):
    """Fix copyright header in file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Determine new header
        header = f"// Copyright {YEAR} {COPYRIGHT_HOLDER}\n"
        if old_header and COPYRIGHT_HOLDER in old_header:
            # Preserve year range if exists
            if match := re.search(r"(\d{4})", old_header):
                if (year := match.group(1)) != str(YEAR):
                    header = f"// Copyright {year}-{YEAR} {COPYRIGHT_HOLDER}\n"

        # Update or insert header
        if lines and lines[0].startswith("// Copyright") and COPYRIGHT_HOLDER in lines[0]:
            lines[0] = header
        else:
            lines.insert(0, header)

        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return True
    except Exception as e:
        print(f"  Error: {filepath}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Check and fix copyright headers in Rust files"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix missing and wrong format copyright headers",
    )
    args = parser.parse_args()

    files = get_rust_files("crates")
    if not files:
        print("No Rust files found in crates/")
        return 1

    # Group files by their first line
    headers = defaultdict(list)
    for filepath in files:
        first_line = check_file(filepath)
        key = (
            first_line
            if first_line and first_line.startswith("// Copyright")
            else "[NO COPYRIGHT]"
        )
        headers[key].append(filepath)

    # Categorize headers
    categories = {"correct": {}, "other": {}, "wrong": {}, "missing": {}}

    for header, file_list in headers.items():
        if header == "[NO COPYRIGHT]":
            categories["missing"][header] = file_list
        elif VALID_PATTERN.match(header):
            categories["correct"][header] = file_list
        elif COPYRIGHT_HOLDER in header:
            categories["wrong"][header] = file_list
        else:
            categories["other"][header] = file_list

    # Print results
    print(f"Expected: // Copyright {YEAR} {COPYRIGHT_HOLDER}")
    print(f"      or: // Copyright YYYY-{YEAR} {COPYRIGHT_HOLDER}")
    print("=" * 70)

    sections = [
        ("✅ CORRECT COPYRIGHT", categories["correct"], False),
        (f"🔸 OTHER LICENSE (not {COPYRIGHT_HOLDER})", categories["other"], False),
        (f"⚠️  WRONG COPYRIGHT FORMAT (has {COPYRIGHT_HOLDER})", categories["wrong"], True),
        ("❌ NO COPYRIGHT", categories["missing"], True),
    ]

    for title, group, show_files in sections:
        if group:
            print(f"\n{title}\n{'-' * 70}")
            for header, file_list in sorted(group.items(), key=lambda x: -len(x[1])):
                print(f"{len(file_list):6}  {header}")
                if show_files:
                    for filepath in file_list:
                        print(f"        {filepath}")

    # Summary
    totals = {k: sum(len(f) for f in v.values()) for k, v in categories.items()}
    print("\n" + "=" * 70)
    print(f"✅ Correct:      {totals['correct']:4}")
    print(f"🔸 Other:        {totals['other']:4}")
    print(f"⚠️  Wrong format: {totals['wrong']:4}")
    print(f"❌ No copyright: {totals['missing']:4}")
    print("-" * 20)
    print(f"📊 TOTAL:        {len(files):4}")

    # Fix if requested
    to_fix = totals["wrong"] + totals["missing"]
    if args.fix and to_fix > 0:
        print(f"\n{'=' * 70}\n🔧 FIXING FILES...\n{'=' * 70}")

        fixed = 0
        for cat_name, needs_old in [("wrong", True), ("missing", False)]:
            for header, file_list in categories[cat_name].items():
                for filepath in file_list:
                    if fix_file(filepath, header if needs_old else None):
                        fixed += 1
                        action = "Fixed" if needs_old else "Added"
                        print(f"  ✅ {action}: {filepath}")

        print(f"\n{'-' * 70}\n🔧 Fixed: {fixed}/{to_fix} files")
        return 0 if fixed == to_fix else 1

    if to_fix > 0:
        if not args.fix:
            print(f"\n💡 Run with --fix to automatically fix {to_fix} files")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
