"""Print a markdown table from a jsonl file.

Usage: python tools/jsonl_to_md.py <file.jsonl> <field1> <field2> ...
"""

import json
import sys


def main() -> int:
    path, fields = sys.argv[1], sys.argv[2:]
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    print("| " + " | ".join(fields) + " |")
    print("|" + "---|" * len(fields))
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field, "")
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
