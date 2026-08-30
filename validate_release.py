from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IGNORED = {".git", ".venv", "__pycache__", ".ipynb_checkpoints"}
SECRET_PATTERNS = [
    re.compile(r"BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*['\"]?[^\s'\"]+", re.I),
]


def files() -> list[Path]:
    return [p for p in ROOT.rglob("*") if p.is_file() and not any(part in IGNORED for part in p.parts)]


def main() -> int:
    problems: list[str] = []
    all_files = files()

    if any(p.name == ".git" for p in ROOT.rglob("*")):
        problems.append("nested .git directory found")

    for path in all_files:
        if path.stat().st_size > 25 * 1024 * 1024:
            problems.append(f"oversized file: {path.relative_to(ROOT)}")
        if path.suffix.lower() in {".pt", ".onnx", ".tflite", ".engine", ".mp4", ".avi", ".mov", ".mkv", ".zip"}:
            problems.append(f"release-risk artifact: {path.relative_to(ROOT)}")
        if path.suffix.lower() in {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".ipynb"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                problems.append(f"unreadable text file {path.relative_to(ROOT)}: {exc}")
                continue
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    problems.append(f"possible secret pattern in {path.relative_to(ROOT)}")
                    break
        if path.suffix.lower() == ".py" and path.name != Path(__file__).name:
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError) as exc:
                problems.append(f"Python syntax issue in {path.relative_to(ROOT)}: {exc}")

    notebook = ROOT / "fireDetector" / "kaggle" / "fire_detection_train.ipynb"
    try:
        notebook_data = json.loads(notebook.read_text(encoding="utf-8"))
        if not isinstance(notebook_data.get("cells"), list):
            problems.append("training notebook has no valid cells list")
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"training notebook is not valid JSON: {exc}")

    required = [
        ROOT / "README.md",
        ROOT / ".gitignore",
        ROOT / "PUBLICATION_CHECKLIST.md",
        notebook,
        ROOT / "fireDetector" / "laptop" / "requirements.txt",
    ]
    for path in required:
        if not path.is_file():
            problems.append(f"missing required file: {path.relative_to(ROOT)}")

    if problems:
        print("FAIL")
        print("\n".join(f"- {problem}" for problem in problems))
        return 1

    print(f"PASS: {len(all_files)} release files checked")
    print("PASS: no nested Git metadata, oversized files, release-risk artifacts, or detected secret patterns")
    print("PASS: Python source parsed successfully")
    print("PASS: training notebook is valid JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
