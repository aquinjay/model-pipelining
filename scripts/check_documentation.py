"""Check repository Markdown links/anchors and Python/Bash example syntax.

Pass --walkthrough to also execute the forecasting guide's trusted Python blocks
in temporary storage. Other examples are syntax-checked, never executed.
"""

import argparse
import ast
from collections import Counter
from contextlib import chdir
import json
from pathlib import Path
import re
import runpy
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def blocks_and_prose(text):
    """Return fenced blocks with line numbers and prose with code blanked out."""
    blocks, prose, opened, code = [], [], None, []
    for number, line in enumerate(text.splitlines(), 1):
        if opened:
            if re.fullmatch(r"\s{0,3}"+re.escape(opened[0][0])+"{"+str(len(opened[0]))+r",}\s*", line):
                blocks.append((opened[1], opened[2], "\n".join(code)))
                opened, code = None, []
            else:
                code.append(line)
            prose.append("")
        else:
            match = re.fullmatch(r"\s{0,3}(`{3,}|~{3,})([\w-]*)\s*", line)
            if match:
                opened = (match[1], number, match[2].lower())
                prose.append("")
            else:
                prose.append(line)
    if opened:
        raise ValueError(f"Unclosed fenced block at line {opened[1]}")
    return blocks, "\n".join(prose)


def anchors(text):
    """GitHub-style anchors for the ATX headings used in project documentation."""
    _, prose = blocks_and_prose(text)
    result, counts = set(), Counter()
    for line in prose.splitlines():
        match = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            heading = re.sub(r"<[^>]*>", "", match[1]).lower()
            base = re.sub(r"[^\w -]", "", heading).strip().replace(" ", "-")
            suffix = counts[base]
            value = base if not suffix else f"{base}-{suffix}"
            while value in result:
                suffix += 1
                value = f"{base}-{suffix}"
            counts[base] = suffix+1
            result.add(value)
    result.update(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', prose))
    return result


def check_document(path, root):
    """Return syntax counts or raise a descriptive local documentation error."""
    text = path.read_text(encoding="utf-8")
    blocks, prose = blocks_and_prose(text)
    counts = Counter()
    for number, language, code in blocks:
        try:
            if language == "python":
                ast.parse(code, filename=f"{path}:{number}")
                counts["python"] += 1
            elif language in ("bash", "sh"):
                result = subprocess.run(["bash", "--noprofile", "--norc", "-n"], input=code,
                                        text=True, capture_output=True, check=False)
                if result.returncode:
                    raise ValueError(result.stderr.strip())
                counts["bash"] += 1
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"{path}:{number}: invalid {language} example: {error}") from error
    # Bracketed Python indexing in inline code is not a reference link.
    prose = re.sub(r"(`+)([^\n]*?)\1(?!`)", lambda m: " "*len(m[0]), prose)
    definitions = list(re.finditer(r"^\s*\[([^]\n]+)\]:\s*(<[^>\n]+>|\S+)", prose, re.M))
    normalize = lambda label: " ".join(label.casefold().split())
    references = {normalize(match[1]): match[2] for match in definitions}
    links = [(match.start(), match[1]) for match in re.finditer(
        r"\[[^]\n]*\]\(\s*(<[^>\n]+>|[^()\s]+)(?:\s+[\"'][^\n]*?[\"'])?\s*\)", prose)]
    links.extend((match.start(), match[2]) for match in definitions)
    for match in re.finditer(r"\[([^]\n]+)\]\[([^]\n]*)\]", prose):
        label = normalize(match[2] or match[1])
        if label not in references:
            raise ValueError(f"{path}: undefined reference link {label}")
    for position, target in links:
        target = target.removeprefix("<").removesuffix(">")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            continue  # External links are intentionally not fetched by CI.
        destination = (path.parent/unquote(parsed.path)).resolve() if parsed.path else path.resolve()
        line = prose.count("\n", 0, position)+1
        if not destination.is_relative_to(root.resolve()) or not destination.exists():
            raise ValueError(f"{path}:{line}: missing/escaping local link {target}")
        if parsed.fragment and destination.suffix.lower() == ".md":
            if unquote(parsed.fragment) not in anchors(destination.read_text(encoding="utf-8")):
                raise ValueError(f"{path}:{line}: missing local anchor {target}")
        counts["links"] += 1
    return counts


def documentation_paths(root):
    """Include tracked and new Markdown, excluding ignored local run/data output."""
    result = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others",
                             "--exclude-standard", "-z", "--", "*.md"],
                            capture_output=True, check=True)
    return sorted({root/Path(name.decode()) for name in result.stdout.split(b"\0") if name})


def exercise_walkthrough(root):
    """Execute the guide's Python blocks verbatim, with a fresh example input."""
    blocks, _ = blocks_and_prose((root/"docs/forecasting.md").read_text(encoding="utf-8"))
    source = "\n\n".join(code for _, language, code in blocks if language == "python")
    generator = runpy.run_path(str(root/"scripts/generate_synthetic.py"))["generate"]
    with TemporaryDirectory(prefix="model-pipelining-docs-") as temporary, chdir(temporary):
        generator(Path("data/synthetic"))
        exec(compile(source, "docs/forecasting.md examples", "exec"), {"__name__": "__documentation_example__"})
        for name in ("tutorial-mean-001", "tutorial-local-level-001"):
            run = Path("runs")/name
            manifest = json.loads((run/"manifest.json").read_text())
            scores = json.loads((run/"scores.json").read_text())
            if manifest["status"] != "completed" or not scores["complete"] or scores["scored_count"] != 170:
                raise ValueError("Forecasting walkthrough did not complete its documented 170 cases")
            if not (run/"forecast.png").is_file():
                raise ValueError("Forecasting walkthrough chart is missing")
        comparison = json.loads(Path("runs/tutorial-comparison-001/comparison.json").read_text())
        if comparison["case_count"] != 170 or not Path("runs/tutorial-comparison-001/comparison.png").is_file():
            raise ValueError("Forecasting walkthrough comparison is incomplete")
    print("Forecasting guide executed: two 170-case runs and saved comparison; temporary output removed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--walkthrough", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    paths = [root/path for path in args.paths] if args.paths else documentation_paths(root)
    if not paths:
        parser.error("No documentation files found")
    counts = Counter()
    for path in paths:
        counts.update(check_document(path, root))
    print(f"{len(paths)} Markdown files: {counts['links']} local links/anchors valid; "
          f"{counts['python']} Python and {counts['bash']} Bash examples parse")
    if args.walkthrough:
        exercise_walkthrough(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
