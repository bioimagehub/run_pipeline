#!/usr/bin/env python3
"""
Generic CLI Argument Extractor
------------------------------

This script analyzes a given command or Python script to extract its command-line
arguments in a structured way.

Features:
- Detects argparse, click, typer in Python source files (via AST).
- Executes binaries or .exe tools safely with --help to extract args.
- Logs detailed info on INFO level, only errors on ERROR level.
- Designed for clean modularity and easy extension.

Usage:
    python cli_arg_extractor.py --script-path ./example.py
"""

import argparse
import ast
import logging
import subprocess
import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------

def configure_logging(level: str = "WARNING") -> None:
    """Configure logging for the script."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.WARNING
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

# ------------------------------------------------------------------------------
# AST Parsing for Python Scripts
# ------------------------------------------------------------------------------

class PythonCLIExtractor(ast.NodeVisitor):
    """Extract CLI args from Python files that use argparse, click, or typer."""
    def __init__(self):
        self.arguments = []

    def visit_Call(self, node: ast.Call):
        # argparse pattern
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    self.arguments.append(arg.value)
            for kw in node.keywords:
                if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                    self.arguments.append(f"--{kw.value.value}")

        # click / typer pattern
        if isinstance(node.func, ast.Name) and node.func.id in ("option", "argument"):
            for kw in node.keywords:
                if kw.arg == "param_decls" and isinstance(kw.value, ast.Tuple):
                    for el in kw.value.elts:
                        if isinstance(el, ast.Constant) and isinstance(el.value, str):
                            self.arguments.append(el.value)
        self.generic_visit(node)


def extract_python_args(path: Path) -> List[str]:
    """Extract CLI arguments from a Python file via static AST analysis."""
    try:
        logging.info(f"Parsing Python script: {path}")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        extractor = PythonCLIExtractor()
        extractor.visit(tree)
        args = sorted(set(extractor.arguments))
        logging.info(f"Found {len(args)} possible arguments in {path.name}")
        return args
    except Exception as e:
        logging.error(f"Failed to parse Python file {path}: {e}")
        return []

# ------------------------------------------------------------------------------
# Dynamic Help Extraction for Any CLI Command or .exe
# ------------------------------------------------------------------------------

def run_help_command(command: str) -> Optional[str]:
    """Attempt to run a CLI command with common help flags and return output."""
    help_flags = ["--help", "-h", "help"]
    for flag in help_flags:
        try:
            logging.info(f"Running '{command} {flag}' to extract help text")
            result = subprocess.run(
                [command, flag],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout:
                logging.info(f"Successfully retrieved help output using '{flag}'")
                return result.stdout
            if result.stderr:
                logging.info(f"Retrieved help output from stderr using '{flag}'")
                return result.stderr
        except FileNotFoundError:
            logging.error(f"Command not found: {command}")
            return None
        except subprocess.TimeoutExpired:
            logging.error(f"Command timed out: {command} {flag}")
        except Exception as e:
            logging.error(f"Error running {command} {flag}: {e}")
    return None


def parse_help_output(help_text: str) -> List[str]:
    """Parse help text and extract argument flags."""
    flags = re.findall(r"(?:--?\w[\w-]*)", help_text)
    flags = sorted(set(flags))
    logging.info(f"Extracted {len(flags)} argument flags from help text")
    return flags


def extract_cli_help(command: str) -> List[str]:
    """Run help extraction and parse flags for any CLI command."""
    output = run_help_command(command)
    if not output:
        logging.error(f"No help output found for {command}")
        return []
    return parse_help_output(output)

# ------------------------------------------------------------------------------
# Universal Extractor
# ------------------------------------------------------------------------------

def universal_cli_extractor(path_or_command: str) -> Dict[str, List[str]]:
    """Dispatch extraction based on file type or executable."""
    path = Path(path_or_command)
    if path.suffix == ".py" and path.exists():
        args = extract_python_args(path)
        return {"type": "python", "arguments": args}
    else:
        args = extract_cli_help(path_or_command)
        return {"type": "command", "arguments": args}

# ------------------------------------------------------------------------------
# Main Entrypoint
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this tool itself."""
    parser = argparse.ArgumentParser(description="Generic CLI argument extractor.")
    parser.add_argument(
        "--script-path",
        required=True,
        help="Path to a Python script or command to analyze."
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set log level (default: WARNING)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    result = universal_cli_extractor(args.script_path)
    if result["arguments"]:
        print("\nDetected arguments:")
        for arg in result["arguments"]:
            print(f"  {arg}")
    else:
        logging.warning("No arguments detected.")


if __name__ == "__main__":
    main()

"""
[project]
name = "cli-arg-extractor"
version = "0.1.0"
description = "Generic command-line argument extractor for Python scripts and executables."
readme = "README.md"
requires-python = ">=3.9"
license = "MIT"
authors = [
    { name = "Your Name", email = "your.email@example.com" }
]

[project.scripts]
cli-arg-extractor = "cli_arg_extractor:main"

[tool.uv]
# UV-compatible project config
# You can run: uv run cli-arg-extractor --script-path yourtool.py

[tool.ruff]
line-length = 100



"""