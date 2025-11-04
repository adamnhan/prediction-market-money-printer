#!/usr/bin/env bash
# Script to run tests with coverage and open HTML report (macOS / Linux)
set -e
# script directory and repo root
SCRIPT_DIR=$(dirname "$0")
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

if [ "$1" = "install" ]; then
  echo "Installing dependencies..."
  pip install -r "$REPO_ROOT/requirements.txt"
fi

echo "Running tests with coverage..."
RCFILE="$REPO_ROOT/.coveragerc"
if [ -f "$RCFILE" ]; then
  coverage run --rcfile="$RCFILE" -m pytest
else
  echo "Couldn't find $RCFILE, running coverage without rcfile"
  coverage run -m pytest
fi

echo "Generating coverage.html..."
if [ -f "$RCFILE" ]; then
  coverage html --rcfile="$RCFILE"
else
  coverage html
fi

REPORT="$REPO_ROOT/htmlcov/index.html"
if [ -f "$REPORT" ]; then
  echo "Opening coverage report $REPORT"
  if which xdg-open > /dev/null; then
    xdg-open "$REPORT" || true
  elif which open > /dev/null; then
    open "$REPORT" || true
  else
    echo "Open $REPORT in your browser to view the report"
  fi
else
  echo "Coverage report not found at $REPORT"
fi
