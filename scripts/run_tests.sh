#!/usr/bin/env bash
# Script to run tests with coverage and open HTML report (macOS / Linux)
set -e
if [ "$1" = "install" ]; then
  echo "Installing dependencies..."
  pip install -r ../requirements.txt
fi

echo "Running tests with coverage..."
coverage run -m pytest

echo "Generating coverage.html..."
coverage html

REPORT="../htmlcov/index.html"
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
