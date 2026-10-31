#!/bin/bash
set -e
git add -A
git status --short | head -30
git commit -m "Update FleetPM: dashboard and deck fixes" || echo "nothing new to commit"
git push -u origin main
echo "---- done ----"
git log --oneline | head -3
