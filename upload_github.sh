#!/bin/bash
set -e

for p in '*.zip' '*.bak_*' 'cleanup_backup_*' 'write_v*.py' 'write_ext.py' 'mdtodocs.py' 'nohup.out' '*.pptx' '*.db' '*.npz' '*.joblib' 'artifacts/' 'data/' '__pycache__/' '.pytest_cache/' '*.egg-info/' '.DS_Store' 'upload_github.sh'; do
  grep -qxF "$p" .gitignore 2>/dev/null || echo "$p" >> .gitignore
done

[ -d .git ] || git init
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/abhishekk-km/FLEET-PM.git

git rm -r --cached -q . 2>/dev/null || true
git add -A
echo "---- files staged ----"
git status --short | head -60
echo "---- largest staged files (KB) ----"
git ls-files -z | xargs -0 du -k 2>/dev/null | sort -rn | head -5

git commit -m "FleetPM: predictive maintenance on NASA C-MAPSS (anomaly, RUL, triage, conformal bounds, cost replay, dashboard, deck builder)" || echo "nothing new to commit"

git push -u origin main || { git pull origin main --allow-unrelated-histories --no-edit && git push -u origin main; }

echo "---- done ----"
git log --oneline | head -3
