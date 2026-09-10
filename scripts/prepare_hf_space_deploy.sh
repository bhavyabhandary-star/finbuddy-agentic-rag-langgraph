#!/usr/bin/env bash
set -euo pipefail

# Prepares a single-commit deployment snapshot for a HuggingFace Space, and
# prints the exact `git push` command to run — it does NOT push itself,
# deliberately: your HF write-credentials stay in your own hands, never
# passed to this script or to Claude.
#
# Mirrors finbuddy-project's own proven deploy pattern
# (.github/workflows/deploy.yml)'s documented reasoning: a fresh
# single-commit snapshot, not `git subtree push` (fails against a
# freshly-created Space's auto-generated initial commit — non-fast-forward)
# and not this repo's full history (binary artifacts committed as plain
# blobs get rejected by HF's server without LFS).
#
# Copies the current WORKING TREE (not `git archive HEAD`), so it picks up
# chroma_data/, the risk-trend artifact, and the Setu profile — all
# deliberately gitignored in the dev repo (regenerable or third-party) but
# genuinely needed for the deployed service to work, since they already
# exist on disk from this session's own verified runs.
#
# Usage: ./scripts/prepare_hf_space_deploy.sh <space-name>
#   e.g. ./scripts/prepare_hf_space_deploy.sh finbuddy-langgraph-agent

SPACE_NAME="${1:?Usage: $0 <space-name>}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="/tmp/deploy-${SPACE_NAME}"

echo "Preparing deployment snapshot in ${DEPLOY_DIR} ..."
rm -rf "${DEPLOY_DIR}"
mkdir -p "${DEPLOY_DIR}"

cd "${PROJECT_ROOT}"
tar \
  --exclude='./.venv' \
  --exclude='./.venv-deploy-test' \
  --exclude='./.git' \
  --exclude='**/__pycache__' \
  --exclude='./.pytest_cache' \
  --exclude='./traces' \
  --exclude='./mlruns' \
  --exclude='./mlflow.db' \
  --exclude='./data/raw_pdfs' \
  --exclude='./.env' \
  -cf - . | (cd "${DEPLOY_DIR}" && tar -xf -)

echo "Bundled artifacts present in the snapshot:"
for f in chroma_data mlops/risk_trend_monitor/artifacts/risk_trend_logreg.joblib \
         mlops/intent_router/registry_store data/setu_real_profiles.jsonl; do
  if [ -e "${DEPLOY_DIR}/${f}" ]; then
    echo "  OK: ${f}"
  else
    echo "  MISSING: ${f}  <-- that feature won't work in the deployed instance"
  fi
done

cd "${DEPLOY_DIR}"

# CRITICAL: the dev repo's own .gitignore was copied along with everything
# else (it's a real file, not excluded above) — and it deliberately ignores
# chroma_data/, the risk-trend artifact, the registry, and the Setu profile
# in the DEV repo. Left in place, `git add -A` below would silently respect
# those same rules here too and skip committing exactly the artifacts this
# snapshot exists to bundle. Found by verifying the committed snapshot
# actually contained them (git cat-file), not assumed. A deploy snapshot has
# no business ignoring anything already on disk — remove it before adding.
rm -f .gitignore

git init -q -b main
git config user.email "deploy@local"
git config user.name "FinBuddy LangGraph Deploy"
git lfs install --local >/dev/null
git lfs track "*.sqlite3" "*.bin" "*.joblib" >/dev/null
git add -A
git commit -q -m "Deploy ${SPACE_NAME} snapshot ($(date -u +%Y-%m-%dT%H:%M:%SZ))"

echo
echo "=========================================================================="
echo "Snapshot ready at: ${DEPLOY_DIR}"
echo "=========================================================================="
echo
echo "Next steps (yours to run — this script never touches your HF credentials):"
echo
echo "1. Create the Space (if you haven't already):"
echo "     https://huggingface.co/new-space"
echo "     Name: ${SPACE_NAME}   SDK: Docker   Visibility: your choice"
echo
echo "2. Push this snapshot (git will prompt for your HF username, then a"
echo "   write-scoped token as the password — https://huggingface.co/settings/tokens):"
echo
echo "     cd ${DEPLOY_DIR}"
echo "     git push https://huggingface.co/spaces/<your-username>/${SPACE_NAME} HEAD:main --force"
echo
echo "3. In the Space's Settings > Variables and secrets, add:"
echo "     LLM_PROVIDER = huggingface   (or claude/ollama — see .env.example)"
echo "     HF_TOKEN = <your HF token, if using huggingface>"
echo "     ANTHROPIC_API_KEY = <your key, if using claude>"
echo
echo "   FINBUDDY_SCORING_API_BASE_URL, RISK_TREND_ARTIFACT_PATH, and"
echo "   CHROMA_PERSIST_DIR already have working defaults baked into the code"
echo "   (see .env.example) — no need to set them unless overriding."
