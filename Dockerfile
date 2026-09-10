# Per kickoff_prompt.md's Deployment section. Deploys to a NEW, separate
# HuggingFace Space — never the existing production FinBuddy Spaces.
#
# Uses requirements-deploy.txt, not requirements.txt: docling/mlflow/ragas
# are only used by one-time/offline scripts (ingestion.build_corpus,
# mlops.intent_router.train, eval.ragas_eval) that never run inside this
# container — the pre-built chroma_data/ and trained router/risk-trend
# artifacts are bundled into the deploy snapshot instead (see
# scripts/prepare_hf_space_deploy.sh). See requirements-deploy.txt's own
# comment for the exact import-tracing that justifies excluding them.
FROM python:3.11-slim

WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
