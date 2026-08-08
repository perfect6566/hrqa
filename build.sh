#!/usr/bin/env bash
# Render build script: install dependencies and build the RAG index so the
# deployed service can boot without paying the indexing cost at startup.
#
# Render runs this from the repository root (rootDir in render.yaml).
set -euo pipefail

echo "==> Upgrade pip"
python -m pip install --upgrade pip

echo "==> Install dependencies"
pip install -r requirements.txt

echo "==> Pre-build RAG index"
# Build the FAISS vector store into ./data/vector_store so the app boots
# with a warm index on the first cold start. Skip if the policies directory
# is missing (e.g. during a smoke deploy) so we never block the build.
if [ -d "policies" ] && [ -n "$(ls -A policies 2>/dev/null)" ]; then
  python -c "from src.rag.rag_pipeline import RAGPipeline; RAGPipeline.build_index()"
else
  echo "==> WARN: policies/ directory missing or empty; skipping RAG pre-build"
fi

echo "==> Build complete"
