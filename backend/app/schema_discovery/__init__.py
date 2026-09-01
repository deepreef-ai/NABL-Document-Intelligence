"""Offline schema-discovery stage (Step 3) — takes Step 2's normalized
documents (app/dataset_normalization) and sends a small, domain-diverse
SAMPLE of them (never the whole dataset) to the project's existing LLM
chain, asking only "what field/parameter names are present in this report?"
— never values, never ground truth. Output is a per-sampled-document
{document_id, candidate_domain, keys} plus a domain -> keys aggregate,
which is the actual deliverable this stage exists to produce.

Deliberately independent of the live FastAPI app, same as
dataset_normalization/ — a standalone batch tool over that stage's output,
run via scripts/discover_schema.py. Reuses app/llm/factory.py's LlmChain
exactly as documents/classifier.py and documents/extractor.py already do;
no new LLM provider code.
"""
