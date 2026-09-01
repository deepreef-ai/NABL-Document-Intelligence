"""Offline master-schema stage (Step 4) — takes Step 3's schema_discovery
output (app/schema_discovery) and turns the raw, noisy per-domain discovered
keys into a usable schema: keys grouped by domain, common keys shared across
domains, domain-specific keys, and duplicate/synonym CANDIDATE clusters
(never auto-merged — see clustering.py). Writes master_schema.json (the
schema Step 5's labeling stage reads from) and key_mapping.json (the
canonical-key/alias/status record this stage was asked to produce).

Deliberately independent of the live FastAPI app, same as
dataset_normalization/ and schema_discovery/ — a standalone batch tool over
schema_discovery's output, run via scripts/build_master_schema.py. Makes no
LLM calls at all: everything here is deterministic clustering/classification
over keys the LLM already discovered in Step 3.
"""
