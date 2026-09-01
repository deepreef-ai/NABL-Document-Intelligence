"""Groups a domain's discovered keys into duplicate/synonym CANDIDATE
clusters (Step 4, requirement 4). Deliberately conservative: two keys only
land in the same cluster when they're identical, share a known
abbreviation/expansion (synonyms.ABBREVIATION_EXPANSIONS), or are lexically
very close (synonyms.are_likely_synonyms) — two fields that are only
semantically related stay separate rather than risk a wrong automatic merge.

A cluster with more than one member is exactly the kind of ambiguous merge
the task says not to make automatically, so it is always marked "review"; a
cluster of one is just that key on its own with nothing to decide, so it's
"approved".
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from app.master_schema.synonyms import are_likely_synonyms, normalize_key, tokens


@dataclass
class KeyCluster:
    canonical_key: str
    aliases: list[str]  # original discovered key strings, sorted
    status: str  # "approved" | "review"
    total_frequency: int


class _UnionFind:
    def __init__(self, items: list[str]):
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a


def _choose_canonical(members: list[str], key_frequency: dict[str, int]) -> str:
    """Prefer a known full expansion over an abbreviation, then the most
    frequent member, then the more descriptive (longer) spelling, then
    alphabetical — purely for a stable, readable label; it does not change
    which keys are grouped together."""
    expanded = {normalize_key(m) for m in members if normalize_key(m) != m}
    if expanded:
        def expansion_weight(exp: str) -> int:
            return sum(key_frequency[m] for m in members if normalize_key(m) == exp)
        return sorted(expanded, key=lambda exp: (-expansion_weight(exp), exp))[0]
    return sorted(members, key=lambda m: (-key_frequency[m], -len(m), m))[0]


def _dynamic_generic_tokens(keys: list[str]) -> frozenset[str]:
    """A token appearing in an unusually large share of THIS domain's own
    keys is boilerplate for that domain even if it isn't on the fixed
    synonyms._GENERIC_TOKENS list (e.g. "urine" across a medical panel's
    many "high_urine_*"/"low_urine_*" flags) — extra, domain-specific
    protection against the same hub-chaining failure mode on tokens this
    module's author didn't anticipate."""
    frequency = Counter()
    for key in keys:
        frequency.update(tokens(normalize_key(key)))
    threshold = max(3, round(0.03 * len(keys)))
    return frozenset(token for token, count in frequency.items() if count >= threshold)


def cluster_domain_keys(key_frequency: dict[str, int]) -> list[KeyCluster]:
    keys = list(key_frequency)
    if not keys:
        return []

    uf = _UnionFind(keys)

    by_normalized: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_normalized[normalize_key(key)].append(key)
    for group in by_normalized.values():
        for other in group[1:]:
            uf.union(group[0], other)

    dynamic_generic = _dynamic_generic_tokens(keys)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            if uf.find(a) == uf.find(b):
                continue
            if are_likely_synonyms(normalize_key(a), normalize_key(b), dynamic_generic):
                uf.union(a, b)

    groups: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        groups[uf.find(key)].append(key)

    clusters = []
    for members in groups.values():
        members_sorted = sorted(members)
        clusters.append(KeyCluster(
            canonical_key=_choose_canonical(members_sorted, key_frequency),
            aliases=members_sorted,
            status="approved" if len(members_sorted) == 1 else "review",
            total_frequency=sum(key_frequency[m] for m in members_sorted),
        ))
    return sorted(clusters, key=lambda c: (-c.total_frequency, c.canonical_key))
