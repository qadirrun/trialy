
import re
import math
import unicodedata

import pandas as pd

from collections import Counter, defaultdict
from rapidfuzz import fuzz, process

# ==========================================
# NORMALIZE TEXT
# ==========================================

def normalize(text):

    text = unicodedata.normalize(
        "NFKC", str(text)
    ).lower()

    text = re.sub(r"[\W_]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()

# ==========================================
# BUILD MASTERLIST
# ==========================================

def build_masterlist(
    df,
    domain_col,
    group_col,
    entity_col,
    attribute_col,
    alias_col=None,
    uom_col=None,
    forward_fill=True
):

    profiles = defaultdict(dict)

    hierarchy = [None, None, None]

    hierarchy_columns = [
        domain_col,
        group_col,
        entity_col
    ]

    for _, row in df.iterrows():

        for level, column in enumerate(
            hierarchy_columns
        ):

            value = row[column]

            if pd.notna(value) and str(value).strip():

                value = str(value).strip()

                if value != hierarchy[level]:

                    hierarchy[level] = value

                    # Reset child hierarchy
                    for child in range(level + 1, 3):

                        hierarchy[child] = None

            elif not forward_fill:

                hierarchy[level] = None

        if any(value is None for value in hierarchy):

            continue

        if pd.isna(row[attribute_col]):

            continue

        attribute = str(
            row[attribute_col]
        ).strip()

        key = normalize(attribute)

        if not key:

            continue

        profile = tuple(hierarchy)

        if key not in profiles[profile]:

            profiles[profile][key] = {
                "attribute": attribute,
                "aliases": {key},
                "expected_uom": ""
            }

        information = profiles[profile][key]

        # Optional synonyms
        if alias_col and pd.notna(row[alias_col]):

            aliases = re.split(
                r"[;|\n]",
                str(row[alias_col])
            )

            for alias in aliases:

                cleaned = normalize(alias)

                if cleaned:

                    information["aliases"].add(
                        cleaned
                    )

        # Optional expected unit
        if uom_col and pd.notna(row[uom_col]):

            information["expected_uom"] = str(
                row[uom_col]
            ).strip()

    if not profiles:

        raise ValueError(
            "No valid classification profiles found."
        )

    # Calculate attribute rarity.

    frequency = Counter()

    for attributes in profiles.values():

        terms = {
            alias
            for info in attributes.values()
            for alias in info["aliases"]
        }

        frequency.update(terms)

    weights = {
        term: 1 + math.log(
            (len(profiles) + 1) / (count + 1)
        )
        for term, count in frequency.items()
    }

    return dict(profiles), weights

# ==========================================
# BUILD DOCUMENT PHRASE INDEX
# ==========================================

def build_phrase_index(segments, max_words):

    index = defaultdict(set)

    for segment in segments:

        words = normalize(segment).split()

        for size in range(
            1,
            min(max_words, len(words)) + 1
        ):

            for start in range(
                len(words) - size + 1
            ):

                phrase = " ".join(
                    words[start:start + size]
                )

                index[size].add(phrase)

    return {
        size: sorted(phrases)
        for size, phrases in index.items()
    }

# ==========================================
# MATCH ATTRIBUTES AGAINST ENTIRE PDF
# ==========================================

def match_document_attributes(
    segments,
    profiles,
    fuzzy_threshold=90,
    enable_fuzzy=True
):

    terms = {
        alias
        for attributes in profiles.values()
        for info in attributes.values()
        for alias in info["aliases"]
    }

    if not terms:

        return {}

    max_words = max(
        len(term.split())
        for term in terms
    )

    phrase_index = build_phrase_index(
        segments,
        max_words + 1
    )

    all_phrases = {
        phrase
        for phrases in phrase_index.values()
        for phrase in phrases
    }

    matches = {}

    for term in sorted(terms):

        if term in all_phrases:

            matches[term] = {
                "text": term,
                "similarity": 100.0,
                "method": "Exact"
            }

            continue

        if not enable_fuzzy or len(term) < 7:

            continue

        count = len(term.split())

        candidates = []

        for size in range(
            max(1, count - 1),
            count + 2
        ):

            candidates.extend(
                phrase_index.get(size, [])
            )

        if not candidates:

            continue

        best = process.extractOne(
            term,
            candidates,
            scorer=fuzz.ratio,
            score_cutoff=fuzzy_threshold
        )

        if best:

            matches[term] = {
                "text": best[0],
                "similarity": float(best[1]),
                "method": "Fuzzy"
            }

    return matches

# ==========================================
# CLASSIFY WHOLE PDF
# ==========================================

def classify_document(
    segments,
    profiles,
    weights,
    fuzzy_threshold=90,
    enable_fuzzy=True,
    minimum_score=45,
    minimum_matches=3,
    minimum_margin=5
):

    evidence = match_document_attributes(
        segments,
        profiles,
        fuzzy_threshold,
        enable_fuzzy
    )

    candidates = []

    for profile, attributes in profiles.items():

        matched = []
        missing = []

        total_weight = 0.0
        matched_weight = 0.0

        used_phrases = set()

        ordered = sorted(
            attributes.values(),
            key=lambda info: (
                min(
                    weights.get(alias, 1)
                    for alias in info["aliases"]
                ),
                len(normalize(info["attribute"]))
            ),
            reverse=True
        )

        for info in ordered:

            attribute = info["attribute"]

            weight = min(
                weights.get(alias, 1.0)
                for alias in info["aliases"]
            )

            total_weight += weight

            possible = [
                evidence[alias]
                for alias in info["aliases"]
                if alias in evidence
                and evidence[alias]["text"]
                not in used_phrases
            ]

            if not possible:

                missing.append(attribute)

                continue

            best = max(
                possible,
                key=lambda x: x["similarity"]
            )

            matched_weight += (
                weight * best["similarity"] / 100
            )

            used_phrases.add(best["text"])

            matched.append({
                "Attribute": attribute,
                "OCR Text": best["text"],
                "Similarity": best["similarity"],
                "Method": best["method"]
            })

        if total_weight == 0:

            continue

        coverage = matched_weight / total_weight

        support = 1 - math.exp(
            -matched_weight / 5
        )

        score = 100 * (
            0.70 * coverage +
            0.30 * support
        )

        candidates.append({
            "profile": profile,
            "Data Domain": profile[0],
            "Data Group": profile[1],
            "Entity": profile[2],
            "Score": round(score, 2),
            "Coverage": round(coverage * 100, 2),
            "Matched": len(matched),
            "Total Attributes": len(attributes),
            "matched_attributes": matched,
            "missing_attributes": missing
        })

    candidates.sort(
        key=lambda x: (
            x["Score"],
            x["Matched"]
        ),
        reverse=True
    )

    if not candidates or candidates[0]["Matched"] == 0:

        return {
            "best": None,
            "status": "Review Required",
            "candidates": [],
            "reasons": ["No matching attributes"],
            "margin": 0
        }

    best = candidates[0]

    second_score = (
        candidates[1]["Score"]
        if len(candidates) > 1
        else 0
    )

    margin = best["Score"] - second_score

    reasons = []

    if best["Score"] < minimum_score:

        reasons.append("Low matching score")

    if best["Matched"] < minimum_matches:

        reasons.append("Insufficient attributes")

    if len(candidates) > 1 and margin < minimum_margin:

        reasons.append("Similar entity candidates")

    return {
        "best": best,
        "status": (
            "Review Required"
            if reasons
            else "Classified"
        ),
        "candidates": candidates[:10],
        "reasons": reasons,
        "margin": round(margin, 2)
    }
