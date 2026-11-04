import hashlib
import io
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

from rapidfuzz import fuzz, process, utils
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Equipment Name Matcher",
    layout="centered"
)

st.title("Equipment Name Matcher")

st.write(
    "Find the closest benchmark equipment using "
    "equipment names and benchmark descriptions."
)


# =========================================================
# 2. MATCHING CONFIGURATION
# =========================================================

CANDIDATE_LIMIT = 500

NAME_WEIGHT = 0.75
CATALOGUE_WEIGHT = 0.25


# =========================================================
# 3. TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))

    normalized = utils.default_process(text)

    return normalized.strip() if normalized else ""


# =========================================================
# 4. LOAD BENCHMARK EXCEL AND BUILD TF-IDF INDEX
# =========================================================

@st.cache_resource(show_spinner="Loading benchmark and building search index...")

def load_benchmark(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        usecols=[0, 1, 2],
        dtype=str,
        engine="openpyxl",
        keep_default_na=False
    )

    # Assign consistent column names
    df.columns = [
        "code",
        "name",
        "description"
    ]

    # Clean Excel values
    for column in ["code", "name", "description"]:

        df[column] = df[column].str.strip()

    # Remove records without benchmark names
    df = df[df["name"] != ""].copy()

    # Normalize benchmark names
    df["normalized_name"] = df["name"].apply(
        normalize_text
    )

    df = df[df["normalized_name"] != ""].copy()

    # Remove only completely duplicated records
    df = df.drop_duplicates(
        subset=["code", "name", "description"],
        keep="first"
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid benchmark records found.")

    # Give benchmark names additional importance
    # while retaining their descriptions for retrieval
    df["search_text"] = (
        df["name"] + " " +
        df["name"] + " " +
        df["description"]
    )

    # Build TF-IDF index
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=300000,
        sublinear_tf=True,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w+\b"
    )

    benchmark_matrix = vectorizer.fit_transform(
        df["search_text"]
    )

    # Precompute normalized benchmark names
    benchmark_names = df["normalized_name"].tolist()

    return (
        df,
        vectorizer,
        benchmark_matrix,
        benchmark_names
    )


# =========================================================
# 5. PARSE ALTERNATIVE NAMES
# =========================================================

def parse_alternatives(description):

    if not description:
        return []

    alternatives = re.split(
        r"[|\r\n]+",
        description
    )

    cleaned = []
    seen = set()

    for item in alternatives:

        item = item.strip()

        normalized = normalize_text(item)

        if normalized and normalized not in seen:

            cleaned.append(item)

            seen.add(normalized)

    return cleaned


# =========================================================
# 6. PREPARE SEARCH QUERIES
# =========================================================

def prepare_queries(equipment_name, alternatives):

    queries = [equipment_name] + alternatives

    unique_queries = []
    seen = set()

    for query in queries:

        normalized = normalize_text(query)

        if normalized and normalized not in seen:

            unique_queries.append(normalized)

            seen.add(normalized)

    return unique_queries


# =========================================================
# 7. TF-IDF CANDIDATE RETRIEVAL
# =========================================================

def retrieve_candidates(
    queries,
    vectorizer,
    benchmark_matrix
):

    # Combine the equipment name and all alternatives
    query_context = " ".join(queries)

    query_vector = vectorizer.transform(
        [query_context]
    )

    # Compare input context against all benchmark records
    similarities = (
        benchmark_matrix @ query_vector.T
    ).toarray().ravel()

    # Ignore candidates with zero relevance
    relevant_indices = np.flatnonzero(
        similarities > 0
    )

    if len(relevant_indices) == 0:

        return {}, similarities

    # Limit the candidate pool
    candidate_count = min(
        CANDIDATE_LIMIT,
        len(relevant_indices)
    )

    if len(relevant_indices) > candidate_count:

        relevant_scores = similarities[relevant_indices]

        selected_positions = np.argpartition(
            relevant_scores,
            -candidate_count
        )[-candidate_count:]

        candidate_indices = relevant_indices[
            selected_positions
        ]

    else:

        candidate_indices = relevant_indices

    # Sort candidates by TF-IDF relevance
    candidate_indices = candidate_indices[
        np.argsort(
            similarities[candidate_indices]
        )[::-1]
    ]

    candidate_scores = {

        int(index): float(similarities[index])

        for index in candidate_indices

    }

    return candidate_scores, similarities


# =========================================================
# 8. RAPIDFUZZ FALLBACK
# =========================================================

def retrieve_fuzzy_candidates(
    queries,
    benchmark_names
):

    candidates = {}

    for query in queries:

        matches = process.extract(
            query,
            benchmark_names,
            scorer=fuzz.WRatio,
            processor=None,
            limit=CANDIDATE_LIMIT
        )

        for matched_name, score, index in matches:

            candidates[index] = 0.0

    return candidates


# =========================================================
# 9. MATCHING ENGINE
# =========================================================

def find_best_matches(
    equipment_name,
    alternatives,
    benchmark_df,
    vectorizer,
    benchmark_matrix,
    benchmark_names
):

    start_time = time.perf_counter()

    queries = prepare_queries(
        equipment_name,
        alternatives
    )

    if not queries:

        return [], 0.0, 0

    # ---------------------------------------------
    # Stage 1: TF-IDF candidate retrieval
    # ---------------------------------------------

    candidate_scores, similarities = retrieve_candidates(
        queries,
        vectorizer,
        benchmark_matrix
    )

    # ---------------------------------------------
    # Stage 2: Fallback if no vocabulary overlaps
    # ---------------------------------------------

    if not candidate_scores:

        candidate_scores = retrieve_fuzzy_candidates(
            queries,
            benchmark_names
        )

    # ---------------------------------------------
    # Stage 3: RapidFuzz name matching
    # ---------------------------------------------

    results = []

    for index, catalogue_score in candidate_scores.items():

        row = benchmark_df.iloc[index]

        benchmark_name = row["normalized_name"]

        # Find the highest name similarity
        # across original name and alternatives
        name_score = max(
            fuzz.WRatio(query, benchmark_name)
            for query in queries
        )

        # Convert TF-IDF score into percentage
        catalogue_score = catalogue_score * 100

        # Calculate combined score
        if catalogue_score > 0:

            final_score = (
                NAME_WEIGHT * name_score
                + CATALOGUE_WEIGHT * catalogue_score
            )

        else:

            final_score = name_score

        results.append({

            "code": row["code"],

            "name": row["name"],

            "description": row["description"],

            "name_score": name_score,

            "catalogue_score": catalogue_score,

            "final_score": final_score

        })

    # ---------------------------------------------
    # Stage 4: Rank candidates
    # ---------------------------------------------

    results.sort(
        key=lambda item: (
            -item["final_score"],
            -item["name_score"],
            item["name"].lower(),
            item["code"]
        )
    )

    elapsed = time.perf_counter() - start_time

    return (
        results[:3],
        elapsed,
        len(candidate_scores)
    )


# =========================================================
# 10. SESSION STATE
# =========================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None

if "search_time" not in st.session_state:
    st.session_state.search_time = 0.0

if "candidate_count" not in st.session_state:
    st.session_state.candidate_count = 0

if "submitted_equipment_name" not in st.session_state:
    st.session_state.submitted_equipment_name = ""


# =========================================================
# 11. EXCEL UPLOAD
# =========================================================

st.divider()

st.subheader("Benchmark Catalogue")

uploaded_file = st.file_uploader(
    "Upload benchmark Excel file",
    type=["xlsx"],
    help=(
        "Column A: Benchmark Code | "
        "Column B: Benchmark Name | "
        "Column C: Benchmark Description"
    )
)

if uploaded_file is None:

    st.session_state.results = []

    st.session_state.file_signature = None

    st.info(
        "Upload your benchmark Excel file to begin."
    )

    st.stop()


# Read uploaded Excel bytes
file_bytes = uploaded_file.getvalue()

# Identify uploaded file
file_signature = hashlib.sha256(
    file_bytes
).hexdigest()


# Clear previous results if file changes
if file_signature != st.session_state.file_signature:

    st.session_state.results = []

    st.session_state.file_signature = file_signature

    st.session_state.pop(
        "selected_equipment",
        None
    )


# Load benchmark catalogue
try:

    (
        benchmark_df,
        vectorizer,
        benchmark_matrix,
        benchmark_names

    ) = load_benchmark(file_bytes)

    st.success(
        f"Benchmark loaded: "
        f"{len(benchmark_df):,} records"
    )

except Exception as error:

    st.error(
        f"Unable to load benchmark Excel: {error}"
    )

    st.stop()


# =========================================================
# 12. USER INPUT
# =========================================================

st.divider()

st.subheader("Equipment Information")

with st.form("equipment_matching_form"):

    equipment_name = st.text_input(
        "Equipment Name",
        placeholder="Example: Pressure Measurement"
    )

    alternative_text = st.text_area(
        "Alternative Names",
        placeholder=(
            "Pressure sensor | "
            "Pressure monitoring device | PT"
        ),
        height=130,
        help=(
            "Separate alternative names using | "
            "or enter each name on a new line."
        )
    )

    submitted = st.form_submit_button(
        "Find Matching Equipment",
        type="primary",
        use_container_width=True
    )


# =========================================================
# 13. EXECUTE MATCHING
# =========================================================

if submitted:

    # Clear old results
    st.session_state.results = []

    if not equipment_name.strip():

        st.warning(
            "Please enter an equipment name."
        )

    else:

        alternatives = parse_alternatives(
            alternative_text
        )

        try:

            with st.spinner(
                "Searching benchmark catalogue..."
            ):

                (
                    results,
                    elapsed,
                    candidate_count

                ) = find_best_matches(

                    equipment_name=equipment_name,

                    alternatives=alternatives,

                    benchmark_df=benchmark_df,

                    vectorizer=vectorizer,

                    benchmark_matrix=benchmark_matrix,

                    benchmark_names=benchmark_names
                )

            st.session_state.results = results

            st.session_state.submitted_equipment_name = (
                equipment_name.strip()
            )

            st.session_state.search_time = elapsed

            st.session_state.candidate_count = (
                candidate_count
            )

            st.session_state.pop(
                "selected_equipment",
                None
            )

        except Exception as error:

            st.error(
                f"Matching failed: {error}"
            )


# =========================================================
# 14. DISPLAY TOP 3 RESULTS
# =========================================================

if st.session_state.results:

    results = st.session_state.results

    st.divider()

    st.subheader("Top 3 Benchmark Matches")

    st.caption(
        f"Input equipment: "
        f"{st.session_state.submitted_equipment_name}"
    )

    # Prepare result table
    results_df = pd.DataFrame([

        {
            "Rank": index + 1,

            "Benchmark Code": result["code"],

            "Benchmark Name": result["name"],

            "Description": result["description"],

            "Name Similarity (%)": round(
                result["name_score"], 2
            ),

            "Catalogue Relevance (%)": round(
                result["catalogue_score"], 2
            ),

            "Combined Score (%)": round(
                result["final_score"], 2
            )

        }

        for index, result in enumerate(results)

    ])

    st.dataframe(
        results_df,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Combined score: 75% name similarity + "
        "25% catalogue relevance when TF-IDF finds relevant terms."
    )

    # Performance information
    st.caption(
        f"Search time: {st.session_state.search_time:.2f} seconds "
        f"| Candidates evaluated: "
        f"{st.session_state.candidate_count:,}"
    )


    # =====================================================
    # 15. FINAL EQUIPMENT SELECTION
    # =====================================================

    st.divider()

    st.subheader("Final Equipment Name")

    # Use code in labels so duplicate names remain distinguishable
    options = {
        f"{result['name']} — {result['code']}": result
        for result in results
    }

    selected_label = st.selectbox(
        "Select the correct benchmark equipment",
        options=list(options.keys()),
        key="selected_equipment"
    )

    selected_result = options[selected_label]

    st.success(
        f"Final selected equipment name: "
        f"{selected_result['name']}"
    )

    st.write(
        f"**Benchmark Code:** {selected_result['code']}"
    )

    st.write(
        f"**Description:** "
        f"{selected_result['description'] or 'No description available'}"
    )

else:

    st.info(
        "Enter an equipment name and alternative names, "
        "then click Find Matching Equipment."
    )