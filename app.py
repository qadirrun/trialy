
import io
import re

import pandas as pd
import streamlit as st

from rapidfuzz import fuzz, process, utils


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Equipment Name Matcher",
    page_icon="",
    layout="centered"
)

st.title("Equipment Name Matcher")

st.write(
    "Find the closest benchmark equipment name "
    "using your uploaded equipment catalogue."
)


# ==========================================
# LOAD AND CACHE BENCHMARK EXCEL
# ==========================================

@st.cache_data(show_spinner="Loading benchmark catalogue...")
def load_benchmark(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        usecols=[0, 1],
        dtype=str,
        engine="openpyxl"
    )

    df.columns = ["code", "name"]

    # Clean benchmark names
    df["name"] = (
        df["name"]
        .fillna("")
        .str.strip()
    )

    df = df[df["name"] != ""].copy()

    # Normalize names for duplicate detection
    df["normalized"] = df["name"].apply(
        lambda x: utils.default_process(x) or ""
    )

    # Remove duplicate benchmark names
    df = df[df["normalized"] != ""]

    df = df.drop_duplicates(
        subset=["normalized"],
        keep="first"
    )

    return df[["code", "name"]].reset_index(drop=True)


# ==========================================
# PARSE ALTERNATIVE NAMES
# ==========================================

def parse_alternatives(description):

    alternatives = re.split(
        r"[|\n]+",
        description
    )

    cleaned = []
    seen = set()

    for item in alternatives:

        item = item.strip()

        normalized = utils.default_process(item)

        if normalized and normalized not in seen:
            cleaned.append(item)
            seen.add(normalized)

    return cleaned


# ==========================================
# MATCHING ENGINE
# ==========================================

def find_best_matches(queries, benchmark_names):

    best_scores = {}

    for query in queries:

        matches = process.extract(
            query,
            benchmark_names,
            scorer=fuzz.WRatio,
            processor=utils.default_process,
            limit=3
        )

        for name, score, index in matches:

            if (
                index not in best_scores
                or score > best_scores[index]["score"]
            ):

                best_scores[index] = {
                    "name": name,
                    "score": score,
                    "index": index
                }

    # Rank all unique benchmark candidates
    ranked = sorted(
        best_scores.values(),
        key=lambda x: (-x["score"], x["name"].lower())
    )

    return ranked[:3]


# ==========================================
# EXCEL UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Benchmark Excel",
    type=["xlsx"]
)

if uploaded_file is None:

    st.info("Upload your benchmark Excel file to begin.")

    st.stop()


try:

    benchmark_df = load_benchmark(
        uploaded_file.getvalue()
    )

    benchmark_names = benchmark_df["name"].tolist()

    st.success(
        f"Benchmark loaded: {len(benchmark_names):,} unique names"
    )

except Exception as e:

    st.error(f"Unable to read Excel file: {e}")

    st.stop()


# ==========================================
# USER INPUT
# ==========================================

st.divider()

equipment_name = st.text_input(
    "Equipment Name",
    placeholder="Example: Analysis Time"
)

alternative_text = st.text_area(
    "Alternative Names",
    placeholder=(
        "quoted delivery time | timer | autonomy time | "
        "stress analysis required | TIME | TIMEPL"
    ),
    height=150
)

# ==========================================
# RUN MATCHING
# ==========================================

if st.button("Find Matching Equipment", type="primary"):

    if not equipment_name.strip():

        st.warning("Please enter an equipment name.")

    else:

        alternatives = parse_alternatives(
            alternative_text
        )

        # Include the original equipment name as well
        queries = [equipment_name.strip()] + alternatives

        with st.spinner("Finding closest benchmark names..."):

            results = find_best_matches(
                queries,
                benchmark_names
            )

        if not results:

            st.warning("No matching equipment found.")

        else:

            st.divider()

            st.subheader("Top 3 Benchmark Matches")

            results_df = pd.DataFrame([
                {
                    "Rank": i + 1,
                    "Benchmark Name": result["name"],
                    "Similarity (%)": round(result["score"], 2)
                }
                for i, result in enumerate(results)
            ])

            st.dataframe(
                results_df,
                hide_index=True,
                use_container_width=True
            )

            st.divider()

            st.subheader("Final Equipment Name")

            selected_name = st.selectbox(
                "Choose the correct benchmark name",
                options=[r["name"] for r in results]
            )

            st.success(
                f"Final selected equipment name: {selected_name}"
            )
