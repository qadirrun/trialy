
import streamlit as st
import pandas as pd
import pymupdf
import tempfile
import unicodedata
import hashlib
import re
import math
import time
import json
import io

from pathlib import Path
from collections import Counter, defaultdict
from rapidfuzz import fuzz, process

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption
)

from docling.datamodel.base_models import InputFormat

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode
)

from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
    AcceleratorDevice
)

# ==========================================
# STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Engineering Document Intelligence",
    page_icon="📄",
    layout="wide"
)

st.title("Engineering Document Intelligence")

st.caption(
    "Docling OCR and Excel-based "
    "Engineering Datasheet Classification"
)

# ==========================================
# SESSION STATE
# ==========================================

defaults = {
    "results": [],
    "master_profiles": {},
    "master_weights": {},
    "master_filename": None,
    "master_loaded": False,
    "master_summary": {},
    "master_version": 0,
    "pdf_version": 0,
    "processed_files": set()
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value

# ==========================================
# TEXT NORMALIZATION
# ==========================================

def normalize(text):

    text = unicodedata.normalize(
        "NFKC",
        str(text)
    )

    text = text.lower()

    text = re.sub(
        r"[\W_]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# ==========================================
# FILE HASH
# ==========================================

def get_file_hash(file_bytes):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()

# ==========================================
# CACHE DOCLING MODEL
# ==========================================

@st.cache_resource(
    show_spinner=False
)
def load_converter(
    full_page_ocr,
    table_mode
):

    options = PdfPipelineOptions()

    options.do_ocr = True

    options.do_table_structure = True

    options.ocr_options = RapidOcrOptions(
        force_full_page_ocr=full_page_ocr,
        backend="onnxruntime"
    )

    options.table_structure_options.mode = (
        TableFormerMode.FAST
        if table_mode == "FAST"
        else TableFormerMode.ACCURATE
    )

    options.table_structure_options.do_cell_matching = True

    options.accelerator_options = AcceleratorOptions(
        num_threads=2,
        device=AcceleratorDevice.CPU
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options
            )
        }
    )

    return converter

# ==========================================
# BUILD MASTERLIST PROFILES
# ==========================================

def build_masterlist(
    df,
    domain_col,
    group_col,
    entity_col,
    attribute_col,
    forward_fill,
    alias_col=None
):

    selected_columns = [
        domain_col,
        group_col,
        entity_col,
        attribute_col
    ]

    if alias_col:

        selected_columns.append(alias_col)

    data = df[selected_columns].copy()

    new_columns = [
        "domain",
        "group",
        "entity",
        "attribute"
    ]

    if alias_col:

        new_columns.append("aliases")

    data.columns = new_columns

    data = data.replace(
        r"^\s*$",
        pd.NA,
        regex=True
    )

    # Support merged Excel hierarchy cells.
    # Reset child hierarchy when parent changes.

    if forward_fill:

        current_domain = None
        current_group = None
        current_entity = None

        for index, row in data.iterrows():

            if pd.notna(row["domain"]):

                domain = str(row["domain"]).strip()

                if domain != current_domain:

                    current_domain = domain
                    current_group = None
                    current_entity = None

            if pd.notna(row["group"]):

                group = str(row["group"]).strip()

                if group != current_group:

                    current_group = group
                    current_entity = None

            if pd.notna(row["entity"]):

                current_entity = str(
                    row["entity"]
                ).strip()

            data.at[index, "domain"] = current_domain

            data.at[index, "group"] = current_group

            data.at[index, "entity"] = current_entity

    data = data.dropna(
        subset=[
            "domain",
            "group",
            "entity",
            "attribute"
        ]
    )

    profiles = defaultdict(dict)

    for _, row in data.iterrows():

        key = (
            str(row["domain"]).strip(),
            str(row["group"]).strip(),
            str(row["entity"]).strip()
        )

        attribute = str(
            row["attribute"]
        ).strip()

        normalized = normalize(attribute)

        if not normalized:

            continue

        if normalized not in profiles[key]:

            profiles[key][normalized] = {
                "name": attribute,
                "aliases": {normalized}
            }

        # Optional aliases or synonyms.

        if alias_col and pd.notna(row["aliases"]):

            aliases = re.split(
                r"[;\n|]",
                str(row["aliases"])
            )

            for alias in aliases:

                cleaned = normalize(alias)

                if cleaned:

                    profiles[key][normalized][
                        "aliases"
                    ].add(cleaned)

    # Count attribute frequency across entities.

    frequency = Counter()

    for attributes in profiles.values():

        profile_terms = set()

        for information in attributes.values():

            profile_terms.update(
                information["aliases"]
            )

        frequency.update(profile_terms)

    total_profiles = len(profiles)

    # Distinctive attributes receive higher weights.

    weights = {
        term: 1 + math.log(
            (total_profiles + 1) / (count + 1)
        )
        for term, count in frequency.items()
    }

    return dict(profiles), weights

# ==========================================
# OCR PDF USING DOCLING
# ==========================================

def extract_pdf(
    pdf_bytes,
    converter
):

    with tempfile.TemporaryDirectory() as tmp:

        pdf_path = Path(tmp) / "document.pdf"

        pdf_path.write_bytes(pdf_bytes)

        result = converter.convert(
            pdf_path,
            raises_on_error=False
        )

        status = result.status.value

        if status not in [
            "success",
            "partial_success"
        ]:

            raise RuntimeError(
                f"Docling conversion failed: {status}"
            )

        document = result.document

        markdown = document.export_to_markdown()

        json_data = json.dumps(
            document.export_to_dict(),
            indent=2,
            ensure_ascii=False
        )

        segments = []

        tables = []

        warnings = []

        # ----------------------------------
        # Extract text blocks
        # ----------------------------------

        for item in document.texts:

            text = getattr(
                item,
                "text",
                ""
            )

            if text:

                segments.extend(
                    str(text).splitlines()
                )

        # ----------------------------------
        # Extract table cells
        # ----------------------------------

        for index, table in enumerate(
            document.tables,
            start=1
        ):

            try:

                for cell in table.data.table_cells:

                    text = getattr(
                        cell,
                        "text",
                        ""
                    )

                    if text:

                        segments.append(
                            str(text)
                        )

                df = table.export_to_dataframe(
                    doc=document
                )

                tables.append({
                    "name": f"Table {index}",
                    "data": df
                })

            except Exception as error:

                warnings.append(
                    f"Table {index}: {error}"
                )

        # ----------------------------------
        # Markdown fallback
        # ----------------------------------

        for line in markdown.splitlines():

            segments.extend(
                re.split(
                    r"[|:]",
                    line
                )
            )

        # ----------------------------------
        # Normalize extracted content
        # ----------------------------------

        cleaned_segments = set()

        for segment in segments:

            cleaned = normalize(segment)

            if cleaned:

                cleaned_segments.add(cleaned)

        return {
            "markdown": markdown,
            "json": json_data,
            "segments": cleaned_segments,
            "tables": tables,
            "status": status,
            "warnings": warnings,
            "detected_tables": len(document.tables),
            "detected_pictures": len(document.pictures)
        }

# ==========================================
# BUILD DOCUMENT PHRASE INDEX
# ==========================================

def build_phrase_index(
    segments,
    max_words
):

    phrase_index = defaultdict(set)

    for segment in segments:

        words = segment.split()

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

                phrase_index[size].add(phrase)

    return {
        size: sorted(phrases)
        for size, phrases in phrase_index.items()
    }

# ==========================================
# ATTRIBUTE MATCHING ENGINE
# ==========================================

def match_attributes(
    segments,
    profiles,
    fuzzy_threshold,
    enable_fuzzy
):

    all_terms = {
        alias
        for attributes in profiles.values()
        for information in attributes.values()
        for alias in information["aliases"]
    }

    if not all_terms:

        return {}

    max_words = max(
        len(term.split())
        for term in all_terms
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

    for term in sorted(all_terms):

        # ----------------------------------
        # Exact matching
        # ----------------------------------

        if term in all_phrases:

            matches[term] = {
                "matched_text": term,
                "similarity": 100.0,
                "method": "Exact"
            }

            continue

        # ----------------------------------
        # Fuzzy matching
        # ----------------------------------

        if not enable_fuzzy:

            continue

        # Avoid fuzzy matching short abbreviations.

        if len(term) < 7:

            continue

        word_count = len(term.split())

        candidates = []

        for size in range(
            max(1, word_count - 1),
            word_count + 2
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

            matched_text, similarity, _ = best

            matches[term] = {
                "matched_text": matched_text,
                "similarity": float(similarity),
                "method": "Fuzzy"
            }

    return matches

# ==========================================
# DOCUMENT CLASSIFICATION
# ==========================================

def classify_document(
    segments,
    profiles,
    weights,
    fuzzy_threshold,
    enable_fuzzy,
    minimum_score,
    minimum_matches,
    minimum_margin
):

    evidence = match_attributes(
        segments,
        profiles,
        fuzzy_threshold,
        enable_fuzzy
    )

    candidates = []

    for key, attributes in profiles.items():

        domain, group, entity = key

        matched = []

        missing = []

        matched_weight = 0.0

        total_weight = 0.0

        used_phrases = set()

        ordered_attributes = sorted(
            attributes.items(),
            key=lambda item: (
                min(
                    weights.get(alias, 1.0)
                    for alias in item[1]["aliases"]
                ),
                len(item[0])
            ),
            reverse=True
        )

        for normalized, information in ordered_attributes:

            attribute_name = information["name"]

            aliases = information["aliases"]

            weight = min(
                weights.get(alias, 1.0)
                for alias in aliases
            )

            total_weight += weight

            possible_matches = [
                evidence[alias]
                for alias in aliases
                if alias in evidence
                and evidence[alias]["matched_text"]
                not in used_phrases
            ]

            if not possible_matches:

                missing.append(attribute_name)

                continue

            best = max(
                possible_matches,
                key=lambda x: x["similarity"]
            )

            matched_weight += (
                weight * best["similarity"] / 100
            )

            used_phrases.add(
                best["matched_text"]
            )

            matched.append({
                "Attribute": attribute_name,
                "Matched OCR Text": best["matched_text"],
                "Similarity (%)": round(
                    best["similarity"],
                    2
                ),
                "Method": best["method"],
                "Weight": round(weight, 3)
            })

        if total_weight == 0:

            continue

        coverage = (
            matched_weight / total_weight
        )

        support = 1 - math.exp(
            -matched_weight / 5
        )

        score = 100 * (
            0.70 * coverage +
            0.30 * support
        )

        candidates.append({
            "Data Domain": domain,
            "Data Group": group,
            "Entity": entity,
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

    # No matching evidence.

    if not candidates or candidates[0]["Matched"] == 0:

        return {
            "best": None,
            "candidates": [],
            "status": "Review Required",
            "margin": 0,
            "reasons": [
                "No matching attributes detected"
            ]
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

        reasons.append(
            "Matching score below threshold"
        )

    if best["Matched"] < minimum_matches:

        reasons.append(
            "Insufficient matched attributes"
        )

    if (
        len(candidates) > 1
        and margin < minimum_margin
    ):

        reasons.append(
            "Multiple entities have similar scores"
        )

    status = (
        "Review Required"
        if reasons
        else "Classified"
    )

    return {
        "best": best,
        "candidates": candidates[:10],
        "status": status,
        "margin": round(margin, 2),
        "reasons": reasons
    }

# ==========================================
# BUILD SUMMARY DATAFRAME
# ==========================================

def build_summary(results):

    summary = []

    for result in results:

        classification = result.get(
            "classification",
            {}
        )

        best = classification.get("best") or {}

        summary.append({
            "File Name": result["name"],

            "Data Domain": best.get(
                "Data Domain",
                "-"
            ),

            "Data Group": best.get(
                "Data Group",
                "-"
            ),

            "Entity": best.get(
                "Entity",
                "-"
            ),

            "Score (%)": best.get(
                "Score",
                0
            ),

            "Coverage (%)": best.get(
                "Coverage",
                0
            ),

            "Matched Attributes": best.get(
                "Matched",
                0
            ),

            "Total Attributes": best.get(
                "Total Attributes",
                0
            ),

            "Status": classification.get(
                "status",
                "Failed"
            ),

            "OCR Status": result.get(
                "status",
                ""
            ),

            "Review Reason": "; ".join(
                classification.get(
                    "reasons",
                    []
                )
            ),

            "Time (s)": round(
                result.get("time", 0),
                2
            ),

            "Error": result.get(
                "error",
                ""
            )
        })

    return pd.DataFrame(summary)

# ==========================================
# EXPORT RESULTS TO EXCEL
# ==========================================

def create_excel(results):

    summary_df = build_summary(results)

    attributes = []

    alternative_candidates = []

    for result in results:

        classification = result.get(
            "classification",
            {}
        )

        best = classification.get("best") or {}

        for match in best.get(
            "matched_attributes",
            []
        ):

            attributes.append({
                "File Name": result["name"],
                "Entity": best.get("Entity", ""),
                **match,
                "Status": "Matched"
            })

        for missing in best.get(
            "missing_attributes",
            []
        ):

            attributes.append({
                "File Name": result["name"],
                "Entity": best.get("Entity", ""),
                "Attribute": missing,
                "Status": "Not Detected"
            })

        for candidate in classification.get(
            "candidates",
            []
        ):

            alternative_candidates.append({
                "File Name": result["name"],
                "Data Domain": candidate["Data Domain"],
                "Data Group": candidate["Data Group"],
                "Entity": candidate["Entity"],
                "Score": candidate["Score"],
                "Matched": candidate["Matched"],
                "Coverage": candidate["Coverage"]
            })

    buffer = io.BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="Classification",
            index=False
        )

        pd.DataFrame(attributes).to_excel(
            writer,
            sheet_name="Attribute Matching",
            index=False
        )

        pd.DataFrame(alternative_candidates).to_excel(
            writer,
            sheet_name="Candidates",
            index=False
        )

        # Prevent uploaded strings beginning with "="
        # from becoming executable Excel formulas.

        for worksheet in writer.book.worksheets:

            for row in worksheet.iter_rows():

                for cell in row:

                    if cell.data_type == "f":

                        cell.data_type = "s"

    return buffer.getvalue()

# ==========================================
# SIDEBAR SETTINGS
# ==========================================

with st.sidebar:

    st.header("OCR Configuration")

    table_mode = st.selectbox(
        "Table recognition mode",
        ["FAST", "ACCURATE"]
    )

    full_page_ocr = st.checkbox(
        "Force full-page OCR",
        value=False
    )

    st.divider()

    st.header("Classification Settings")

    enable_fuzzy = st.checkbox(
        "Enable fuzzy matching",
        value=True
    )

    fuzzy_threshold = st.slider(
        "Fuzzy matching threshold",
        min_value=70,
        max_value=100,
        value=90
    )

    minimum_score = st.slider(
        "Minimum classification score",
        min_value=0,
        max_value=100,
        value=45
    )

    minimum_matches = st.number_input(
        "Minimum matched attributes",
        min_value=1,
        max_value=100,
        value=3
    )

    minimum_margin = st.slider(
        "Minimum score difference",
        min_value=0,
        max_value=30,
        value=5
    )

    st.divider()

    st.header("Session Management")

    if st.session_state.master_loaded:

        st.success("Masterlist loaded")

        st.caption(
            st.session_state.master_filename
        )

    else:

        st.warning("No masterlist loaded")

    st.metric(
        "Processed documents",
        len(st.session_state.results)
    )

    if st.button(
        "Clear Classification Results",
        use_container_width=True
    ):

        # Keep masterlist loaded.

        st.session_state.results = []

        st.session_state.processed_files = set()

        st.session_state.pdf_version += 1

        st.rerun()

# ==========================================
# MAIN NAVIGATION
# ==========================================

master_tab, pdf_tab, result_tab = st.tabs([
    "Masterlist",
    "PDF Classification",
    "Results"
])

# ==========================================
# TAB 1: MASTERLIST MANAGEMENT
# ==========================================

with master_tab:

    st.header("Classification Masterlist")

    # --------------------------------------
    # MASTERLIST ALREADY SAVED
    # --------------------------------------

    if st.session_state.master_loaded:

        st.success(
            "Masterlist is saved and ready to use. "
            "You do not need to upload it again "
            "during this session."
        )

        st.caption(
            f"Active masterlist: "
            f"{st.session_state.master_filename}"
        )

        summary = st.session_state.master_summary

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Data Domains",
            summary["domains"]
        )

        c2.metric(
            "Data Groups",
            summary["groups"]
        )

        c3.metric(
            "Entities",
            summary["entities"]
        )

        c4.metric(
            "Attributes",
            summary["attributes"]
        )

        profile_preview = []

        for key, attributes in (
            st.session_state.master_profiles.items()
        ):

            profile_preview.append({
                "Data Domain": key[0],
                "Data Group": key[1],
                "Entity": key[2],
                "Attributes": len(attributes)
            })

        st.subheader("Available Classification Profiles")

        st.dataframe(
            pd.DataFrame(profile_preview),
            hide_index=True,
            use_container_width=True
        )

        # View individual entity attributes.

        profile_keys = list(
            st.session_state.master_profiles.keys()
        )

        selected_profile = st.selectbox(
            "Inspect entity attributes",
            profile_keys,
            format_func=lambda key: " / ".join(key)
        )

        if selected_profile:

            attributes = (
                st.session_state.master_profiles[
                    selected_profile
                ]
            )

            attribute_preview = []

            for information in attributes.values():

                attribute_preview.append({
                    "Attribute": information["name"],
                    "Aliases": "; ".join(
                        sorted(information["aliases"])
                    )
                })

            st.dataframe(
                pd.DataFrame(attribute_preview),
                hide_index=True,
                use_container_width=True
            )

        st.divider()

        st.warning(
            "Replacing the masterlist will clear "
            "existing classification results."
        )

        if st.button(
            "Replace Masterlist",
            use_container_width=True
        ):

            st.session_state.master_profiles = {}

            st.session_state.master_weights = {}

            st.session_state.master_filename = None

            st.session_state.master_loaded = False

            st.session_state.master_summary = {}

            st.session_state.results = []

            st.session_state.processed_files = set()

            st.session_state.master_version += 1

            st.session_state.pdf_version += 1

            st.rerun()

    # --------------------------------------
    # UPLOAD NEW MASTERLIST
    # --------------------------------------

    else:

        st.info(
            "Upload your Excel masterlist once "
            "before classifying PDF documents."
        )

        master_file = st.file_uploader(
            "Upload Excel Masterlist",
            type=["xlsx"],
            key=(
                f"master_upload_"
                f"{st.session_state.master_version}"
            )
        )

        if master_file:

            try:

                workbook = pd.ExcelFile(
                    io.BytesIO(
                        master_file.getvalue()
                    ),
                    engine="openpyxl"
                )

                sheet = st.selectbox(
                    "Select worksheet",
                    workbook.sheet_names
                )

                header_row = st.number_input(
                    "Excel header row",
                    min_value=1,
                    value=1
                )

                master_df = pd.read_excel(
                    workbook,
                    sheet_name=sheet,
                    header=int(header_row) - 1,
                    dtype=str
                )

                master_df.columns = [
                    str(column).strip()
                    for column in master_df.columns
                ]

                columns = list(
                    master_df.columns
                )

                with st.expander(
                    "Preview Masterlist",
                    expanded=True
                ):

                    st.dataframe(
                        master_df.head(100),
                        use_container_width=True
                    )

                if len(columns) < 4:

                    st.error(
                        "Masterlist must contain "
                        "at least four columns."
                    )

                elif len(set(columns)) != len(columns):

                    st.error(
                        "Excel contains duplicate "
                        "column names."
                    )

                else:

                    # --------------------------
                    # COLUMN AUTO-DETECTION
                    # --------------------------

                    def default_column(
                        name,
                        fallback
                    ):

                        target = normalize(name)

                        for index, column in enumerate(
                            columns
                        ):

                            if normalize(column) == target:

                                return index

                        return min(
                            fallback,
                            len(columns) - 1
                        )

                    st.subheader(
                        "Map Masterlist Columns"
                    )

                    c1, c2 = st.columns(2)

                    with c1:

                        domain_col = st.selectbox(
                            "Data Domain",
                            columns,
                            index=default_column(
                                "Data Domain",
                                0
                            )
                        )

                        group_col = st.selectbox(
                            "Data Group",
                            columns,
                            index=default_column(
                                "Data Group",
                                1
                            )
                        )

                    with c2:

                        entity_col = st.selectbox(
                            "Entity",
                            columns,
                            index=default_column(
                                "Entity",
                                2
                            )
                        )

                        attribute_col = st.selectbox(
                            "Attribute",
                            columns,
                            index=default_column(
                                "Attribute",
                                3
                            )
                        )

                    alias_col = st.selectbox(
                        "Synonym / Alias (Optional)",
                        options=[None] + columns,
                        format_func=lambda x: (
                            "Not Available"
                            if x is None
                            else x
                        )
                    )

                    forward_fill = st.checkbox(
                        "Fill blank hierarchy cells "
                        "(for merged Excel cells)",
                        value=True
                    )

                    # --------------------------
                    # SAVE MASTERLIST
                    # --------------------------

                    if st.button(
                        "Save Masterlist",
                        type="primary",
                        use_container_width=True
                    ):

                        selected_columns = [
                            domain_col,
                            group_col,
                            entity_col,
                            attribute_col
                        ]

                        if alias_col:

                            selected_columns.append(
                                alias_col
                            )

                        if len(
                            set(selected_columns)
                        ) != len(selected_columns):

                            st.error(
                                "Please select "
                                "different columns."
                            )

                        else:

                            with st.spinner(
                                "Building masterlist profiles..."
                            ):

                                profiles, weights = (
                                    build_masterlist(
                                        master_df,
                                        domain_col,
                                        group_col,
                                        entity_col,
                                        attribute_col,
                                        forward_fill,
                                        alias_col
                                    )
                                )

                            if not profiles:

                                st.error(
                                    "No valid entity "
                                    "profiles found."
                                )

                            else:

                                st.session_state.master_profiles = (
                                    profiles
                                )

                                st.session_state.master_weights = (
                                    weights
                                )

                                st.session_state.master_filename = (
                                    master_file.name
                                )

                                st.session_state.master_loaded = True

                                st.session_state.master_summary = {

                                    "domains": len({
                                        key[0]
                                        for key in profiles
                                    }),

                                    "groups": len({
                                        (key[0], key[1])
                                        for key in profiles
                                    }),

                                    "entities": len(
                                        profiles
                                    ),

                                    "attributes": sum(
                                        len(attributes)
                                        for attributes
                                        in profiles.values()
                                    )
                                }

                                st.session_state.results = []

                                st.session_state.processed_files = set()

                                st.rerun()

            except Exception as error:

                st.error(
                    f"Unable to load masterlist: {error}"
                )

# ==========================================
# TAB 2: PDF CLASSIFICATION
# ==========================================

with pdf_tab:

    st.header("PDF Datasheet Classification")

    if not st.session_state.master_loaded:

        st.warning(
            "Please upload and save the Excel "
            "masterlist in the Masterlist tab first."
        )

    else:

        st.success(
            "Using saved masterlist: "
            f"{st.session_state.master_filename}"
        )

        uploaded_files = st.file_uploader(
            "Upload PDF Datasheets",
            type=["pdf"],
            accept_multiple_files=True,
            key=(
                f"pdf_upload_"
                f"{st.session_state.pdf_version}"
            )
        )

        skip_duplicates = st.checkbox(
            "Skip previously processed PDF files",
            value=True
        )

        if uploaded_files:

            st.caption(
                f"{len(uploaded_files)} PDF file(s) uploaded."
            )

            preview_files = []

            for uploaded_file in uploaded_files:

                file_hash = get_file_hash(
                    uploaded_file.getvalue()
                )

                already_processed = (
                    file_hash in
                    st.session_state.processed_files
                )

                preview_files.append({
                    "File Name": uploaded_file.name,
                    "Size (MB)": round(
                        uploaded_file.size / 1024 / 1024,
                        2
                    ),
                    "Status": (
                        "Already Processed"
                        if already_processed
                        else "Ready"
                    )
                })

            st.dataframe(
                pd.DataFrame(preview_files),
                hide_index=True,
                use_container_width=True
            )

        # ----------------------------------
        # START OCR
        # ----------------------------------

        start = st.button(
            "Extract & Classify Documents",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files
        )

        if start:

            profiles = (
                st.session_state.master_profiles
            )

            weights = (
                st.session_state.master_weights
            )

            converter = None

            try:

                with st.spinner(
                    "Initializing Docling..."
                ):

                    converter = load_converter(
                        full_page_ocr,
                        table_mode
                    )

            except Exception as error:

                st.error(
                    f"Unable to initialize Docling: {error}"
                )

            if converter is not None:

                progress = st.progress(0)

                status_message = st.empty()

                classified_count = 0

                review_count = 0

                failed_count = 0

                skipped_count = 0

                total_files = len(uploaded_files)

                for index, uploaded_file in enumerate(
                    uploaded_files,
                    start=1
                ):

                    filename = uploaded_file.name

                    pdf_bytes = (
                        uploaded_file.getvalue()
                    )

                    file_hash = get_file_hash(
                        pdf_bytes
                    )

                    if (
                        skip_duplicates
                        and file_hash in
                        st.session_state.processed_files
                    ):

                        skipped_count += 1

                        progress.progress(
                            index / total_files
                        )

                        continue

                    start_time = time.time()

                    try:

                        status_message.info(
                            f"[{index}/{total_files}] "
                            f"OCR Processing: {filename}"
                        )

                        extracted = extract_pdf(
                            pdf_bytes,
                            converter
                        )

                        status_message.info(
                            f"[{index}/{total_files}] "
                            f"Classifying: {filename}"
                        )

                        classification = classify_document(
                            extracted["segments"],
                            profiles,
                            weights,
                            fuzzy_threshold,
                            enable_fuzzy,
                            minimum_score,
                            int(minimum_matches),
                            minimum_margin
                        )

                        # Mark incomplete OCR for review.

                        if extracted["status"] == "partial_success":

                            classification["status"] = (
                                "Review Required"
                            )

                            classification["reasons"].append(
                                "Incomplete OCR conversion"
                            )

                        if extracted["warnings"]:

                            classification["status"] = (
                                "Review Required"
                            )

                            classification["reasons"].append(
                                "Table extraction warnings"
                            )

                        if classification["status"] == "Classified":

                            classified_count += 1

                        else:

                            review_count += 1

                        # Save results without clearing
                        # previously processed documents.

                        st.session_state.results.append({
                            "name": filename,
                            "pdf": pdf_bytes,
                            "hash": file_hash,
                            **extracted,
                            "classification": classification,
                            "time": (
                                time.time() - start_time
                            )
                        })

                        st.session_state.processed_files.add(
                            file_hash
                        )

                    except Exception as error:

                        failed_count += 1

                        st.session_state.results.append({
                            "name": filename,
                            "pdf": pdf_bytes,
                            "hash": file_hash,
                            "status": "failed",
                            "error": str(error),
                            "time": (
                                time.time() - start_time
                            )
                        })

                    progress.progress(
                        index / total_files
                    )

                status_message.success(
                    "Batch processing completed."
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Classified",
                    classified_count
                )

                c2.metric(
                    "Review Required",
                    review_count
                )

                c3.metric(
                    "Failed",
                    failed_count
                )

                c4.metric(
                    "Skipped",
                    skipped_count
                )

                st.info(
                    "Open the Results tab to inspect "
                    "the classification output."
                )

        # ----------------------------------
        # UPLOAD ANOTHER BATCH
        # ----------------------------------

        if st.button(
            "Upload Another Batch",
            use_container_width=True
        ):

            # Reset only PDF uploader.
            # Masterlist and results remain saved.

            st.session_state.pdf_version += 1

            st.rerun()

# ==========================================
# TAB 3: CLASSIFICATION RESULTS
# ==========================================

with result_tab:

    st.header("Classification Results")

    results = st.session_state.results

    if not results:

        st.info(
            "No classification results available."
        )

    else:

        summary_df = build_summary(results)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Documents",
            len(results)
        )

        c2.metric(
            "Classified",
            int(
                (
                    summary_df["Status"]
                    == "Classified"
                ).sum()
            )
        )

        c3.metric(
            "Review Required",
            int(
                (
                    summary_df["Status"]
                    == "Review Required"
                ).sum()
            )
        )

        c4.metric(
            "Failed",
            int(
                (
                    summary_df["Status"]
                    == "Failed"
                ).sum()
            )
        )

        # ----------------------------------
        # RESULT FILTER
        # ----------------------------------

        status_filter = st.multiselect(
            "Filter classification status",
            [
                "Classified",
                "Review Required",
                "Failed"
            ],
            default=[
                "Classified",
                "Review Required",
                "Failed"
            ]
        )

        filtered_df = summary_df[
            summary_df["Status"].isin(
                status_filter
            )
        ]

        st.caption(
            "Review Required results show suggested "
            "classifications, not confirmed entities."
        )

        st.dataframe(
            filtered_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Score (%)":
                    st.column_config.ProgressColumn(
                        "Match Score",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%"
                    ),

                "Coverage (%)":
                    st.column_config.ProgressColumn(
                        "Attribute Coverage",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%"
                    )
            }
        )

        st.download_button(
            "Download All Results (.xlsx)",
            data=create_excel(results),
            file_name="classification_results.xlsx",
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            use_container_width=True
        )

        # ----------------------------------
        # DOCUMENT INSPECTION
        # ----------------------------------

        st.divider()

        st.subheader("Document Inspection")

        selected_index = st.selectbox(
            "Select PDF document",
            options=list(
                range(len(results))
            ),
            format_func=lambda index: (
                f"{index + 1}. "
                f"{results[index]['name']}"
            )
        )

        selected = results[selected_index]

        if selected["status"] == "failed":

            st.error(
                selected.get(
                    "error",
                    "Unknown processing error"
                )
            )

        else:

            classification = (
                selected["classification"]
            )

            best = classification.get("best")

            if classification["status"] == "Classified":

                st.success(
                    "Document successfully classified."
                )

            else:

                st.warning(
                    "Review Required: "
                    + "; ".join(
                        classification["reasons"]
                    )
                )

            if best:

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Data Domain",
                    best["Data Domain"]
                )

                c2.metric(
                    "Data Group",
                    best["Data Group"]
                )

                c3.metric(
                    "Entity",
                    best["Entity"]
                )

                c4.metric(
                    "Match Score",
                    f"{best['Score']:.2f}%"
                )

                st.caption(
                    f"Matched {best['Matched']} of "
                    f"{best['Total Attributes']} attributes"
                    f" | Coverage: {best['Coverage']}%"
                    f" | Score gap: "
                    f"{classification['margin']}%"
                )

            else:

                st.warning(
                    "No suitable entity identified."
                )

            st.caption(
                f"Processing time: "
                f"{selected['time']:.2f} seconds"
                f" | Tables: "
                f"{selected['detected_tables']}"
                f" | Pictures: "
                f"{selected['detected_pictures']}"
            )

            left, right = st.columns([1, 1])

            # ------------------------------
            # ORIGINAL PDF PREVIEW
            # ------------------------------

            with left:

                st.subheader("Original PDF")

                try:

                    with pymupdf.open(
                        stream=selected["pdf"],
                        filetype="pdf"
                    ) as pdf:

                        page_number = st.number_input(
                            "Page",
                            min_value=1,
                            max_value=len(pdf),
                            value=1,
                            key=f"page_{selected_index}"
                        )

                        page = pdf[
                            int(page_number) - 1
                        ]

                        pix = page.get_pixmap(
                            matrix=pymupdf.Matrix(
                                1.5,
                                1.5
                            )
                        )

                        st.image(
                            pix.tobytes("png"),
                            use_container_width=True
                        )

                except Exception as error:

                    st.error(
                        f"PDF preview failed: {error}"
                    )

            # ------------------------------
            # ANALYSIS RESULTS
            # ------------------------------

            with right:

                st.subheader("Document Analysis")

                (
                    candidate_tab,
                    matched_tab,
                    missing_tab,
                    ocr_tab,
                    table_tab
                ) = st.tabs([
                    "Candidates",
                    "Matched",
                    "Missing",
                    "OCR",
                    "Tables"
                ])

                # --------------------------
                # ALTERNATIVE CANDIDATES
                # --------------------------

                with candidate_tab:

                    candidate_df = pd.DataFrame([
                        {
                            "Data Domain":
                                candidate["Data Domain"],

                            "Data Group":
                                candidate["Data Group"],

                            "Entity":
                                candidate["Entity"],

                            "Score":
                                candidate["Score"],

                            "Matched":
                                candidate["Matched"],

                            "Coverage":
                                candidate["Coverage"]
                        }
                        for candidate
                        in classification["candidates"]
                    ])

                    if candidate_df.empty:

                        st.warning(
                            "No matching candidates."
                        )

                    else:

                        st.dataframe(
                            candidate_df,
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "Score":
                                st.column_config.ProgressColumn(
                                    "Match Score",
                                    min_value=0,
                                    max_value=100,
                                    format="%.2f%%"
                                )
                            }
                        )

                        chart_df = candidate_df.copy()

                        chart_df["Profile"] = (
                            chart_df["Data Group"]
                            + " / "
                            + chart_df["Entity"]
                        )

                        st.bar_chart(
                            chart_df.set_index(
                                "Profile"
                            )["Score"]
                        )

                # --------------------------
                # MATCHED ATTRIBUTES
                # --------------------------

                with matched_tab:

                    if (
                        best
                        and best["matched_attributes"]
                    ):

                        st.dataframe(
                            pd.DataFrame(
                                best["matched_attributes"]
                            ),
                            hide_index=True,
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "No matched attributes."
                        )

                # --------------------------
                # MISSING ATTRIBUTES
                # --------------------------

                with missing_tab:

                    st.caption(
                        "Attributes not detected in "
                        "the extracted PDF content."
                    )

                    if best:

                        st.dataframe(
                            pd.DataFrame({
                                "Attribute":
                                best["missing_attributes"]
                            }),
                            hide_index=True,
                            use_container_width=True
                        )

                # --------------------------
                # OCR OUTPUT
                # --------------------------

                with ocr_tab:

                    filename = Path(
                        selected["name"]
                    ).stem

                    c1, c2 = st.columns(2)

                    with c1:

                        st.download_button(
                            "Download Markdown",
                            selected["markdown"],
                            file_name=f"{filename}.md",
                            mime="text/markdown"
                        )

                    with c2:

                        st.download_button(
                            "Download JSON",
                            selected["json"],
                            file_name=f"{filename}.json",
                            mime="application/json"
                        )

                    with st.container(height=650):

                        st.markdown(
                            selected["markdown"]
                        )

                # --------------------------
                # EXTRACTED TABLES
                # --------------------------

                with table_tab:

                    for warning in selected["warnings"]:

                        st.warning(warning)

                    if not selected["tables"]:

                        st.warning(
                            "No tables extracted."
                        )

                    for index, table in enumerate(
                        selected["tables"]
                    ):

                        with st.expander(
                            table["name"],
                            expanded=(index == 0)
                        ):

                            st.dataframe(
                                table["data"],
                                use_container_width=True
                            )

                            st.download_button(
                                "Download CSV",
                                table["data"].to_csv(
                                    index=False
                                ).encode("utf-8-sig"),
                                file_name=(
                                    f"{Path(selected['name']).stem}"
                                    f"_table_{index + 1}.csv"
                                ),
                                mime="text/csv",
                                key=(
                                    f"csv_"
                                    f"{selected_index}_"
                                    f"{index}"
                                )
                            )
