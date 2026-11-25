
import streamlit as st
import pandas as pd
import pymupdf
import tempfile
import unicodedata
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
    "OCR, attribute matching and automatic "
    "datasheet classification"
)

if "results" not in st.session_state:
    st.session_state.results = []

# ==========================================
# TEXT NORMALIZATION
# ==========================================

def normalize(text):

    text = unicodedata.normalize("NFKC", str(text))

    text = text.lower()

    text = re.sub(r"[\W_]+", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ==========================================
# LOAD DOCLING
# ==========================================

@st.cache_resource(show_spinner=False)
def load_converter(full_page_ocr, table_mode):

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

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options
            )
        }
    )

# ==========================================
# BUILD CLASSIFICATION MASTERLIST
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

    data = data.replace(r"^\s*$", pd.NA, regex=True)

    if forward_fill:

        data["domain"] = data["domain"].ffill()

        data["group"] = (
            data.groupby("domain", sort=False)["group"]
            .ffill()
        )

        data["entity"] = (
            data.groupby(
                ["domain", "group"],
                sort=False
            )["entity"]
            .ffill()
        )

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

        attribute = str(row["attribute"]).strip()

        normalized = normalize(attribute)

        if not normalized:
            continue

        if normalized not in profiles[key]:

            profiles[key][normalized] = {
                "name": attribute,
                "aliases": {normalized}
            }

        # Optional synonym / alias column
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

    # Count how many entity profiles
    # contain each attribute or alias.

    frequency = Counter()

    for attributes in profiles.values():

        profile_terms = set()

        for information in attributes.values():

            profile_terms.update(
                information["aliases"]
            )

        frequency.update(profile_terms)

    total_profiles = len(profiles)

    # Rare attributes receive higher weights.

    weights = {
        term: 1 + math.log(
            (total_profiles + 1) / (count + 1)
        )
        for term, count in frequency.items()
    }

    return dict(profiles), weights

# ==========================================
# EXTRACT PDF USING DOCLING
# ==========================================

def extract_pdf(pdf_bytes, converter):

    with tempfile.TemporaryDirectory() as tmp:

        path = Path(tmp) / "document.pdf"

        path.write_bytes(pdf_bytes)

        result = converter.convert(
            path,
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

        # Extract text items directly.

        for item in document.texts:

            text = getattr(item, "text", "")

            if text:

                segments.extend(
                    str(text).splitlines()
                )

        # Extract structured table cells.

        for index, table in enumerate(
            document.tables,
            start=1
        ):

            try:

                for cell in table.data.table_cells:

                    text = getattr(cell, "text", "")

                    if text:

                        segments.append(str(text))

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

        # Include Markdown as fallback evidence.

        for line in markdown.splitlines():

            segments.extend(
                re.split(r"[|:]", line)
            )

        cleaned_segments = {
            normalize(segment)
            for segment in segments
            if normalize(segment)
        }

        return {
            "markdown": markdown,
            "json": json_data,
            "segments": cleaned_segments,
            "tables": tables,
            "status": status,
            "warnings": warnings,
            "detected_tables": len(document.tables)
        }

# ==========================================
# BUILD DOCUMENT PHRASE INDEX
# ==========================================

def build_phrase_index(segments, max_words):

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
# ATTRIBUTE MATCHING
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

    # Match every unique term only once.

    for term in sorted(all_terms):

        # Exact word-boundary phrase matching.

        if term in all_phrases:

            matches[term] = {
                "matched_text": term,
                "similarity": 100.0,
                "method": "Exact"
            }

            continue

        if not enable_fuzzy:
            continue

        # Prevent fuzzy matching of short
        # abbreviations such as IP, AC or DC.

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
# CLASSIFICATION ENGINE
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

        # More distinctive attributes first.
        ordered_attributes = sorted(
            attributes.items(),
            key=lambda item: (
                min(
                    weights.get(a, 1.0)
                    for a in item[1]["aliases"]
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
                "Matched OCR text": best["matched_text"],
                "Similarity (%)": round(
                    best["similarity"], 2
                ),
                "Method": best["method"],
                "Weight": round(weight, 3)
            })

        if total_weight == 0:
            continue

        coverage = matched_weight / total_weight

        # Combine weighted coverage with the
        # amount of supporting evidence.
        #
        # Small entity profiles cannot dominate
        # solely because of one matching attribute.

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

    if not candidates:

        return {
            "best": None,
            "candidates": [],
            "status": "Review Required",
            "reasons": ["No classification profiles"]
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

    if len(candidates) > 1 and margin < minimum_margin:

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
# EXPORT CLASSIFICATION RESULTS
# ==========================================

def create_excel(results):

    summary = []

    attributes = []

    for result in results:

        classification = result.get(
            "classification", {}
        )

        best = classification.get("best") or {}

        summary.append({
            "File Name": result["name"],
            "Data Domain": best.get("Data Domain", ""),
            "Data Group": best.get("Data Group", ""),
            "Entity": best.get("Entity", ""),
            "Match Score": best.get("Score", 0),
            "Coverage (%)": best.get("Coverage", 0),
            "Matched Attributes": best.get("Matched", 0),
            "Total Attributes": best.get(
                "Total Attributes", 0
            ),
            "Status": classification.get(
                "status", "Failed"
            ),
            "Review Reason": "; ".join(
                classification.get("reasons", [])
            ),
            "OCR Status": result.get("status", ""),
            "Processing Time": round(
                result.get("time", 0), 2
            ),
            "Error": result.get("error", "")
        })

        for match in best.get(
            "matched_attributes", []
        ):

            attributes.append({
                "File Name": result["name"],
                "Entity": best.get("Entity", ""),
                **match,
                "Status": "Matched"
            })

        for missing in best.get(
            "missing_attributes", []
        ):

            attributes.append({
                "File Name": result["name"],
                "Entity": best.get("Entity", ""),
                "Attribute": missing,
                "Status": "Not detected"
            })

    buffer = io.BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:

        pd.DataFrame(summary).to_excel(
            writer,
            sheet_name="Classification",
            index=False
        )

        pd.DataFrame(attributes).to_excel(
            writer,
            sheet_name="Attribute Matching",
            index=False
        )

    return buffer.getvalue()

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================

with st.sidebar:

    st.header("OCR Settings")

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
        "Minimum score gap between top entities",
        min_value=0,
        max_value=30,
        value=5
    )

    st.divider()

    if st.button(
        "Clear results",
        use_container_width=True
    ):

        st.session_state.results = []

        st.rerun()

# ==========================================
# STEP 1: UPLOAD EXCEL MASTERLIST
# ==========================================

st.header("1. Upload Classification Masterlist")

master_file = st.file_uploader(
    "Upload Excel masterlist",
    type=["xlsx"],
    key="masterlist"
)

profiles = {}

weights = {}

if master_file:

    try:

        workbook = pd.ExcelFile(
            io.BytesIO(master_file.getvalue()),
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

        columns = list(master_df.columns)

        with st.expander("Preview Excel masterlist"):

            st.dataframe(
                master_df,
                use_container_width=True
            )

        if len(columns) < 4:

            st.error(
                "Masterlist requires at least four columns."
            )

            st.stop()

        # Automatically suggest column mappings.

        def default_column(name, fallback):

            target = normalize(name)

            for index, col in enumerate(columns):

                if normalize(col) == target:
                    return index

            return min(fallback, len(columns) - 1)

        st.subheader("Map Excel Columns")

        c1, c2 = st.columns(2)

        with c1:

            domain_col = st.selectbox(
                "Data Domain column",
                columns,
                index=default_column(
                    "Data Domain", 0
                )
            )

            group_col = st.selectbox(
                "Data Group column",
                columns,
                index=default_column(
                    "Data Group", 1
                )
            )

        with c2:

            entity_col = st.selectbox(
                "Entity column",
                columns,
                index=default_column(
                    "Entity", 2
                )
            )

            attribute_col = st.selectbox(
                "Attribute column",
                columns,
                index=default_column(
                    "Attribute", 3
                )
            )

        alias_col = st.selectbox(
            "Synonym / Alias column (optional)",
            options=[None] + columns,
            format_func=lambda x: (
                "Not available" if x is None else x
            )
        )

        forward_fill = st.checkbox(
            "Fill blank hierarchy cells "
            "(for merged Excel cells)",
            value=True
        )

        selected_columns = [
            domain_col,
            group_col,
            entity_col,
            attribute_col
        ]

        if alias_col:
            selected_columns.append(alias_col)

        if len(set(selected_columns)) != len(
            selected_columns
        ):

            st.error(
                "Please select different columns "
                "for each field."
            )

            st.stop()

        profiles, weights = build_masterlist(
            master_df,
            domain_col,
            group_col,
            entity_col,
            attribute_col,
            forward_fill,
            alias_col
        )

        if not profiles:

            st.error(
                "No valid classification profiles found."
            )

            st.stop()

        domains = {key[0] for key in profiles}

        groups = {
            (key[0], key[1])
            for key in profiles
        }

        total_attributes = sum(
            len(attributes)
            for attributes in profiles.values()
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Data Domains", len(domains))

        c2.metric("Data Groups", len(groups))

        c3.metric("Entity Profiles", len(profiles))

        c4.metric("Attributes", total_attributes)

        st.success("Masterlist loaded successfully.")

        profile_preview = []

        for key, attributes in profiles.items():

            profile_preview.append({
                "Data Domain": key[0],
                "Data Group": key[1],
                "Entity": key[2],
                "Attributes": len(attributes)
            })

        with st.expander(
            "View available classification profiles"
        ):

            st.dataframe(
                pd.DataFrame(profile_preview),
                use_container_width=True
            )

    except Exception as error:

        st.error(f"Unable to load Excel: {error}")

        st.stop()

# ==========================================
# STEP 2: UPLOAD PDF DATASHEETS
# ==========================================

st.divider()

st.header("2. Upload PDF Datasheets")

uploaded_files = st.file_uploader(
    "Upload one or multiple PDF datasheets",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_files"
)

if not master_file:

    st.info(
        "Upload an Excel masterlist before classification."
    )

if uploaded_files:

    st.caption(
        f"{len(uploaded_files)} documents uploaded."
    )

# ==========================================
# STEP 3: OCR AND CLASSIFICATION
# ==========================================

start = st.button(
    "Extract & Classify Documents",
    type="primary",
    use_container_width=True,
    disabled=not profiles or not uploaded_files
)

if start:

    st.session_state.results = []

    with st.spinner("Initializing Docling models..."):

        converter = load_converter(
            full_page_ocr,
            table_mode
        )

    progress = st.progress(0)

    message = st.empty()

    for index, uploaded_file in enumerate(
        uploaded_files,
        start=1
    ):

        start_time = time.time()

        filename = uploaded_file.name

        pdf_bytes = uploaded_file.getvalue()

        try:

            message.info(
                f"[{index}/{len(uploaded_files)}] "
                f"Performing OCR: {filename}"
            )

            extracted = extract_pdf(
                pdf_bytes,
                converter
            )

            message.info(
                f"[{index}/{len(uploaded_files)}] "
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

            if extracted["status"] == "partial_success":

                classification["status"] = (
                    "Review Required"
                )

                classification["reasons"].append(
                    "Incomplete OCR conversion"
                )

            st.session_state.results.append({
                "name": filename,
                "pdf": pdf_bytes,
                **extracted,
                "classification": classification,
                "time": time.time() - start_time
            })

        except Exception as error:

            st.session_state.results.append({
                "name": filename,
                "pdf": pdf_bytes,
                "status": "failed",
                "error": str(error),
                "time": time.time() - start_time
            })

        progress.progress(
            index / len(uploaded_files)
        )

    message.success(
        "Document extraction and classification completed."
    )

# ==========================================
# STEP 4: CLASSIFICATION SUMMARY
# ==========================================

results = st.session_state.results

if results:

    st.divider()

    st.header("3. Classification Results")

    summary = []

    for result in results:

        classification = result.get(
            "classification", {}
        )

        best = classification.get("best") or {}

        summary.append({
            "File Name": result["name"],
            "Data Domain": best.get(
                "Data Domain", "-"
            ),
            "Data Group": best.get(
                "Data Group", "-"
            ),
            "Entity": best.get("Entity", "-"),
            "Score (%)": best.get("Score", 0),
            "Matched": best.get("Matched", 0),
            "Status": classification.get(
                "status", "Failed"
            ),
            "Time (s)": round(result["time"], 2)
        })

    summary_df = pd.DataFrame(summary)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Documents", len(results))

    c2.metric(
        "Classified",
        int(
            (summary_df["Status"] == "Classified").sum()
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
            (summary_df["Status"] == "Failed").sum()
        )
    )

    st.caption(
        "Results requiring review display the suggested "
        "classification, not a confirmed classification."
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score (%)": st.column_config.ProgressColumn(
                "Match Score",
                min_value=0,
                max_value=100,
                format="%.2f%%"
            )
        }
    )

    st.download_button(
        "Download Classification Results (.xlsx)",
        data=create_excel(results),
        file_name="classification_results.xlsx",
        mime=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        )
    )

    # ======================================
    # STEP 5: DOCUMENT INSPECTION
    # ======================================

    st.divider()

    st.header("4. Document Inspection")

    selected_index = st.selectbox(
        "Select document",
        options=range(len(results)),
        format_func=lambda i: results[i]["name"]
    )

    selected = results[selected_index]

    if selected["status"] == "failed":

        st.error(selected["error"])

        st.stop()

    classification = selected["classification"]

    best = classification.get("best")

    if classification["status"] == "Review Required":

        st.warning(
            "Review Required: " +
            "; ".join(classification["reasons"])
        )

    else:

        st.success("Document successfully classified.")

    # Classification information

    if best:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Suggested Data Domain",
            best["Data Domain"]
        )

        c2.metric(
            "Suggested Data Group",
            best["Data Group"]
        )

        c3.metric(
            "Suggested Entity",
            best["Entity"]
        )

        c4.metric(
            "Match Score",
            f"{best['Score']:.2f}%"
        )

        st.caption(
            f"Matched {best['Matched']} of "
            f"{best['Total Attributes']} attributes | "
            f"Weighted coverage: {best['Coverage']}%"
        )

    left, right = st.columns([1, 1])

    # ======================================
    # PDF PREVIEW
    # ======================================

    with left:

        st.subheader("Original PDF")

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

            page = pdf[int(page_number) - 1]

            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(1.5, 1.5)
            )

            st.image(
                pix.tobytes("png"),
                use_container_width=True
            )

    # ======================================
    # CLASSIFICATION AND OCR RESULTS
    # ======================================

    with right:

        st.subheader("Analysis")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Candidates",
            "Matched",
            "Missing",
            "OCR",
            "Tables"
        ])

        with tab1:

            candidate_df = pd.DataFrame([
                {
                    "Data Domain": item["Data Domain"],
                    "Data Group": item["Data Group"],
                    "Entity": item["Entity"],
                    "Score": item["Score"],
                    "Matched": item["Matched"],
                    "Coverage": item["Coverage"]
                }
                for item in classification["candidates"]
            ])

            if not candidate_df.empty:

                st.dataframe(
                    candidate_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Score":
                        st.column_config.ProgressColumn(
                            "Score",
                            min_value=0,
                            max_value=100,
                            format="%.2f%%"
                        )
                    }
                )

                st.bar_chart(
                    candidate_df.assign(
                        Profile=(
                            candidate_df["Data Domain"]
                            + " / "
                            + candidate_df["Data Group"]
                            + " / "
                            + candidate_df["Entity"]
                        )
                    ).set_index("Profile")["Score"]
                )

        with tab2:

            if best and best["matched_attributes"]:

                st.dataframe(
                    pd.DataFrame(
                        best["matched_attributes"]
                    ),
                    hide_index=True,
                    use_container_width=True
                )

            else:

                st.warning("No matched attributes.")

        with tab3:

            st.caption(
                "These attributes were not detected "
                "in the extracted PDF content."
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

        with tab4:

            filename = Path(selected["name"]).stem

            st.download_button(
                "Download Markdown",
                selected["markdown"],
                file_name=f"{filename}.md",
                mime="text/markdown"
            )

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

        with tab5:

            for warning in selected["warnings"]:

                st.warning(warning)

            if not selected["tables"]:

                st.warning("No tables extracted.")

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
                        ),
                        file_name=f"table_{index+1}.csv",
                        mime="text/csv",
                        key=(
                            f"table_{selected_index}_{index}"
                        )
                    )
