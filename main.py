
import io
import time
import hashlib
import json

from pathlib import Path

import streamlit as st
import pandas as pd
import pymupdf

from classifier import (
    normalize,
    build_masterlist,
    classify_document
)

from extractor import (
    load_converter,
    extract_pdf,
    extract_parameters
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Engineering Document Intelligence",
    page_icon="📄",
    layout="wide"
)

st.title("Engineering Document Intelligence")

st.caption(
    "Docling OCR | Document Classification | "
    "Attribute and Parameter Extraction"
)

# ==========================================
# SESSION STATE
# ==========================================

defaults = {
    "profiles": {},
    "weights": {},
    "master_filename": "",
    "master_loaded": False,
    "master_version": 0,
    "pdf_version": 0,
    "results": [],
    "processed_files": set()
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def file_hash(content):

    return hashlib.sha256(
        content
    ).hexdigest()

def get_summary(results):

    rows = []

    for result in results:

        classification = result.get(
            "classification", {}
        )

        best = classification.get("best") or {}

        parameters = result.get(
            "parameters", []
        )

        expected = {
            p["Attribute"]
            for p in parameters
        }

        extracted = {
            p["Attribute"]
            for p in parameters
            if p["Status"] == "EXTRACTED"
        }

        review = {
            p["Attribute"]
            for p in parameters
            if p["Status"] in {
                "AMBIGUOUS",
                "CONFLICT",
                "VALUE_NOT_RESOLVED"
            }
            or p.get("Validation")
        }

        # Exclude attributes with unresolved
        # conflicting observations.

        extracted -= review

        rows.append({
            "File Name": result["name"],
            "Pages": result.get("page_count", 0),

            "Data Domain":
                best.get("Data Domain", ""),

            "Data Group":
                best.get("Data Group", ""),

            "Entity":
                best.get("Entity", ""),

            "Match Score (%)":
                best.get("Score", 0),

            "Expected Attributes":
                len(expected),

            "Extracted Attributes":
                len(extracted),

            "Extraction Coverage (%)": round(
                100 * len(extracted) / len(expected)
                if expected
                else 0,
                2
            ),

            "Attributes for Review":
                len(review),

            "Classification Status":
                classification.get(
                    "status", "Failed"
                ),

            "OCR Status":
                result.get("status", ""),

            "Time (s)": round(
                result.get("time", 0),
                2
            ),

            "Error":
                result.get("error", "")
        })

    return pd.DataFrame(rows)

# ==========================================
# EXCEL EXPORT
# ==========================================

def export_excel(results):

    summary = get_summary(results)

    all_parameters = []

    missing = []

    conflicts = []

    candidates = []

    logs = []

    for result in results:

        classification = result.get(
            "classification", {}
        )

        best = classification.get("best") or {}

        metadata = {
            "File Name": result["name"],

            "Data Domain":
                best.get("Data Domain", ""),

            "Data Group":
                best.get("Data Group", ""),

            "Entity":
                best.get("Entity", ""),

            "Classification Status":
                classification.get(
                    "status", "Failed"
                )
        }

        for parameter in result.get(
            "parameters", []
        ):

            record = {
                **metadata,
                **parameter
            }

            all_parameters.append(record)

            if parameter["Status"] in {
                "NOT_DETECTED",
                "EMPTY",
                "UNSPECIFIED",
                "VALUE_NOT_RESOLVED"
            }:

                missing.append(record)

            if parameter["Status"] in {
                "CONFLICT",
                "AMBIGUOUS"
            } or parameter.get("Validation"):

                conflicts.append(record)

        for candidate in classification.get(
            "candidates", []
        ):

            candidates.append({
                "File Name": result["name"],

                "Data Domain":
                    candidate["Data Domain"],

                "Data Group":
                    candidate["Data Group"],

                "Entity":
                    candidate["Entity"],

                "Score":
                    candidate["Score"],

                "Matched Attributes":
                    candidate["Matched"],

                "Coverage":
                    candidate["Coverage"]
            })

        logs.append({
            "File Name": result["name"],

            "Pages":
                result.get("page_count", 0),

            "OCR Status":
                result.get("status", ""),

            "Classification Status":
                classification.get(
                    "status", "Failed"
                ),

            "Review Reasons": "; ".join(
                classification.get(
                    "reasons", []
                )
            ),

            "Warnings": "; ".join(
                result.get("warnings", [])
            ),

            "Time (s)": round(
                result.get("time", 0),
                2
            ),

            "Error":
                result.get("error", "")
        })

    buffer = io.BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:

        sheets = {
            "Classification": summary,

            "Extracted Parameters":
                pd.DataFrame(all_parameters),

            "Missing Attributes":
                pd.DataFrame(missing),

            "Conflicting Values":
                pd.DataFrame(conflicts),

            "Classification Candidates":
                pd.DataFrame(candidates),

            "Processing Log":
                pd.DataFrame(logs)
        }

        for name, df in sheets.items():

            df.to_excel(
                writer,
                sheet_name=name,
                index=False
            )

        # Treat uploaded strings as text
        # rather than executable Excel formulas.

        for worksheet in writer.book.worksheets:

            worksheet.freeze_panes = "A2"

            if worksheet.max_row > 1:

                worksheet.auto_filter.ref = (
                    worksheet.dimensions
                )

            for row in worksheet.iter_rows():

                for cell in row:

                    if cell.data_type == "f":

                        cell.data_type = "s"

            for column in worksheet.columns:

                letter = column[0].column_letter

                max_length = max(
                    len(str(cell.value or ""))
                    for cell in column
                )

                worksheet.column_dimensions[
                    letter
                ].width = min(
                    max(max_length + 2, 12),
                    60
                )

    return buffer.getvalue()

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("OCR Configuration")

    table_mode = st.selectbox(
        "Table recognition",
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
        "Fuzzy similarity threshold",
        70,
        100,
        90
    )

    minimum_score = st.slider(
        "Minimum classification score",
        0,
        100,
        45
    )

    minimum_matches = st.number_input(
        "Minimum matched attributes",
        min_value=1,
        max_value=100,
        value=3
    )

    minimum_margin = st.slider(
        "Minimum classification score gap",
        0,
        30,
        5
    )

    st.divider()

    st.header("Parameter Extraction")

    label_threshold = st.slider(
        "Attribute label similarity",
        80,
        100,
        92
    )

    skip_duplicates = st.checkbox(
        "Skip previously processed PDFs",
        value=True
    )

    st.divider()

    st.header("Session")

    if st.session_state.master_loaded:

        st.success("Masterlist loaded")

        st.caption(
            st.session_state.master_filename
        )

    else:

        st.warning("Masterlist not configured")

    st.metric(
        "Processed documents",
        len(st.session_state.results)
    )

    if st.button(
        "Clear Results",
        use_container_width=True
    ):

        st.session_state.results = []

        st.session_state.processed_files = set()

        st.session_state.pdf_version += 1

        st.rerun()

# ==========================================
# APPLICATION TABS
# ==========================================

(
    master_tab,
    process_tab,
    results_tab,
    parameters_tab,
    preview_tab
) = st.tabs([
    "Masterlist",
    "PDF Processing",
    "Classification",
    "Extracted Parameters",
    "OCR Preview"
])

# ==========================================
# TAB 1: MASTERLIST
# ==========================================

with master_tab:

    st.header("Classification Masterlist")

    if st.session_state.master_loaded:

        profiles = st.session_state.profiles

        st.success(
            "Masterlist saved. You can continue "
            "uploading PDFs without uploading "
            "the Excel file again."
        )

        st.caption(
            f"Active masterlist: "
            f"{st.session_state.master_filename}"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Data Domains",
            len({
                key[0]
                for key in profiles
            })
        )

        c2.metric(
            "Data Groups",
            len({
                (key[0], key[1])
                for key in profiles
            })
        )

        c3.metric(
            "Entity Profiles",
            len(profiles)
        )

        c4.metric(
            "Attributes",
            sum(
                len(attributes)
                for attributes in profiles.values()
            )
        )

        profile_rows = []

        for key, attributes in profiles.items():

            profile_rows.append({
                "Data Domain": key[0],
                "Data Group": key[1],
                "Entity": key[2],
                "Attributes": len(attributes)
            })

        st.dataframe(
            pd.DataFrame(profile_rows),
            hide_index=True,
            use_container_width=True
        )

        selected_profile = st.selectbox(
            "Inspect entity attributes",
            options=list(profiles.keys()),
            format_func=lambda key: " / ".join(key)
        )

        attribute_rows = []

        for info in profiles[
            selected_profile
        ].values():

            attribute_rows.append({
                "Attribute": info["attribute"],

                "Aliases": "; ".join(
                    sorted(info["aliases"])
                ),

                "Expected UoM":
                    info["expected_uom"]
            })

        st.dataframe(
            pd.DataFrame(attribute_rows),
            hide_index=True,
            use_container_width=True
        )

        st.warning(
            "Replacing the masterlist will clear "
            "previous classification results."
        )

        if st.button("Replace Masterlist"):

            st.session_state.profiles = {}

            st.session_state.weights = {}

            st.session_state.master_filename = ""

            st.session_state.master_loaded = False

            st.session_state.results = []

            st.session_state.processed_files = set()

            st.session_state.master_version += 1

            st.session_state.pdf_version += 1

            st.rerun()

    else:

        master_file = st.file_uploader(
            "Upload Excel Masterlist",
            type=["xlsx"],
            key=(
                f"master_"
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
                    "Worksheet",
                    workbook.sheet_names
                )

                header_row = st.number_input(
                    "Header row",
                    min_value=1,
                    value=1
                )

                df = pd.read_excel(
                    workbook,
                    sheet_name=sheet,
                    header=int(header_row) - 1,
                    dtype=str,
                    keep_default_na=False
                )

                df.columns = [
                    str(column).strip()
                    for column in df.columns
                ]

                columns = list(df.columns)

                with st.expander(
                    "Preview Excel masterlist",
                    expanded=True
                ):

                    st.dataframe(
                        df.head(100),
                        use_container_width=True
                    )

                if len(columns) < 4:

                    st.error(
                        "At least four columns "
                        "are required."
                    )

                elif len(set(columns)) != len(columns):

                    st.error(
                        "Duplicate column names found."
                    )

                else:

                    def default_column(
                        names,
                        fallback
                    ):

                        targets = {
                            normalize(name)
                            for name in names
                        }

                        for index, column in enumerate(
                            columns
                        ):

                            if normalize(column) in targets:

                                return index

                        return min(
                            fallback,
                            len(columns) - 1
                        )

                    st.subheader("Map Excel Columns")

                    c1, c2 = st.columns(2)

                    with c1:

                        domain_col = st.selectbox(
                            "Data Domain",
                            columns,
                            index=default_column(
                                [
                                    "Data Domain",
                                    "Domain"
                                ],
                                0
                            )
                        )

                        group_col = st.selectbox(
                            "Data Group",
                            columns,
                            index=default_column(
                                [
                                    "Data Group",
                                    "Group"
                                ],
                                1
                            )
                        )

                        entity_col = st.selectbox(
                            "Entity",
                            columns,
                            index=default_column(
                                [
                                    "Entity",
                                    "Entity Name"
                                ],
                                2
                            )
                        )

                    with c2:

                        attribute_col = st.selectbox(
                            "Attribute",
                            columns,
                            index=default_column(
                                [
                                    "Attribute",
                                    "Attribute Name",
                                    "Parameter"
                                ],
                                3
                            )
                        )

                        alias_col = st.selectbox(
                            "Attribute Alias (Optional)",
                            [None] + columns,
                            format_func=lambda x: (
                                "Not Available"
                                if x is None
                                else x
                            )
                        )

                        uom_col = st.selectbox(
                            "Expected UoM (Optional)",
                            [None] + columns,
                            format_func=lambda x: (
                                "Not Available"
                                if x is None
                                else x
                            )
                        )

                    forward_fill = st.checkbox(
                        "Fill merged or blank "
                        "hierarchy cells",
                        value=True
                    )

                    if st.button(
                        "Save Masterlist",
                        type="primary",
                        use_container_width=True
                    ):

                        mapped = [
                            domain_col,
                            group_col,
                            entity_col,
                            attribute_col
                        ]

                        if alias_col:

                            mapped.append(alias_col)

                        if uom_col:

                            mapped.append(uom_col)

                        if len(set(mapped)) != len(mapped):

                            st.error(
                                "Select different "
                                "columns for each field."
                            )

                        else:

                            profiles, weights = (
                                build_masterlist(
                                    df=df,
                                    domain_col=domain_col,
                                    group_col=group_col,
                                    entity_col=entity_col,
                                    attribute_col=attribute_col,
                                    alias_col=alias_col,
                                    uom_col=uom_col,
                                    forward_fill=forward_fill
                                )
                            )

                            st.session_state.profiles = (
                                profiles
                            )

                            st.session_state.weights = (
                                weights
                            )

                            st.session_state.master_filename = (
                                master_file.name
                            )

                            st.session_state.master_loaded = True

                            st.rerun()

            except Exception as error:

                st.error(
                    f"Masterlist error: {error}"
                )

# ==========================================
# TAB 2: PROCESS PDF
# ==========================================

with process_tab:

    st.header("PDF Document Processing")

    if not st.session_state.master_loaded:

        st.warning(
            "Upload and save the Excel "
            "masterlist first."
        )

    else:

        st.success(
            "Using saved masterlist: "
            + st.session_state.master_filename
        )

        st.caption(
            "All pages within each PDF are "
            "processed as one complete document."
        )

        uploaded_files = st.file_uploader(
            "Upload PDF Datasheets",
            type=["pdf"],
            accept_multiple_files=True,
            key=(
                f"pdf_"
                f"{st.session_state.pdf_version}"
            )
        )

        if uploaded_files:

            st.info(
                f"{len(uploaded_files)} PDF(s) uploaded."
            )

        if st.button(
            "Extract, Classify & Extract Parameters",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files
        ):

            try:

                with st.spinner(
                    "Initializing Docling models..."
                ):

                    converter = load_converter(
                        full_page_ocr,
                        table_mode
                    )

            except Exception as error:

                st.error(
                    f"Unable to load Docling: {error}"
                )

                converter = None

            if converter is not None:

                progress = st.progress(0)

                message = st.empty()

                total_files = len(uploaded_files)

                batch_success = 0
                batch_failed = 0
                batch_skipped = 0

                for index, uploaded in enumerate(
                    uploaded_files,
                    start=1
                ):

                    pdf_bytes = uploaded.getvalue()

                    digest = file_hash(pdf_bytes)

                    filename = uploaded.name

                    if (
                        skip_duplicates
                        and digest in
                        st.session_state.processed_files
                    ):

                        batch_skipped += 1

                        progress.progress(
                            index / total_files
                        )

                        continue

                    started = time.time()

                    page_count = 0

                    try:

                        # ------------------
                        # Count PDF pages
                        # ------------------

                        with pymupdf.open(
                            stream=pdf_bytes,
                            filetype="pdf"
                        ) as pdf:

                            page_count = len(pdf)

                        message.info(
                            f"[{index}/{total_files}] "
                            f"OCR: {filename} "
                            f"({page_count} pages)"
                        )

                        # ------------------
                        # OCR entire PDF
                        # ------------------

                        extracted = extract_pdf(
                            pdf_bytes,
                            converter
                        )

                        message.info(
                            f"[{index}/{total_files}] "
                            f"Classifying {filename}"
                        )

                        # ------------------
                        # Classify entire PDF
                        # ------------------

                        classification = classify_document(
                            segments=extracted["segments"],

                            profiles=(
                                st.session_state.profiles
                            ),

                            weights=(
                                st.session_state.weights
                            ),

                            fuzzy_threshold=(
                                fuzzy_threshold
                            ),

                            enable_fuzzy=enable_fuzzy,

                            minimum_score=minimum_score,

                            minimum_matches=int(
                                minimum_matches
                            ),

                            minimum_margin=minimum_margin
                        )

                        if extracted["status"] != "success":

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
                                "OCR or table warnings"
                            )

                        # ------------------
                        # Retrieve expected attributes
                        # and extract values
                        # ------------------

                        best = classification.get(
                            "best"
                        )

                        parameters = []

                        if best:

                            message.info(
                                f"[{index}/{total_files}] "
                                f"Extracting parameters "
                                f"for {best['Entity']}"
                            )

                            entity_profile = (
                                best["profile"]
                            )

                            expected_attributes = (
                                st.session_state.profiles[
                                    entity_profile
                                ]
                            )

                            parameters = extract_parameters(
                                extracted=extracted,

                                attributes=(
                                    expected_attributes
                                ),

                                label_threshold=(
                                    label_threshold
                                )
                            )

                        # ------------------
                        # Save complete document
                        # ------------------

                        st.session_state.results.append({
                            "name": filename,
                            "pdf": pdf_bytes,
                            "hash": digest,
                            "page_count": page_count,
                            **extracted,
                            "classification": classification,
                            "parameters": parameters,
                            "time": time.time() - started
                        })

                        st.session_state.processed_files.add(
                            digest
                        )

                        batch_success += 1

                    except Exception as error:

                        batch_failed += 1

                        st.session_state.results.append({
                            "name": filename,
                            "pdf": pdf_bytes,
                            "hash": digest,
                            "page_count": page_count,
                            "status": "failed",
                            "error": str(error),
                            "parameters": [],
                            "time": time.time() - started
                        })

                    progress.progress(
                        index / total_files
                    )

                message.success(
                    "Batch processing completed."
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Processed",
                    batch_success
                )

                c2.metric(
                    "Failed",
                    batch_failed
                )

                c3.metric(
                    "Skipped",
                    batch_skipped
                )

        if st.button("Upload Another Batch"):

            st.session_state.pdf_version += 1

            st.rerun()

# ==========================================
# TAB 3: CLASSIFICATION RESULTS
# ==========================================

with results_tab:

    st.header("Document Classification")

    results = st.session_state.results

    if not results:

        st.info("No processed documents.")

    else:

        summary = get_summary(results)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "PDF Documents",
            len(summary)
        )

        c2.metric(
            "Total Pages",
            int(summary["Pages"].sum())
        )

        c3.metric(
            "Classified",
            int(
                (
                    summary["Classification Status"]
                    == "Classified"
                ).sum()
            )
        )

        c4.metric(
            "Review Required",
            int(
                (
                    summary["Classification Status"]
                    == "Review Required"
                ).sum()
            )
        )

        st.caption(
            "Each row represents one complete PDF, "
            "not one page."
        )

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Match Score (%)":
                    st.column_config.ProgressColumn(
                        "Match Score",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%"
                    ),

                "Extraction Coverage (%)":
                    st.column_config.ProgressColumn(
                        "Extraction Coverage",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%"
                    )
            }
        )

        st.download_button(
            "Download Complete Excel Report",
            data=export_excel(results),
            file_name="engineering_extraction.xlsx",
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            use_container_width=True
        )

        st.divider()

        selected_index = st.selectbox(
            "Inspect classification",
            options=range(len(results)),
            format_func=lambda i: (
                f"{i + 1}. {results[i]['name']}"
            ),
            key="classification_select"
        )

        selected = results[selected_index]

        classification = selected.get(
            "classification", {}
        )

        if selected["status"] == "failed":

            st.error(selected["error"])

        else:

            best = classification.get("best")

            if best:

                c1, c2, c3 = st.columns(3)

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

                st.caption(
                    f"Matched {best['Matched']} "
                    f"of {best['Total Attributes']} "
                    f"attributes across "
                    f"{selected['page_count']} pages."
                )

            for reason in classification.get(
                "reasons", []
            ):

                st.warning(reason)

            candidate_rows = [
                {
                    "Data Domain": c["Data Domain"],
                    "Data Group": c["Data Group"],
                    "Entity": c["Entity"],
                    "Score": c["Score"],
                    "Matched": c["Matched"],
                    "Coverage": c["Coverage"]
                }
                for c in classification.get(
                    "candidates", []
                )
            ]

            st.subheader("Alternative Candidates")

            st.dataframe(
                pd.DataFrame(candidate_rows),
                hide_index=True,
                use_container_width=True
            )

            if best:

                with st.expander(
                    "Matched classification attributes"
                ):

                    st.dataframe(
                        pd.DataFrame(
                            best["matched_attributes"]
                        ),
                        hide_index=True,
                        use_container_width=True
                    )

# ==========================================
# TAB 4: EXTRACTED PARAMETERS
# ==========================================

with parameters_tab:

    st.header("Extracted Attributes & Parameters")

    results = st.session_state.results

    if not results:

        st.info("No extraction results available.")

    else:

        selected_index = st.selectbox(
            "Select PDF document",
            options=range(len(results)),
            format_func=lambda i: (
                f"{i + 1}. {results[i]['name']}"
            ),
            key="parameter_select"
        )

        selected = results[selected_index]

        classification = selected.get(
            "classification", {}
        )

        best = classification.get("best") or {}

        parameters = selected.get(
            "parameters", []
        )

        if not parameters:

            st.warning(
                "No parameters available. "
                "The document may be unclassified "
                "or OCR processing may have failed."
            )

        else:

            st.caption(
                f"{selected['name']} | "
                f"{selected['page_count']} pages | "
                f"{best.get('Entity', '')}"
            )

            if classification.get("status") != "Classified":

                st.warning(
                    "Parameters are based on a "
                    "suggested entity that requires review."
                )

            df = pd.DataFrame(parameters)

            total_attributes = df[
                "Attribute"
            ].nunique()

            extracted_attributes = set(
                df.loc[
                    df["Status"] == "EXTRACTED",
                    "Attribute"
                ]
            )

            review_attributes = set(
                df.loc[
                    df["Status"].isin([
                        "AMBIGUOUS",
                        "CONFLICT",
                        "VALUE_NOT_RESOLVED"
                    ])
                    | df["Validation"].ne(""),
                    "Attribute"
                ]
            )

            extracted_attributes -= review_attributes

            missing_attributes = set(
                df.loc[
                    df["Status"] == "NOT_DETECTED",
                    "Attribute"
                ]
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Expected Attributes",
                total_attributes
            )

            c2.metric(
                "Extracted",
                len(extracted_attributes)
            )

            c3.metric(
                "Not Detected",
                len(missing_attributes)
            )

            c4.metric(
                "For Review",
                len(review_attributes)
            )

            status_options = sorted(
                df["Status"].unique().tolist()
            )

            c1, c2 = st.columns([1, 2])

            with c1:

                status_filter = st.multiselect(
                    "Filter extraction status",
                    status_options,
                    default=status_options
                )

            with c2:

                search = st.text_input(
                    "Search attribute or value"
                )

            filtered = df[
                df["Status"].isin(status_filter)
            ]

            if search:

                filtered = filtered[
                    filtered["Attribute"].str.contains(
                        search,
                        case=False,
                        regex=False,
                        na=False
                    )
                    |
                    filtered["Value"].str.contains(
                        search,
                        case=False,
                        regex=False,
                        na=False
                    )
                ]

            visible_columns = [
                "Attribute",
                "Value",
                "UoM",
                "Expected UoM",
                "Condition",
                "Page",
                "Status",
                "Validation"
            ]

            st.dataframe(
                filtered[visible_columns],
                hide_index=True,
                use_container_width=True,
                height=500
            )

            # ------------------------------
            # Evidence inspection
            # ------------------------------

            with st.expander(
                "View OCR Source Evidence"
            ):

                st.dataframe(
                    filtered[[
                        "Attribute",
                        "Value",
                        "Page",
                        "Method",
                        "Label Similarity",
                        "Source Location",
                        "Source"
                    ]],
                    hide_index=True,
                    use_container_width=True
                )

            # ------------------------------
            # Download individual parameters
            # ------------------------------

            filename = Path(
                selected["name"]
            ).stem

            csv_df = df.copy()

            # Protect spreadsheet CSV consumers
            # from formula injection.

            for column in csv_df.columns:

                if csv_df[column].dtype == "object":

                    csv_df[column] = csv_df[
                        column
                    ].map(
                        lambda value:
                            "'" + value
                            if isinstance(value, str)
                            and value.lstrip().startswith(
                                ("=", "+", "-", "@")
                            )
                            else value
                    )

            c1, c2 = st.columns(2)

            with c1:

                st.download_button(
                    "Download Parameters CSV",
                    data=csv_df.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name=(
                        f"{filename}_parameters.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True
                )

            with c2:

                st.download_button(
                    "Download Parameters JSON",
                    data=json.dumps(
                        parameters,
                        ensure_ascii=False,
                        indent=2
                    ),
                    file_name=(
                        f"{filename}_parameters.json"
                    ),
                    mime="application/json",
                    use_container_width=True
                )

# ==========================================
# TAB 5: OCR PREVIEW
# ==========================================

with preview_tab:

    st.header("Original PDF & OCR Preview")

    results = st.session_state.results

    if not results:

        st.info("No documents available.")

    else:

        selected_index = st.selectbox(
            "Select document",
            options=range(len(results)),
            format_func=lambda i: (
                f"{i + 1}. {results[i]['name']}"
            ),
            key="preview_select"
        )

        selected = results[selected_index]

        if selected["status"] == "failed":

            st.error(selected["error"])

        else:

            left, right = st.columns([1, 1])

            with left:

                st.subheader("Original PDF")

                try:

                    with pymupdf.open(
                        stream=selected["pdf"],
                        filetype="pdf"
                    ) as pdf:

                        page_number = st.number_input(
                            "Preview page",
                            min_value=1,
                            max_value=len(pdf),
                            value=1,
                            key=(
                                f"page_"
                                f"{selected_index}"
                            )
                        )

                        page = pdf[
                            int(page_number) - 1
                        ]

                        pix = page.get_pixmap(
                            matrix=pymupdf.Matrix(
                                1.5, 1.5
                            )
                        )

                        st.image(
                            pix.tobytes("png"),
                            use_container_width=True
                        )

                except Exception as error:

                    st.error(
                        f"PDF preview error: {error}"
                    )

            with right:

                st.subheader("OCR Content")

                markdown_tab, table_tab, json_tab = (
                    st.tabs([
                        "Markdown",
                        "Tables",
                        "JSON"
                    ])
                )

                filename = Path(
                    selected["name"]
                ).stem

                with markdown_tab:

                    st.download_button(
                        "Download Markdown",
                        data=selected["markdown"],
                        file_name=f"{filename}.md",
                        mime="text/markdown"
                    )

                    with st.container(height=750):

                        st.markdown(
                            selected["markdown"]
                        )

                with table_tab:

                    for warning in selected.get(
                        "warnings", []
                    ):

                        st.warning(warning)

                    tables = selected.get(
                        "tables", []
                    )

                    if not tables:

                        st.warning(
                            "No tables extracted."
                        )

                    for table_index, table in enumerate(
                        tables
                    ):

                        pages = ", ".join(
                            str(page)
                            for page in table["pages"]
                        )

                        with st.expander(
                            f"{table['name']} | "
                            f"Page {pages or 'Unknown'}"
                        ):

                            st.dataframe(
                                pd.DataFrame(
                                    table["grid"]
                                ),
                                use_container_width=True
                            )

                with json_tab:

                    st.download_button(
                        "Download Docling JSON",
                        data=selected["json"],
                        file_name=f"{filename}.json",
                        mime="application/json"
                    )

                    st.json(
                        json.loads(selected["json"]),
                        expanded=False
                    )
