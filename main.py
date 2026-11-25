
import streamlit as st
import pymupdf
import pandas as pd
import tempfile
import json
import time

from pathlib import Path

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
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Docling OCR",
    page_icon="📄",
    layout="wide"
)

st.title("Docling Document Intelligence")
st.caption("Engineering Datasheet OCR & Table Extraction")

# ==========================================
# SESSION STATE
# ==========================================

if "results" not in st.session_state:
    st.session_state.results = {}

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("OCR Configuration")

    device = st.selectbox(
        "Processing device",
        ["CPU", "CUDA"]
    )

    num_threads = st.slider(
        "CPU threads",
        min_value=1,
        max_value=16,
        value=4
    )

    table_mode = st.selectbox(
        "Table recognition mode",
        ["ACCURATE", "FAST"]
    )

    enable_ocr = st.checkbox(
        "Enable OCR",
        value=True
    )

    full_page_ocr = st.checkbox(
        "Force full-page OCR",
        value=False
    )

    enable_tables = st.checkbox(
        "Extract table structure",
        value=True
    )

    cell_matching = st.checkbox(
        "Enable cell matching",
        value=True
    )

    st.divider()

    if st.button(
        "Clear results",
        use_container_width=True
    ):
        st.session_state.results = {}
        st.rerun()

# ==========================================
# CACHE DOCLING MODEL
# ==========================================

@st.cache_resource(show_spinner=False)
def load_converter(
    device,
    num_threads,
    table_mode,
    enable_ocr,
    full_page_ocr,
    enable_tables,
    cell_matching
):

    options = PdfPipelineOptions()

    options.do_ocr = enable_ocr

    options.do_table_structure = enable_tables

    options.ocr_options = RapidOcrOptions(
        force_full_page_ocr=full_page_ocr,
        backend="onnxruntime"
    )

    options.table_structure_options.mode = (
        TableFormerMode.ACCURATE
        if table_mode == "ACCURATE"
        else TableFormerMode.FAST
    )

    options.table_structure_options.do_cell_matching = (
        cell_matching
    )

    options.accelerator_options = AcceleratorOptions(
        num_threads=num_threads,
        device=(
            AcceleratorDevice.CPU
            if device == "CPU"
            else AcceleratorDevice.CUDA
        )
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options
            )
        }
    )

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_files = st.file_uploader(
    "Upload engineering datasheets",
    type=["pdf"],
    accept_multiple_files=True
)

# ==========================================
# PROCESS DOCUMENTS
# ==========================================

if uploaded_files:

    st.info(f"{len(uploaded_files)} PDF file(s) uploaded.")

    if st.button(
        "Start OCR Processing",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Initializing Docling..."):

            converter = load_converter(
                device,
                num_threads,
                table_mode,
                enable_ocr,
                full_page_ocr,
                enable_tables,
                cell_matching
            )

        progress = st.progress(0)
        status = st.empty()

        st.session_state.results = {}

        for index, uploaded_file in enumerate(uploaded_files):

            name = uploaded_file.name
            pdf_bytes = uploaded_file.getvalue()

            status.info(f"Processing: {name}")

            start_time = time.time()

            try:

                with tempfile.TemporaryDirectory() as tmp:

                    pdf_path = Path(tmp) / "document.pdf"

                    pdf_path.write_bytes(pdf_bytes)

                    result = converter.convert(
                        pdf_path,
                        raises_on_error=False
                    )

                    conversion_status = result.status.value

                    if conversion_status not in (
                        "success",
                        "partial_success"
                    ):
                        raise RuntimeError(
                            f"Conversion status: "
                            f"{conversion_status}"
                        )

                    document = result.document

                    markdown = document.export_to_markdown()

                    json_data = json.dumps(
                        document.export_to_dict(),
                        indent=2,
                        ensure_ascii=False
                    )

                    tables = []
                    warnings = []

                    for table_index, table in enumerate(
                        document.tables,
                        start=1
                    ):

                        try:

                            df = table.export_to_dataframe(
                                doc=document
                            )

                            tables.append({
                                "name": f"Table {table_index}",
                                "data": df,
                                "rows": len(df),
                                "columns": len(df.columns)
                            })

                        except Exception as e:

                            warnings.append(
                                f"Table {table_index}: {e}"
                            )

                    elapsed = time.time() - start_time

                    st.session_state.results[index] = {
                        "name": name,
                        "pdf": pdf_bytes,
                        "status": conversion_status,
                        "markdown": markdown,
                        "json": json_data,
                        "tables": tables,
                        "detected_tables": len(document.tables),
                        "pictures": len(document.pictures),
                        "time": elapsed,
                        "warnings": warnings
                    }

            except Exception as e:

                st.session_state.results[index] = {
                    "name": name,
                    "pdf": pdf_bytes,
                    "status": "failed",
                    "error": str(e),
                    "time": time.time() - start_time
                }

            progress.progress(
                (index + 1) / len(uploaded_files)
            )

        status.success("Processing completed.")

# ==========================================
# DISPLAY RESULTS
# ==========================================

results = st.session_state.results

if results:

    st.divider()
    st.header("Processing Results")

    successful = sum(
        r["status"] == "success"
        for r in results.values()
    )

    partial = sum(
        r["status"] == "partial_success"
        for r in results.values()
    )

    failed = sum(
        r["status"] == "failed"
        for r in results.values()
    )

    total_tables = sum(
        len(r.get("tables", []))
        for r in results.values()
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Documents", len(results))
    c2.metric("Successful", successful)
    c3.metric("Partial", partial)
    c4.metric("Failed", failed)
    c5.metric("Tables exported", total_tables)

    # ======================================
    # SELECT DOCUMENT
    # ======================================

    selected = st.selectbox(
        "Select datasheet",
        options=list(results.keys()),
        format_func=lambda i: results[i]["name"]
    )

    data = results[selected]

    if data["status"] == "failed":

        st.error(data["error"])

    else:

        if data["status"] == "partial_success":
            st.warning(
                "Document partially processed. "
                "Some content may be missing."
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Processing time",
            f"{data['time']:.2f}s"
        )

        c2.metric(
            "Detected tables",
            data["detected_tables"]
        )

        c3.metric(
            "Detected pictures",
            data["pictures"]
        )

        left, right = st.columns([1, 1])

        # ==================================
        # ORIGINAL PDF PREVIEW
        # ==================================

        with left:

            st.subheader("Original Datasheet")

            with pymupdf.open(
                stream=data["pdf"],
                filetype="pdf"
            ) as pdf:

                page_number = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=len(pdf),
                    value=1,
                    key=f"page_{selected}"
                )

                page = pdf[int(page_number) - 1]

                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(1.5, 1.5)
                )

                st.image(
                    pix.tobytes("png"),
                    use_container_width=True
                )

        # ==================================
        # EXTRACTION PREVIEW
        # ==================================

        with right:

            st.subheader("Extracted Content")

            tab1, tab2, tab3, tab4 = st.tabs([
                "Markdown",
                "Tables",
                "JSON",
                "Raw Markdown"
            ])

            filename = Path(data["name"]).stem

            with tab1:

                st.download_button(
                    "Download Markdown",
                    data["markdown"],
                    file_name=f"{filename}.md",
                    mime="text/markdown"
                )

                with st.container(height=700):

                    st.markdown(data["markdown"])

            with tab2:

                if data["warnings"]:

                    for warning in data["warnings"]:
                        st.warning(warning)

                if not data["tables"]:

                    st.warning("No tables exported.")

                else:

                    for index, table in enumerate(
                        data["tables"]
                    ):

                        with st.expander(
                            f"{table['name']} | "
                            f"{table['rows']} rows × "
                            f"{table['columns']} columns",
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
                                    f"{filename}_"
                                    f"table_{index+1}.csv"
                                ),
                                mime="text/csv",
                                key=f"csv_{selected}_{index}"
                            )

            with tab3:

                st.download_button(
                    "Download JSON",
                    data["json"],
                    file_name=f"{filename}.json",
                    mime="application/json"
                )

                st.json(
                    json.loads(data["json"]),
                    expanded=False
                )

            with tab4:

                st.code(
                    data["markdown"],
                    language="markdown"
                )