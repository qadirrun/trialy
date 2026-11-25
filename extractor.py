
import re
import json
import tempfile

from pathlib import Path
from collections import defaultdict

import pandas as pd
import streamlit as st

from rapidfuzz import fuzz

from classifier import normalize

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
# CACHE DOCLING
# ==========================================

@st.cache_resource(
    show_spinner=False,
    max_entries=1
)
def load_converter(
    full_page_ocr=False,
    table_mode="FAST"
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

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options
            )
        }
    )

# ==========================================
# GET SOURCE PAGE NUMBERS
# ==========================================

def get_pages(item):

    pages = set()

    for provenance in getattr(item, "prov", []):

        page = getattr(
            provenance,
            "page_no",
            None
        )

        if page is not None:

            pages.add(int(page))

    return sorted(pages)

# ==========================================
# CONVERT TABLE CELLS INTO GRID
# ==========================================

def build_table_grid(table):

    data = table.data

    rows = data.num_rows

    cols = data.num_cols

    grid = [
        [""] * cols
        for _ in range(rows)
    ]

    header_rows = set()

    occupied = set()

    for cell in data.table_cells:

        text = str(
            getattr(cell, "text", "")
        ).strip()

        row = cell.start_row_offset_idx

        col = cell.start_col_offset_idx

        if not (
            0 <= row < rows
            and 0 <= col < cols
        ):

            continue

        # Preserve merged-cell anchor positions.

        grid[row][col] = text

        occupied.add((row, col))

        if getattr(cell, "column_header", False):

            header_rows.add(row)

    return grid, header_rows, occupied

# ==========================================
# EXTRACT ENTIRE PDF
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

        segments = []

        text_blocks = []

        tables = []

        warnings = []

        # ----------------------------------
        # All text from all pages
        # ----------------------------------

        for item in document.texts:

            text = str(
                getattr(item, "text", "")
            ).strip()

            if not text:

                continue

            pages = get_pages(item)

            text_blocks.append({
                "text": text,
                "pages": pages
            })

            segments.extend(
                text.splitlines()
            )

        # ----------------------------------
        # All tables from all pages
        # ----------------------------------

        for index, table in enumerate(
            document.tables,
            start=1
        ):

            try:

                grid, headers, occupied = (
                    build_table_grid(table)
                )

                for row in grid:

                    segments.extend(
                        cell for cell in row if cell
                    )

                tables.append({
                    "name": f"Table {index}",
                    "pages": get_pages(table),
                    "grid": grid,
                    "header_rows": headers,
                    "occupied": occupied
                })

            except Exception as error:

                warnings.append(
                    f"Table {index}: {error}"
                )

        # ----------------------------------
        # Markdown fallback for classification
        # ----------------------------------

        markdown = document.export_to_markdown()

        for line in markdown.splitlines():

            segments.extend(
                re.split(r"[|:]", line)
            )

        cleaned_segments = {
            normalize(segment)
            for segment in segments
            if normalize(segment)
        }

        json_data = json.dumps(
            document.export_to_dict(),
            ensure_ascii=False,
            indent=2
        )

        for error in getattr(result, "errors", []):

            warnings.append(str(error))

        return {
            "segments": cleaned_segments,
            "text_blocks": text_blocks,
            "tables": tables,
            "markdown": markdown,
            "json": json_data,
            "status": status,
            "warnings": warnings,
            "detected_tables": len(document.tables),
            "detected_pictures": len(document.pictures)
        }

# ==========================================
# UNIT OF MEASUREMENT
# ==========================================

UNITS = [
    "bar", "barg", "bara",
    "Pa", "kPa", "MPa",
    "psi", "psig", "psia",
    "mm", "cm", "m",
    "inch", "in", "ft",
    "kg", "g", "ton",
    "kg/h", "kg/s", "kg/m3", "kg/m³",
    "m3/h", "m³/h", "m3/s", "m³/s",
    "L/min", "L/s", "lpm", "gpm",
    "Nm3/h", "Nm³/h", "Sm3/h", "Sm³/h",
    "m/s", "mm/s",
    "°C", "°F", "degC", "degF",
    "rpm", "r/min",
    "W", "kW", "MW", "HP",
    "V", "kV", "A", "mA",
    "Hz", "kHz",
    "Nm", "N·m",
    "cP", "cSt",
    "mPa.s", "Pa.s",
    "%", "dB", "dBA",
    "s", "min", "h"
]

UNIT_PATTERN = "|".join(
    re.escape(unit)
    for unit in sorted(
        UNITS,
        key=len,
        reverse=True
    )
)

NUMBER_PATTERN = (
    r"[+-]?(?:\d+(?:[.,]\d+)*|\.\d+)"
    r"(?:[eE][+-]?\d+)?"
)

def is_unit(text, expected_uom=""):

    text = str(text).strip()

    if not text:

        return False

    allowed = set(UNITS)

    if expected_uom:

        allowed.add(expected_uom.strip())

    # Preserve case for engineering units.

    return text in allowed

def split_value_uom(
    text,
    explicit_uom="",
    expected_uom=""
):

    text = str(text).strip()

    if not text:

        return "", explicit_uom

    # Unit from the associated unit column.

    if explicit_uom:

        if text.endswith(explicit_uom):

            prefix = text[
                :-len(explicit_uom)
            ].strip()

            if prefix:

                text = prefix

        return text, explicit_uom

    # Detect units attached to numeric values.

    units = UNIT_PATTERN

    if expected_uom:

        units = (
            re.escape(expected_uom)
            + "|"
            + units
        )

    match = re.fullmatch(
        rf"({NUMBER_PATTERN}"
        rf"(?:\s*(?:-|–|to|±)\s*"
        rf"{NUMBER_PATTERN})?)"
        rf"\s*({units})",
        text
    )

    if match:

        return (
            match.group(1).strip(),
            match.group(2).strip()
        )

    # Do not invent missing units.

    return text, ""

# ==========================================
# ATTRIBUTE LABEL MATCHING
# ==========================================

def match_label(
    label,
    attributes,
    threshold=92
):

    cleaned = normalize(label)

    if not cleaned:

        return None, 0

    # Remove a trailing parenthesized unit.
    # Example: Flow rate (m3/h)

    without_unit = re.sub(
        r"\s*[\[(][^\])]+[\])]\s*$",
        "",
        str(label)
    )

    alternatives = {
        cleaned,
        normalize(without_unit)
    }

    exact = []

    for key, info in attributes.items():

        if alternatives & info["aliases"]:

            exact.append(key)

    if len(exact) == 1:

        return exact[0], 100.0

    if len(exact) > 1:

        return None, 0

    if len(cleaned) < 7:

        return None, 0

    scores = []

    for key, info in attributes.items():

        score = max(
            fuzz.ratio(candidate, alias)
            for candidate in alternatives
            for alias in info["aliases"]
        )

        scores.append((key, score))

    scores.sort(
        key=lambda item: item[1],
        reverse=True
    )

    if not scores:

        return None, 0

    best_key, best_score = scores[0]

    if best_score < threshold:

        return None, 0

    if len(scores) > 1:

        if best_score - scores[1][1] < 5:

            return None, 0

    return best_key, float(best_score)

# ==========================================
# UNIT AND CONDITION HEADERS
# ==========================================

UNIT_HEADERS = {
    "unit",
    "units",
    "uom",
    "unit of measurement",
    "unit of measure"
}

VALUE_HEADERS = {
    "value",
    "values",
    "parameter value",
    "specified value",
    "actual value",
    "data",
    "requirement"
}

CONDITION_HEADERS = {
    "min": "Minimum",
    "minimum": "Minimum",
    "normal": "Normal",
    "norm": "Normal",
    "max": "Maximum",
    "maximum": "Maximum",
    "rated": "Rated",
    "design": "Design",
    "operating": "Operating",
    "test": "Test"
}

NA_VALUES = {
    "n/a",
    "na",
    "not applicable"
}

# ==========================================
# EXTRACT PARAMETER VALUES
# ==========================================

def extract_parameters(
    extracted,
    attributes,
    label_threshold=92
):

    records = defaultdict(list)

    # --------------------------------------
    # Save an extracted observation
    # --------------------------------------

    def add_record(
        key,
        value,
        unit,
        pages,
        source,
        method,
        similarity,
        condition="",
        source_location="",
        blank_confirmed=False
    ):

        value = str(value).strip()

        expected_uom = attributes[key].get(
            "expected_uom", ""
        )

        value, unit = split_value_uom(
            value,
            unit,
            expected_uom
        )

        if not value:

            status = (
                "EMPTY"
                if blank_confirmed
                else "VALUE_NOT_RESOLVED"
            )

        elif value.lower() in NA_VALUES:

            status = "NOT_APPLICABLE"

        elif value in {"-", "--", "—"}:

            status = "UNSPECIFIED"

        else:

            status = "EXTRACTED"

        validation = ""

        if unit and expected_uom:

            if unit != expected_uom:

                validation = (
                    "Unit differs from masterlist"
                )

        records[key].append({
            "Attribute": attributes[key]["attribute"],
            "Value": value,
            "UoM": unit,
            "Expected UoM": expected_uom,
            "Condition": condition,
            "Page": ", ".join(
                str(page) for page in pages
            ),
            "Status": status,
            "Validation": validation,
            "Method": method,
            "Label Similarity": round(
                similarity, 2
            ),
            "Source": source,
            "Source Location": source_location
        })

    # ======================================
    # METHOD 1: STRUCTURED TABLES
    # ======================================

    for table in extracted["tables"]:

        grid = table["grid"]

        pages = table["pages"]

        if not grid:

            continue

        header_rows = table["header_rows"]

        occupied = table["occupied"]

        # Identify known attribute labels.

        label_map = {}

        for r, row in enumerate(grid):

            for c, cell in enumerate(row):

                key, similarity = match_label(
                    cell,
                    attributes,
                    label_threshold
                )

                if key:

                    label_map[(r, c)] = (
                        key,
                        similarity
                    )

        # Find recognized column headers.

        recognized_headers = set(header_rows)

        for r, row in enumerate(grid):

            normalized_row = {
                normalize(cell)
                for cell in row
            }

            if normalized_row & (
                UNIT_HEADERS | VALUE_HEADERS
            ):

                recognized_headers.add(r)

            elif normalized_row & set(
                CONDITION_HEADERS
            ):

                recognized_headers.add(r)

        for (r, c), (
            key,
            similarity
        ) in label_map.items():

            row = grid[r]

            expected_uom = attributes[key].get(
                "expected_uom", ""
            )

            source = " | ".join(row)

            location = (
                f"{table['name']}, "
                f"row {r + 1}, column {c + 1}"
            )

            # ------------------------------
            # Vertical header-value layout
            # ------------------------------

            if r in header_rows:

                # Require exactly one data row
                # immediately below the header.
                # Multiple data rows may represent
                # different equipment or conditions.

                if r + 1 == len(grid) - 1:

                    next_row = grid[r + 1]

                    value = next_row[c]

                    if (
                        (r + 1, c) not in label_map
                        and (r + 1, c) in occupied
                    ):

                        add_record(
                            key,
                            value,
                            "",
                            pages,
                            " | ".join(next_row),
                            "Vertical table",
                            similarity,
                            source_location=location,
                            blank_confirmed=True
                        )

                continue

            # ------------------------------
            # Find horizontal row boundary
            # ------------------------------

            next_labels = [
                col
                for row_id, col in label_map
                if row_id == r and col > c
            ]

            end = (
                min(next_labels)
                if next_labels
                else len(row)
            )

            # Find nearest preceding header.

            previous_headers = [
                h for h in recognized_headers
                if h < r
            ]

            header_index = (
                max(previous_headers)
                if previous_headers
                else None
            )

            headers = (
                grid[header_index]
                if header_index is not None
                else [""] * len(row)
            )

            unit = ""

            # Unit embedded in attribute label.

            unit_in_label = re.search(
                r"[\[(]([^\])]+)[\])]\s*$",
                row[c]
            )

            if unit_in_label:

                candidate = unit_in_label.group(1)

                if is_unit(
                    candidate,
                    expected_uom
                ):

                    unit = candidate

            # Explicit UoM column.

            for j in range(c + 1, end):

                if normalize(headers[j]) in UNIT_HEADERS:

                    if row[j].strip():

                        unit = row[j].strip()

            # ------------------------------
            # Explicit operating conditions
            # and value columns
            # ------------------------------

            extracted_from_header = False

            for j in range(c + 1, end):

                header = normalize(headers[j])

                if (
                    header not in CONDITION_HEADERS
                    and header not in VALUE_HEADERS
                ):

                    continue

                if (r, j) not in occupied:

                    continue

                condition = CONDITION_HEADERS.get(
                    header, ""
                )

                add_record(
                    key,
                    row[j],
                    unit,
                    pages,
                    source,
                    "Structured value column",
                    similarity,
                    condition=condition,
                    source_location=location,
                    blank_confirmed=True
                )

                extracted_from_header = True

            if extracted_from_header:

                continue

            # ------------------------------
            # Simple horizontal layout:
            # Attribute | Value | UoM
            # Attribute | UoM | Value
            # ------------------------------

            # Only infer a value when no competing
            # populated fields remain.

            candidate_values = []

            for j in range(c + 1, end):

                cell = row[j].strip()

                if not cell:

                    continue

                if normalize(headers[j]) in UNIT_HEADERS:

                    continue

                if is_unit(cell, expected_uom):

                    # A single-character unit can
                    # also be a textual value.
                    # Only infer it when another
                    # populated cell is available.

                    others = [
                        row[k].strip()
                        for k in range(c + 1, end)
                        if k != j and row[k].strip()
                    ]

                    if others:

                        unit = cell

                        continue

                candidate_values.append(cell)

            if len(candidate_values) == 1:

                add_record(
                    key,
                    candidate_values[0],
                    unit,
                    pages,
                    source,
                    "Horizontal table",
                    similarity,
                    source_location=location
                )

            elif len(candidate_values) > 1:

                records[key].append({
                    "Attribute":
                        attributes[key]["attribute"],

                    "Value":
                        " | ".join(candidate_values),

                    "UoM": unit,
                    "Expected UoM": expected_uom,
                    "Condition": "",

                    "Page": ", ".join(
                        str(page) for page in pages
                    ),

                    "Status": "AMBIGUOUS",
                    "Validation":
                        "Multiple possible value cells",

                    "Method": "Horizontal table",

                    "Label Similarity":
                        round(similarity, 2),

                    "Source": source,
                    "Source Location": location
                })

            else:

                add_record(
                    key,
                    "",
                    unit,
                    pages,
                    source,
                    "Horizontal table",
                    similarity,
                    source_location=location
                )

    # ======================================
    # METHOD 2: INLINE TEXT
    # Example: Operating Pressure: 150 bar
    # ======================================

    for block in extracted["text_blocks"]:

        pages = block["pages"]

        for line in block["text"].splitlines():

            line = line.strip()

            # Separate at the first colon or equals.

            match = re.match(
                r"^(.{1,150}?)\s*[:=]\s*(.*)$",
                line
            )

            if not match:

                continue

            label = match.group(1).strip()

            value = match.group(2).strip()

            key, similarity = match_label(
                label,
                attributes,
                label_threshold
            )

            if not key:

                continue

            add_record(
                key,
                value,
                "",
                pages,
                line,
                "Inline text",
                similarity
            )

    # ======================================
    # CONSOLIDATE ALL PAGES
    # ======================================

    final_results = []

    for key, info in attributes.items():

        observations = records.get(key, [])

        if not observations:

            # Check if the attribute was detected
            # but its value could not be associated.

            aliases = info["aliases"]

            detected = any(
                f" {alias} " in f" {segment} "
                for alias in aliases
                for segment in extracted["segments"]
            )

            final_results.append({
                "Attribute": info["attribute"],
                "Value": "",
                "UoM": "",
                "Expected UoM": info.get(
                    "expected_uom", ""
                ),
                "Condition": "",
                "Page": "",
                "Status": (
                    "VALUE_NOT_RESOLVED"
                    if detected
                    else "NOT_DETECTED"
                ),
                "Validation": "",
                "Method": "",
                "Label Similarity": 0,
                "Source": "",
                "Source Location": ""
            })

            continue

        # Deduplicate identical observations.
        # Preserve all contributing page references.

        unique = {}

        for record in observations:

            signature = (
                record["Value"].strip(),
                record["UoM"].strip(),
                record["Condition"],
                record["Status"]
            )

            if signature not in unique:

                unique[signature] = record.copy()

            else:

                existing = unique[signature]

                pages = {
                    p.strip()
                    for text in [
                        existing["Page"],
                        record["Page"]
                    ]
                    for p in text.split(",")
                    if p.strip()
                }

                existing["Page"] = ", ".join(
                    sorted(pages, key=int)
                )

                for field in [
                    "Source",
                    "Source Location"
                ]:

                    if (
                        record[field]
                        and record[field]
                        not in existing[field]
                    ):

                        existing[field] += (
                            " || " + record[field]
                        )

        deduplicated = list(unique.values())

        # Detect conflicting values within the
        # same operating condition.

        condition_values = defaultdict(set)

        for record in deduplicated:

            if record["Status"] == "EXTRACTED":

                condition_values[
                    record["Condition"]
                ].add((
                    record["Value"],
                    record["UoM"]
                ))

        for record in deduplicated:

            if (
                record["Status"] == "EXTRACTED"
                and len(
                    condition_values[
                        record["Condition"]
                    ]
                ) > 1
            ):

                record["Status"] = "CONFLICT"

                record["Validation"] = (
                    "Different values for the same "
                    "attribute and condition"
                )

        final_results.extend(deduplicated)

    return final_results
