# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OTSL table post-processing: parse VLM OTSL output and convert it to HTML.

Ported from PaddleX ``paddleocr_vl/uilts.py`` so the OCR pipeline can render
tables without pulling PaddleX into the request path (the layout stage loads
PaddleX lazily and only on the layout device).
"""

import html
import itertools
import re
from dataclasses import dataclass, field

OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"

_NON_CAPTURING_TAG_GROUP = "(?:<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>)"
_OTSL_FIND_PATTERN = re.compile(
    f"{_NON_CAPTURING_TAG_GROUP}.*?(?={_NON_CAPTURING_TAG_GROUP}|$)",
    flags=re.DOTALL,
)
_OTSL_SPLIT_PATTERN = (
    "("
    + "|".join([OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL])
    + ")"
)
_ALL_TAGS = {OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL}
_SPAN_RIGHT = (OTSL_LCEL, OTSL_XCEL)
_SPAN_DOWN = (OTSL_UCEL, OTSL_XCEL)


@dataclass
class TableCell:
    """A single cell in a table, with optional row/col span."""

    start_row_offset_idx: int
    end_row_offset_idx: int
    start_col_offset_idx: int
    end_col_offset_idx: int
    text: str = ""
    row_span: int = 1
    col_span: int = 1
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False


@dataclass
class TableData:
    """Container for all cells in a table plus its row/column dimensions."""

    num_rows: int = 0
    num_cols: int = 0
    table_cells: list[TableCell] = field(default_factory=list)

    @property
    def grid(self) -> list[list[TableCell]]:
        """Return a 2-D grid of TableCell objects (num_rows x num_cols)."""
        table_data = [
            [
                TableCell(
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                for j in range(self.num_cols)
            ]
            for i in range(self.num_rows)
        ]
        for cell in self.table_cells:
            for i in range(
                min(cell.start_row_offset_idx, self.num_rows),
                min(cell.end_row_offset_idx, self.num_rows),
            ):
                for j in range(
                    min(cell.start_col_offset_idx, self.num_cols),
                    min(cell.end_col_offset_idx, self.num_cols),
                ):
                    table_data[i][j] = cell
        return table_data


def export_to_html(table_data: TableData) -> str:
    """Render a TableData object as an HTML ``<table>`` string."""
    if not table_data.table_cells:
        return ""

    grid = table_data.grid
    body = ""
    for i in range(table_data.num_rows):
        body += "<tr>"
        for j in range(table_data.num_cols):
            cell = grid[i][j]
            if cell.start_row_offset_idx != i or cell.start_col_offset_idx != j:
                continue  # already rendered as part of a spanning cell
            content = html.escape(cell.text.strip())
            celltag = "th" if cell.column_header else "td"
            opening = celltag
            if cell.row_span > 1:
                opening += f' rowspan="{cell.row_span}"'
            if cell.col_span > 1:
                opening += f' colspan="{cell.col_span}"'
            body += f"<{opening}>{content}</{celltag}>"
        body += "</tr>"
    return f"<table>{body}</table>"


def otsl_pad_to_sqr(otsl_str: str) -> str:
    """Pad an OTSL string so every row contains the same number of cells.

    Short rows are padded with ``<ecel>``; rows longer than the optimal width
    are truncated. The optimal width minimises the total number of edits across
    all rows while never dropping a row below its last ``<fcel>``.
    """
    otsl_str = otsl_str.strip()
    if OTSL_NL not in otsl_str:
        return otsl_str + OTSL_NL

    row_data = []
    for line in otsl_str.split(OTSL_NL):
        if not line:
            continue
        raw_cells = _OTSL_FIND_PATTERN.findall(line)
        if not raw_cells:
            continue
        min_len = 0
        for i, cell_str in enumerate(raw_cells):
            if cell_str.startswith(OTSL_FCEL):
                min_len = i + 1
        row_data.append(
            {"raw_cells": raw_cells, "total_len": len(raw_cells), "min_len": min_len}
        )

    if not row_data:
        return OTSL_NL

    global_min_width = max(row["min_len"] for row in row_data)
    max_total_len = max(row["total_len"] for row in row_data)
    search_end = max(global_min_width, max_total_len)

    min_total_cost = float("inf")
    optimal_width = search_end
    for width in range(global_min_width, search_end + 1):
        cost = sum(abs(row["total_len"] - width) for row in row_data)
        if cost < min_total_cost:
            min_total_cost = cost
            optimal_width = width

    repaired_lines = []
    for row in row_data:
        cells = row["raw_cells"]
        if len(cells) > optimal_width:
            new_cells = cells[:optimal_width]
        else:
            new_cells = cells + [OTSL_ECEL] * (optimal_width - len(cells))
        repaired_lines.append("".join(new_cells))
    return OTSL_NL.join(repaired_lines) + OTSL_NL


def otsl_extract_tokens_and_text(s: str) -> tuple[list[str], list[str]]:
    """Split an OTSL string into a flat token list and a mixed text/tag list."""
    tokens = re.findall(_OTSL_SPLIT_PATTERN, s)
    text_parts = [t for t in re.split(_OTSL_SPLIT_PATTERN, s) if t.strip()]
    return tokens, text_parts


def _count_right(tokens_2d: list[list[str]], c: int, r: int, which: tuple) -> int:
    span = 0
    while c < len(tokens_2d[r]) and tokens_2d[r][c] in which:
        c += 1
        span += 1
    return span


def _count_down(tokens_2d: list[list[str]], c: int, r: int, which: tuple) -> int:
    span = 0
    while r < len(tokens_2d) and c < len(tokens_2d[r]) and tokens_2d[r][c] in which:
        r += 1
        span += 1
    return span


def otsl_parse_texts(
    texts: list[str], tokens: list[str]
) -> tuple[list[TableCell], list[list[str]]]:
    """Convert a flat OTSL token/text stream into a list of TableCell objects.

    Also returns ``split_row_tokens``, a 2-D list of per-row token strings,
    which is needed to build the final ``TableData`` dimensions.
    """
    split_row_tokens = [
        list(y) for x, y in itertools.groupby(tokens, lambda z: z == OTSL_NL) if not x
    ]

    # Ensure all rows have the same length (second pass, mirrors PaddleX logic)
    if split_row_tokens:
        max_cols = max(len(row) for row in split_row_tokens)
        for row in split_row_tokens:
            row.extend([OTSL_ECEL] * (max_cols - len(row)))

        new_texts: list[str] = []
        text_idx = 0
        for row in split_row_tokens:
            for token in row:
                new_texts.append(token)
                if text_idx < len(texts) and texts[text_idx] == token:
                    text_idx += 1
                    if text_idx < len(texts) and texts[text_idx] not in _ALL_TAGS:
                        new_texts.append(texts[text_idx])
                        text_idx += 1
            new_texts.append(OTSL_NL)
            if text_idx < len(texts) and texts[text_idx] == OTSL_NL:
                text_idx += 1
        texts = new_texts

    table_cells: list[TableCell] = []
    r_idx = 0
    c_idx = 0

    for i, text in enumerate(texts):
        if text in (OTSL_FCEL, OTSL_ECEL):
            row_span = 1
            col_span = 1
            right_offset = 1
            cell_text = ""

            if text != OTSL_ECEL:
                cell_text = texts[i + 1] if i + 1 < len(texts) else ""
                right_offset = 2

            next_right = (
                texts[i + right_offset] if i + right_offset < len(texts) else ""
            )
            next_bottom = ""
            if r_idx + 1 < len(split_row_tokens) and c_idx < len(
                split_row_tokens[r_idx + 1]
            ):
                next_bottom = split_row_tokens[r_idx + 1][c_idx]

            if next_right in _SPAN_RIGHT:
                col_span += _count_right(
                    split_row_tokens, c_idx + 1, r_idx, _SPAN_RIGHT
                )
            if next_bottom in _SPAN_DOWN:
                row_span += _count_down(split_row_tokens, c_idx, r_idx + 1, _SPAN_DOWN)

            table_cells.append(
                TableCell(
                    text=cell_text.strip(),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=r_idx,
                    end_row_offset_idx=r_idx + row_span,
                    start_col_offset_idx=c_idx,
                    end_col_offset_idx=c_idx + col_span,
                )
            )

        if text in _ALL_TAGS and text != OTSL_NL:
            c_idx += 1
        if text == OTSL_NL:
            r_idx += 1
            c_idx = 0

    return table_cells, split_row_tokens


def convert_otsl_to_html(otsl_content: str) -> str:
    """Convert an OTSL-v1.0 string to an HTML ``<table>``.

    Returns an empty string when the input contains no recognisable OTSL
    structure, so callers can fall back to the raw VLM output unchanged.

    Allowed tags: ``<fcel>``, ``<ecel>``, ``<nl>``, ``<lcel>``, ``<ucel>``,
    ``<xcel>``.
    """
    otsl_content = otsl_pad_to_sqr(otsl_content)
    tokens, mixed_texts = otsl_extract_tokens_and_text(otsl_content)
    table_cells, split_row_tokens = otsl_parse_texts(mixed_texts, tokens)
    table_data = TableData(
        num_rows=len(split_row_tokens),
        num_cols=max((len(row) for row in split_row_tokens), default=0),
        table_cells=table_cells,
    )
    return export_to_html(table_data)
