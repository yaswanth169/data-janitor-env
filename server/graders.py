import re
from typing import Any, Dict, List, Optional


def values_match(actual: Any, expected: Any) -> bool:
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False

    try:
        a_str = str(actual).strip().replace(",", "").replace("$", "")
        e_str = str(expected).strip().replace(",", "").replace("$", "")
        a_num = float(a_str)
        e_num = float(e_str)
        return abs(a_num - e_num) < 0.02
    except (ValueError, TypeError):
        pass

    return str(actual).strip().lower() == str(expected).strip().lower()


def grade(
    cleaned_data: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    primary_key: str,
) -> float:
    if not ground_truth:
        return 0.0
    if not cleaned_data:
        return 0.0

    truth_lookup: Dict[str, Dict[str, Any]] = {}
    for row in ground_truth:
        key = str(row[primary_key]).strip()
        truth_lookup[key] = row

    value_columns = [c for c in ground_truth[0].keys() if c != primary_key]
    total_cells = len(ground_truth) * len(value_columns)
    if total_cells == 0:
        return 0.0

    correct_cells = 0
    matched_keys: set = set()

    for row in cleaned_data:
        key = str(row.get(primary_key, "")).strip()
        if key in truth_lookup and key not in matched_keys:
            matched_keys.add(key)
            truth_row = truth_lookup[key]
            for col in value_columns:
                if values_match(row.get(col), truth_row.get(col)):
                    correct_cells += 1

    return round(correct_cells / total_cells, 4)


def detect_issues(
    data: List[Dict[str, Any]],
    target_schema: Optional[Dict[str, str]] = None,
) -> List[str]:
    if not data:
        return ["Dataset is empty."]

    issues: List[str] = []
    columns = list(data[0].keys())

    # Duplicate detection: exact row match
    row_fingerprints = []
    for row in data:
        fp = tuple((col, str(row.get(col, ""))) for col in columns)
        row_fingerprints.append(fp)
    n_exact_unique = len(set(row_fingerprints))
    if n_exact_unique < len(data):
        issues.append(f"{len(data) - n_exact_unique} exact duplicate rows detected.")

    # Duplicate detection: by likely primary key (first column or 'id'-like column)
    pk_candidates = [c for c in columns if "id" in c.lower()]
    if pk_candidates:
        pk = pk_candidates[0]
        pk_values = [str(row.get(pk, "")).strip() for row in data]
        from collections import Counter
        pk_counts = Counter(pk_values)
        pk_dupes = sum(v - 1 for v in pk_counts.values() if v > 1)
        if pk_dupes > 0:
            issues.append(f"{pk_dupes} duplicate values in '{pk}'.")

    for col in columns:
        values = [row.get(col) for row in data]
        non_null = [v for v in values if v is not None]
        str_values = [str(v).strip() for v in non_null]

        # Null/missing check
        null_count = sum(
            1 for v in values
            if v is None or (isinstance(v, str) and v.strip().lower() in (
                "", "n/a", "na", "null", "none", "nan", "-", "missing",
            ))
        )
        if null_count > 0:
            pct = null_count / len(values) * 100
            issues.append(
                f"'{col}' has {null_count} null/missing values ({pct:.0f}%)."
            )

        # Mixed types
        types = set(type(v).__name__ for v in non_null)
        if len(types) > 1:
            issues.append(f"'{col}' has mixed types: {types}.")

        # Casing inconsistency (strings only)
        str_only = [v for v in non_null if isinstance(v, str)]
        if len(str_only) > 5:
            lowered = set(s.strip().lower() for s in str_only)
            original = set(s.strip() for s in str_only)
            if len(original) > len(lowered):
                issues.append(f"'{col}' has inconsistent casing.")

        # Whitespace issues
        if str_only:
            with_spaces = sum(1 for s in str_only if s != s.strip())
            if with_spaces > 0:
                issues.append(
                    f"'{col}' has {with_spaces} values with leading/trailing whitespace."
                )

        # Currency symbols in numeric-looking columns
        if str_only:
            currency_count = sum(1 for s in str_only if "$" in s or "," in s)
            if currency_count > 0 and any(
                c.replace("$", "").replace(",", "").replace(".", "").replace(" ", "").isdigit()
                for c in str_only if "$" in c or "," in c
            ):
                issues.append(
                    f"'{col}' has {currency_count} values with currency formatting."
                )

        # Mixed date formats
        if str_only and len(str_only) > 5:
            date_patterns = [
                r"^\d{4}-\d{2}-\d{2}$",
                r"^\d{2}/\d{2}/\d{4}$",
                r"^[A-Z][a-z]+ \d+,? \d{4}$",
                r"^\d{2}-[A-Z][a-z]+-\d{4}$",
            ]
            matched_formats = set()
            for s in str_only:
                for i, pat in enumerate(date_patterns):
                    if re.match(pat, s.strip()):
                        matched_formats.add(i)
            if len(matched_formats) > 1:
                issues.append(f"'{col}' has dates in {len(matched_formats)} different formats.")

    # Missing expected columns
    if target_schema:
        for expected_col in target_schema:
            if expected_col not in columns:
                issues.append(f"Expected column '{expected_col}' is missing.")

    return issues if issues else ["No obvious issues detected."]
