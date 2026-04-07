import re
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional


DATE_FORMATS = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%b %d, %Y",
    "%d-%b-%Y",
    "%B %d %Y",
    "%B %d, %Y",
    "%Y/%m/%d",
    "%m-%d-%Y",
    "%d.%m.%Y",
]


def is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in (
        "", "n/a", "na", "null", "none", "nan", "-", "missing",
    ):
        return True
    return False


def clean_numeric(value: Any) -> Optional[float]:
    if is_null(value):
        return None
    s = str(value).strip().replace("$", "").replace(",", "").replace(" ", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_date(value: str) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def normalize_phone(value: Any) -> str:
    if is_null(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    if len(digits) == 11 and digits[0] == "1":
        digits = digits[1:]
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return str(value)


class DataEngine:
    COMMANDS = [
        "inspect",
        "drop_duplicates",
        "fill_missing",
        "drop_nulls",
        "convert_type",
        "normalize_text",
        "standardize_date",
        "standardize_phone",
        "rename_column",
        "map_values",
        "filter_rows",
        "split_column",
        "merge_columns",
        "join",
        "add_column",
        "submit",
    ]

    def __init__(
        self,
        data: List[Dict[str, Any]],
        secondary_data: Optional[List[Dict[str, Any]]] = None,
    ):
        self.data = [dict(row) for row in data]
        self.secondary_data = (
            [dict(row) for row in secondary_data] if secondary_data else None
        )

    @property
    def columns(self) -> List[str]:
        return list(self.data[0].keys()) if self.data else []

    def execute(self, command: str, column: Optional[str], params: Dict[str, Any]) -> str:
        if command not in self.COMMANDS:
            return f"Unknown command '{command}'. Available: {', '.join(self.COMMANDS)}"
        if command == "submit":
            return "submitted"
        handler = getattr(self, f"_cmd_{command}", None)
        if not handler:
            return f"Command '{command}' is not implemented."
        try:
            return handler(column, params)
        except Exception as e:
            return f"Error executing '{command}': {e}"

    def _validate_column(self, column: Optional[str]) -> Optional[str]:
        if not column:
            return "Column name is required for this command."
        if column not in self.columns:
            return f"Column '{column}' not found. Available: {self.columns}"
        return None

    # ── inspect ──────────────────────────────────────────────────────────

    def _cmd_inspect(self, column: Optional[str], params: Dict) -> str:
        if column:
            err = self._validate_column(column)
            if err:
                return err
            values = [row.get(column) for row in self.data]
            non_null = [v for v in values if not is_null(v)]
            null_count = len(values) - len(non_null)
            unique = set(str(v) for v in non_null)
            types = set(type(v).__name__ for v in non_null)
            sample = [str(v) for v in non_null[:8]]
            return (
                f"Column '{column}': {len(values)} total, {null_count} nulls, "
                f"{len(unique)} unique, types: {types}. Sample: {sample}"
            )
        return f"Dataset: {len(self.data)} rows, columns: {self.columns}"

    # ── drop_duplicates ──────────────────────────────────────────────────

    def _cmd_drop_duplicates(self, column: Optional[str], params: Dict) -> str:
        subset = params.get("subset", self.columns)
        if isinstance(subset, str):
            subset = [subset]

        seen: set = set()
        unique: List[Dict] = []
        for row in self.data:
            key = tuple(str(row.get(col, "")) for col in subset)
            if key not in seen:
                seen.add(key)
                unique.append(row)

        removed = len(self.data) - len(unique)
        self.data = unique
        return f"Removed {removed} duplicate rows. {len(self.data)} rows remaining."

    # ── fill_missing ─────────────────────────────────────────────────────

    def _cmd_fill_missing(self, column: Optional[str], params: Dict) -> str:
        err = self._validate_column(column)
        if err:
            return err

        strategy = params.get("strategy", "constant")
        fill_value = params.get("value")

        if strategy == "constant" and fill_value is None:
            return "Strategy 'constant' requires a 'value' parameter."

        non_null_values = [row[column] for row in self.data if not is_null(row.get(column))]

        if strategy == "mean":
            nums = [n for n in (clean_numeric(v) for v in non_null_values) if n is not None]
            fill_value = round(statistics.mean(nums), 2) if nums else 0
        elif strategy == "median":
            nums = [n for n in (clean_numeric(v) for v in non_null_values) if n is not None]
            fill_value = round(statistics.median(nums), 2) if nums else 0
        elif strategy == "mode":
            fill_value = (
                max(set(non_null_values), key=non_null_values.count)
                if non_null_values
                else ""
            )

        filled = 0
        for row in self.data:
            if is_null(row.get(column)):
                row[column] = fill_value
                filled += 1

        return f"Filled {filled} missing values in '{column}' with {strategy} ({fill_value})."

    # ── drop_nulls ───────────────────────────────────────────────────────

    def _cmd_drop_nulls(self, column: Optional[str], params: Dict) -> str:
        err = self._validate_column(column)
        if err:
            return err

        before = len(self.data)
        self.data = [row for row in self.data if not is_null(row.get(column))]
        removed = before - len(self.data)
        return f"Dropped {removed} rows with null '{column}'. {len(self.data)} remaining."

    # ── convert_type ─────────────────────────────────────────────────────

    def _cmd_convert_type(self, column: Optional[str], params: Dict) -> str:
        err = self._validate_column(column)
        if err:
            return err

        target = params.get("target_type", "str")
        converted = 0
        errors = 0

        for row in self.data:
            val = row[column]
            if is_null(val):
                row[column] = None
                continue
            try:
                if target == "int":
                    cleaned = clean_numeric(val)
                    row[column] = int(cleaned) if cleaned is not None else None
                elif target == "float":
                    row[column] = clean_numeric(val)
                elif target == "str":
                    row[column] = str(val)
                else:
                    return f"Unsupported target type '{target}'. Use: int, float, str."
                converted += 1
            except (ValueError, TypeError):
                row[column] = None
                errors += 1

        return f"Converted {converted} values in '{column}' to {target}. {errors} errors."

    # ── normalize_text ───────────────────────────────────────────────────

    def _cmd_normalize_text(self, column: Optional[str], params: Dict) -> str:
        err = self._validate_column(column)
        if err:
            return err

        operation = params.get("operation", "trim")
        pattern = params.get("pattern", "")
        replacement = params.get("replacement", "")
        modified = 0

        for row in self.data:
            val = row[column]
            if is_null(val):
                continue
            original = str(val)
            if operation == "trim":
                row[column] = original.strip()
            elif operation == "lower":
                row[column] = original.strip().lower()
            elif operation == "upper":
                row[column] = original.strip().upper()
            elif operation == "title":
                row[column] = original.strip().title()
            elif operation == "regex_replace":
                if not pattern:
                    return "regex_replace requires a 'pattern' parameter."
                row[column] = re.sub(pattern, replacement, original)
            else:
                return (
                    f"Unknown operation '{operation}'. "
                    "Use: trim, lower, upper, title, regex_replace."
                )
            if row[column] != original:
                modified += 1

        return f"Normalized {modified} values in '{column}' with '{operation}'."

    # ── standardize_date ─────────────────────────────────────────────────

    def _cmd_standardize_date(self, column: Optional[str], params: Dict) -> str:
        err = self._validate_column(column)
        if err:
            return err

        target_format = params.get("format", "%Y-%m-%d")
        converted = 0
        failed = 0

        for row in self.data:
            val = row[column]
            if is_null(val):
                continue
            parsed = parse_date(str(val))
            if parsed:
                row[column] = parsed.strftime(target_format)
                converted += 1
            else:
                failed += 1

        return (
            f"Standardized {converted} dates in '{column}'. "
            f"{failed} could not be parsed."
        )

    # ── standardize_phone ────────────────────────────────────────────────

    def _cmd_standardize_phone(self, column: Optional[str], params: Dict) -> str:
        err = self._validate_column(column)
        if err:
            return err

        modified = 0
        for row in self.data:
            val = row[column]
            if is_null(val):
                continue
            normalized = normalize_phone(val)
            if normalized != str(val):
                modified += 1
            row[column] = normalized

        return f"Standardized {modified} phone numbers in '{column}'."

    # ── rename_column ────────────────────────────────────────────────────

    def _cmd_rename_column(self, column: Optional[str], params: Dict) -> str:
        err = self._validate_column(column)
        if err:
            return err

        new_name = params.get("new_name")
        if not new_name:
            return "Parameter 'new_name' is required."

        for row in self.data:
            row[new_name] = row.pop(column, None)

        return f"Renamed '{column}' to '{new_name}'."

    # ── map_values ───────────────────────────────────────────────────────

    def _cmd_map_values(self, column: Optional[str], params: Dict) -> str:
        err = self._validate_column(column)
        if err:
            return err

        mapping = params.get("mapping", {})
        if not mapping:
            return "Parameter 'mapping' (dict) is required."

        modified = 0
        for row in self.data:
            key = str(row[column]) if row[column] is not None else None
            if key in mapping:
                row[column] = mapping[key]
                modified += 1

        return f"Mapped {modified} values in '{column}'."

    # ── filter_rows ──────────────────────────────────────────────────────

    def _cmd_filter_rows(self, column: Optional[str], params: Dict) -> str:
        err = self._validate_column(column)
        if err:
            return err

        operator = params.get("operator", "==")
        value = params.get("value")
        if value is None:
            return "Parameter 'value' is required."

        before = len(self.data)
        kept: List[Dict] = []

        for row in self.data:
            cell = row.get(column)
            remove = False
            try:
                if operator == "==":
                    remove = str(cell).strip() == str(value).strip()
                elif operator == "!=":
                    remove = str(cell).strip() != str(value).strip()
                elif operator in (">", "<", ">=", "<="):
                    num = clean_numeric(cell)
                    threshold = float(value)
                    if num is not None:
                        if operator == ">":
                            remove = num > threshold
                        elif operator == "<":
                            remove = num < threshold
                        elif operator == ">=":
                            remove = num >= threshold
                        elif operator == "<=":
                            remove = num <= threshold
                elif operator == "is_null":
                    remove = is_null(cell)
                else:
                    return f"Unknown operator '{operator}'. Use: ==, !=, >, <, >=, <=, is_null."
            except (ValueError, TypeError):
                pass

            if not remove:
                kept.append(row)

        self.data = kept
        removed = before - len(self.data)
        return f"Removed {removed} rows where '{column}' {operator} {value}. {len(self.data)} remaining."

    # ── split_column ─────────────────────────────────────────────────────

    def _cmd_split_column(self, column: Optional[str], params: Dict) -> str:
        err = self._validate_column(column)
        if err:
            return err

        delimiter = params.get("delimiter", ",")
        new_columns = params.get("new_columns", [])
        if not new_columns:
            return "Parameter 'new_columns' (list of names) is required."

        for row in self.data:
            val = str(row.get(column, ""))
            parts = val.split(delimiter)
            for i, new_col in enumerate(new_columns):
                row[new_col] = parts[i].strip() if i < len(parts) else None

        if params.get("drop_original", False):
            for row in self.data:
                row.pop(column, None)

        return f"Split '{column}' into {new_columns}."

    # ── merge_columns ────────────────────────────────────────────────────

    def _cmd_merge_columns(self, column: Optional[str], params: Dict) -> str:
        columns_list = params.get("columns", [])
        separator = params.get("separator", " ")
        new_column = params.get("new_column", column)

        if not columns_list:
            return "Parameter 'columns' (list) is required."
        if not new_column:
            return "Parameter 'new_column' or column is required."

        for row in self.data:
            parts = [str(row.get(col, "")) for col in columns_list]
            row[new_column] = separator.join(parts)

        if params.get("drop_originals", False):
            for row in self.data:
                for col in columns_list:
                    if col != new_column:
                        row.pop(col, None)

        return f"Merged {columns_list} into '{new_column}'."

    # ── join ─────────────────────────────────────────────────────────────

    def _cmd_join(self, column: Optional[str], params: Dict) -> str:
        if self.secondary_data is None:
            return (
                "Join already completed — secondary dataset was merged earlier this episode. "
                f"Current table has {len(self.data)} rows and columns: {self.columns}. "
                "Do NOT call join again. Clean remaining issues (casing, types, totals) and submit."
            )

        on = column or params.get("on")
        if not on:
            return "Join column required via 'column' or params 'on'."

        how = params.get("how", "inner")
        if how not in ("inner", "left"):
            return "Supported join types: 'inner', 'left'."

        lookup: Dict[str, Dict] = {}
        for row in self.secondary_data:
            key = str(row.get(on, "")).strip()
            lookup[key] = row

        joined: List[Dict] = []
        matched = 0
        for row in self.data:
            key = str(row.get(on, "")).strip()
            merged_row = dict(row)
            if key in lookup:
                for k, v in lookup[key].items():
                    if k != on:
                        merged_row[k] = v
                matched += 1
                joined.append(merged_row)
            elif how == "left":
                joined.append(merged_row)

        self.data = joined
        self.secondary_data = None
        return (
            f"Joined {matched} rows on '{on}' ({how}). "
            f"{len(self.data)} rows in result."
        )

    # ── add_column ───────────────────────────────────────────────────────

    def _cmd_add_column(self, column: Optional[str], params: Dict) -> str:
        if not column:
            return "Column name for the new column is required."

        expression = params.get("expression", "")
        if not expression:
            return "Parameter 'expression' is required (e.g., 'quantity * unit_price')."

        match = re.match(r"^(\w+)\s*([+\-*/])\s*(\w+)$", expression.strip())
        if not match:
            constant = params.get("value")
            if constant is not None:
                for row in self.data:
                    row[column] = constant
                return f"Added column '{column}' with constant value {constant}."
            return (
                f"Expression '{expression}' not supported. "
                "Use: 'column_a operator column_b' (operators: +, -, *, /)."
            )

        col_a, op, col_b = match.groups()
        computed = 0
        for row in self.data:
            a = clean_numeric(row.get(col_a))
            b = clean_numeric(row.get(col_b))
            if a is not None and b is not None:
                if op == "+":
                    row[column] = round(a + b, 2)
                elif op == "-":
                    row[column] = round(a - b, 2)
                elif op == "*":
                    row[column] = round(a * b, 2)
                elif op == "/":
                    row[column] = round(a / b, 2) if b != 0 else None
                computed += 1
            else:
                row[column] = None

        return f"Computed '{column}' = {expression} for {computed} rows."
