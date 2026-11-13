"""Task schemas, read from the active joint (x, y) generator - not re-declared here.

Column roles (categorical / binary / ordinal / continuous / target / constant) are
taken from the module constants of
`00_TabPFN_Transformations/03_SyntheticData_Generation_JointXY/create_*_jointxy.py`
so that the split-safe pools and the saved full-cohort pools are generated from the
same column contract. Dtypes come from the processed CSV and are cross-checked
against the saved data dictionary. Nothing is inferred from dtype alone.
"""
import dataclasses
import importlib.util

from .paths import read_csv, repo_path

TASKS = ("diagnostic", "prognostic")
ID_COL = "patient_id"
SYNTHETIC_ID_PREFIX = "SYN_AUG_"
TASK_CODES = {"diagnostic": "DIAG", "prognostic": "PROG"}


@dataclasses.dataclass(frozen=True)
class TaskSchema:
    task: str
    column_order: tuple          # exact processed-CSV order, patient_id first
    categorical: tuple
    binary: tuple                # generated binary columns (incl. the target/event)
    ordinal: tuple
    continuous: tuple
    target: str                  # diagnostic: mets_liver; prognostic: os_event
    time_col: str | None         # prognostic: os_months
    constant_cols: dict          # prognostic: {"mets_liver": 1}
    dtypes: dict                 # column -> pandas dtype string of the processed CSV
    generator_module: object = dataclasses.field(repr=False, compare=False)

    @property
    def feature_cols(self):
        return tuple(c for c in self.column_order if c != ID_COL)


def _load_generator_module(cfg, task):
    path = repo_path(cfg["inputs"][f"active_joint_generator_{task}"])
    spec = importlib.util.spec_from_file_location(f"_c10_active_jointxy_{task}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def _check_data_dictionary(cfg, schema):
    """Every schema column must be documented, and its declared kind must agree."""
    dd = read_csv(repo_path(cfg["inputs"]["data_dictionary"]))
    documented = dict(zip(dd["column"], dd["dtype"]))
    problems = []
    for c in schema.column_order:
        if c not in documented:
            problems.append(f"{c}: absent from the data dictionary")
            continue
        is_string = documented[c] == "object"
        if (c in schema.categorical or c == ID_COL) != is_string:
            problems.append(f"{c}: data dictionary dtype {documented[c]!r} disagrees with its "
                            f"role in the active generator")
    if problems:
        raise RuntimeError(f"{schema.task}: schema contract violated: {problems}")


def load_schema(cfg, task):
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    mod, _ = _load_generator_module(cfg, task)
    if task == "diagnostic":
        order, target, time_col, const = mod.DIAGNOSTIC_COLUMN_ORDER, "mets_liver", None, {}
    else:
        order, target, time_col = mod.PROGNOSTIC_COLUMN_ORDER, "os_event", mod.TIME_COL
        const = {mod.CONSTANT_COL: 1}
    df_head = read_csv(repo_path(cfg["inputs"][f"{task}_input"]))
    if list(df_head.columns) != list(order):
        raise RuntimeError(f"{task}: processed CSV columns {list(df_head.columns)} differ from the "
                           f"active generator's column order {list(order)}")
    if target not in mod.BINARY_COLS:
        raise RuntimeError(f"{task}: target {target!r} is not a generated binary column")
    schema = TaskSchema(
        task=task, column_order=tuple(order), categorical=tuple(mod.CATEGORICAL_COLS),
        binary=tuple(mod.BINARY_COLS), ordinal=tuple(mod.ORDINAL_COLS),
        continuous=tuple(mod.CONTINUOUS_COLS), target=target, time_col=time_col,
        constant_cols=const, dtypes={c: str(t) for c, t in df_head.dtypes.items()},
        generator_module=mod)
    covered = (set(schema.categorical) | set(schema.binary) | set(schema.ordinal)
               | set(schema.continuous) | set(const) | {ID_COL} | ({time_col} if time_col else set()))
    if covered != set(order):
        raise RuntimeError(f"{task}: columns without a role: {sorted(set(order) - covered)}; "
                           f"roles without a column: {sorted(covered - set(order))}")
    _check_data_dictionary(cfg, schema)
    return schema


def read_task_frame(cfg, schema):
    """The processed task dataset, validated against the schema. No transformation."""
    df = read_csv(repo_path(cfg["inputs"][f"{schema.task}_input"]))
    if list(df.columns) != list(schema.column_order):
        raise RuntimeError(f"{schema.task}: column order changed since the schema was loaded")
    if df[ID_COL].duplicated().any():
        raise RuntimeError(f"{schema.task}: duplicate patient IDs in the processed input")
    if df.isna().any().any():
        raise RuntimeError(f"{schema.task}: missing values in the processed input")
    if df[ID_COL].astype(str).str.startswith(SYNTHETIC_ID_PREFIX).any():
        raise RuntimeError(f"{schema.task}: a real patient ID uses the reserved synthetic prefix "
                           f"{SYNTHETIC_ID_PREFIX!r}")
    for col, val in schema.constant_cols.items():
        if not (df[col] == val).all():
            raise RuntimeError(f"{schema.task}: {col} is not constant {val} in the real cohort")
    return df
