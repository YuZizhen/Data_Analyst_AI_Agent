"""Regression tools with explicit model selection and held-out results."""

from typing import Literal

import numpy as np
import pandas as pd
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .candidates import candidates_for_tool
from .dataset import AnalysisContext, column, numeric_column, working_frame

Model = Literal["decision_tree", "linear_regression", "ridge", "random_forest"]
Split = Literal["random", "chronological", "cross_validation"]
MIN_VALID_ROWS = 30
MAX_MODEL_ROWS = 12_000
MAX_CATEGORIES = 64
VALIDATION_FOLDS = 3
RANDOM_SEED = 42
MODELS = {
    "decision_tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=4, random_state=RANDOM_SEED),
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0, solver="lsqr"),
    "random_forest": RandomForestRegressor(
        n_estimators=64, max_features=0.5, max_depth=18, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=1,
    ),
}


def prediction_candidates(frame, target_column, target, candidate_columns, method, runtime):
    review = candidates_for_tool(runtime, frame, target_column, target, candidate_columns, method)
    if not review.values:
        raise ToolException("No eligible candidate factors in the working data. See the candidate review for details.")
    numeric = [row["field"] for row in review.evidence["fields"] if row["status"] == "included" and row["kind"] == "numeric"]
    categorical = [name for name in review.values if name not in numeric]
    return review.values, numeric, categorical


def validation_partitions(frame, target, split, test_size, time_column):
    if split != "cross_validation" and not 0 < test_size < 1:
        raise ToolException("test_size must be between 0 and 1.")
    if split != "chronological" and time_column:
        raise ToolException("Use split='chronological' when specifying time_column.")
    evidence = {"split": split, "test_size": None if split == "cross_validation" else test_size,
                "split_seed": None if split == "chronological" else RANDOM_SEED}
    if split == "chronological":
        if not time_column or time_column == target.name:
            raise ToolException("Choose a date column distinct from the target for a chronological split.")
        raw_dates = column(frame, time_column).loc[target.index]
        if pd.api.types.is_numeric_dtype(raw_dates):
            raise ToolException("Convert the chronological split field to dates or ISO date text first.")
        dates = pd.to_datetime(raw_dates, format="mixed", errors="coerce", utc=True).dropna().sort_values(kind="stable")
        if len(dates) < MIN_VALID_ROWS or dates.nunique() < 2:
            raise ToolException(f"A chronological split needs {MIN_VALID_ROWS} rows with usable dates and at least two distinct dates.")
        counts = dates.value_counts(sort=False).sort_index().cumsum()
        choices = counts[counts.between(2, len(dates) - 2)]
        if choices.empty:
            raise ToolException("This date boundary leaves too few rows for training or testing. Adjust the test share or scope.")
        boundary = (choices - len(dates) * (1 - test_size)).abs().idxmin()
        train = dates[dates <= boundary]
        test = dates[dates > boundary]
        evidence.update(time_column=time_column, invalid_time_rows=len(target) - len(dates),
                        train_from=train.iloc[0].isoformat(), train_to=train.iloc[-1].isoformat(),
                        test_from=test.iloc[0].isoformat(), test_to=test.iloc[-1].isoformat())
        train, test = target.loc[train.index], target.loc[test.index]
    elif split == "random":
        test_count = int(np.ceil(len(target) * test_size))
        if min(test_count, len(target) - test_count) < 2:
            raise ToolException("Leave at least two records in each of the train and test sets.")
        train, test = train_test_split(target, test_size=test_size, random_state=RANDOM_SEED)
    else:
        if len(target) > MAX_MODEL_ROWS:
            target = target.sample(MAX_MODEL_ROWS, random_state=RANDOM_SEED)
        folds = KFold(n_splits=VALIDATION_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        return [(target.iloc[train], target.iloc[test]) for train, test in folds.split(target)], evidence, target

    if len(train) + len(test) > MAX_MODEL_ROWS:
        test_limit = min(len(test), max(2, round(MAX_MODEL_ROWS * len(test) / (len(train) + len(test)))))
        train_limit = min(len(train), MAX_MODEL_ROWS - test_limit)
        train, test = train.sample(train_limit, random_state=RANDOM_SEED), test.sample(test_limit, random_state=RANDOM_SEED)
    return [(train, test)], evidence, pd.concat([train, test])


def encoded_frame(values, index, numeric, categorical):
    return pd.DataFrame({
        **{name: values[name].loc[index].astype(float) for name in numeric},
        **{name: ("value:" + values[name].loc[index].astype("string")).fillna("missing").astype(object) for name in categorical},
    }, index=index)


def assess_model(
    runtime, target_column, candidate_columns, method, *,
    model="random_forest", split="cross_validation", test_size=0.2, time_column=None, inputs=None,
):
    frame = working_frame(runtime.context, runtime.state)
    target = numeric_column(frame, target_column).dropna()
    available_rows = len(target)
    if available_rows < MIN_VALID_ROWS:
        raise ToolException(
            f"Prediction needs at least {MIN_VALID_ROWS} usable target rows. The working data have {available_rows}. "
            "Descriptive associations are available through rank_relationships."
        )
    if target.nunique() < 2:
        raise ToolException("The numeric target must vary across rows.")
    values, numeric, categorical = prediction_candidates(frame, target_column, target, candidate_columns, method, runtime)
    if inputs is not None:
        if not inputs or set(inputs) != set(values):
            excluded = ", ".join(sorted(set(inputs) - set(values)))
            raise ToolException(f"Use only eligible inputs. Unusable supplied fields: {excluded or '(none)'}."
                                " Review the fields or provide other known values before estimating.")
        for name in numeric:
            try:
                value = float(inputs[name])
            except (TypeError, ValueError) as error:
                raise ToolException(f"Supply a numeric value for {name}.") from error
            if not np.isfinite(value):
                raise ToolException(f"Supply a finite numeric value for {name}.")
    partitions, details, used_target = validation_partitions(frame, target, split, test_size, time_column)
    if any(train.nunique() < 2 for train, _ in partitions):
        raise ToolException("Training target values do not vary in at least one split.")
    features = encoded_frame(values, used_target.index, numeric, categorical)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    preprocess = ColumnTransformer([
        ("numeric", make_pipeline(imputer, StandardScaler()) if model in {"linear_regression", "ridge"} else imputer, numeric),
        ("categorical", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2, max_categories=MAX_CATEGORIES), categorical),
    ])
    pipeline = make_pipeline(preprocess, clone(MODELS[model]))
    sampled = len(used_target) < available_rows - details.get("invalid_time_rows", 0)

    def progress(stage, fold=None):
        runtime.stream_writer({
            "event": "prediction_progress", "stage": stage, "fold": fold, "folds": len(partitions),
            "rows": len(used_target), "available_rows": available_rows, "target": target_column,
            "model": model, "split": split, "sampled": sampled, "tool_call_id": runtime.tool_call_id,
        })

    validation, scores = [], []
    for fold, (train, test) in enumerate(partitions, start=1):
        progress("fit", fold)
        x_test = features.loc[test.index]
        fitted = clone(pipeline).fit(features.loc[train.index], train)
        predicted = fitted.predict(x_test)
        baseline = float(train.median())
        mae = float(mean_absolute_error(test, predicted))
        baseline_mae = float(mean_absolute_error(test, np.full(len(test), baseline)))
        validation.append((x_test, test, mae, baseline_mae, fitted))
        scores.append({"fold": fold, "train_rows": len(train), "test_rows": len(test), "model_mae": mae,
                       "baseline_value": baseline, "baseline_mae": baseline_mae,
                       "rmse": float(np.sqrt(mean_squared_error(test, predicted))),
                       "r2": float(r2_score(test, predicted)) if test.nunique() > 1 else None})

    weights = [score["test_rows"] for score in scores]
    evidence = {
        "target": target_column, "rows": len(used_target), "available_rows": available_rows,
        "sample_seed": RANDOM_SEED if sampled else None, "folds": len(partitions), "fields": list(features.columns),
        "model": model, "model_params": MODELS[model].get_params(),
        "validation_mae": float(np.average([score["model_mae"] for score in scores], weights=weights)),
        "baseline_mae": float(np.average([score["baseline_mae"] for score in scores], weights=weights)),
        "baseline_wins": sum(score["model_mae"] < score["baseline_mae"] for score in scores),
        "split_details": details, "scores": scores,
        "preprocessing": {"numeric_imputation": "median", "numeric_standardization": model in {"linear_regression", "ridge"},
                          "category_encoding": "one_hot", "max_categories": MAX_CATEGORIES, "fit_scope": "training_only"},
    }
    if inputs is not None:
        train = partitions[0][0]
        fitted = validation[0][-1]
        if split == "cross_validation":
            progress("refit")
            train = used_target
            fitted = clone(pipeline).fit(features, train)
        encoder = fitted.named_steps["columntransformer"].named_transformers_["categorical"]
        unseen = [name for name, known in zip(categorical, encoder.categories_ if categorical else ())
                  if f"value:{str(inputs[name]).strip()}" not in known]
        row = pd.DataFrame([{
            **{name: float(inputs[name]) for name in numeric},
            **{name: f"value:{str(inputs[name]).strip()}" for name in categorical},
        }], columns=features.columns)
        evidence["estimate"] = {"value": float(fitted.predict(row)[0]), "inputs": inputs,
                                "unseen_categories": unseen, "training_rows": len(train)}
    return validation, evidence, progress


def evaluation_summary(evidence):
    rows, available = evidence["rows"], evidence["available_rows"]
    split = evidence["split_details"]["split"]
    label = f"{evidence['folds']}-fold validation" if split == "cross_validation" else f"{split} train/test split"
    line = f"{label} | {rows:,} rows"
    if evidence["sample_seed"] is not None:
        line += f" sampled from {available:,} usable target rows (seed {evidence['sample_seed']})"
    return f"{line} | MAE {evidence['validation_mae']:,.4g} vs. median baseline {evidence['baseline_mae']:,.4g}"


def prediction_result(evidence):
    verdict = "beat" if evidence["validation_mae"] < evidence["baseline_mae"] else "did not beat"
    scores = []
    for score in evidence["scores"]:
        r2 = f"{score['r2']:.4g}" if score["r2"] is not None else "undefined"
        scores.append(f"Split {score['fold']}: train {score['train_rows']:,}, test {score['test_rows']:,}, "
                      f"MAE {score['model_mae']:.6g}, median-baseline MAE {score['baseline_mae']:.6g}, "
                      f"RMSE {score['rmse']:.6g}, R² {r2}.")
    details = evidence["split_details"]
    time = (f" Date field {details['time_column']}: training through {details['train_to']}, "
            f"testing from {details['test_from']}. Excluded {details['invalid_time_rows']:,} invalid dates.") if "time_column" in details else ""
    return (
        f"Model {evidence['model']} for {evidence['target']}. {evaluation_summary(evidence)}. "
        f"The model {verdict} the training-median baseline. "
        f"Input fields: {', '.join(evidence['fields'])}. "
        f"Requested test share: {details['test_size']}. Split seed: {details['split_seed']}." + time + "\n" +
        "\n".join(scores) + "\nAll preprocessing is fitted on training rows. MAE and RMSE use the target's units."
    )


@tool(response_format="content_and_artifact", parse_docstring=True)
def evaluate_prediction(
    target_column: str, runtime: ToolRuntime[AnalysisContext], candidate_columns: list[str] | None = None,
    model: Model = "random_forest", split: Split = "random", test_size: float = 0.2, time_column: str | None = None,
) -> tuple[str, dict]:
    """Evaluate a supported regression model on held-out records. No field importance or individual estimate.

    Args:
        target_column: Exact numeric target column name.
        candidate_columns: Exact input column names, or null to review eligible fields.
        model: decision_tree (a single shallow tree), linear_regression, ridge, or random_forest.
        split: random train/test, chronological train/test, or cross_validation using three shuffled folds.
        test_size: Requested fraction for the test set for random or chronological split.
        time_column: Exact date column for chronological validation, with whole dates kept together.
    """
    _, evidence, _ = assess_model(runtime, target_column, candidate_columns, "evaluate_prediction",
                                  model=model, split=split, test_size=test_size, time_column=time_column)
    return prediction_result(evidence), {"prediction_evaluation": evidence}


@tool(response_format="content_and_artifact", parse_docstring=True)
def predict_value(
    target_column: str, inputs: dict[str, str | int | float | bool], runtime: ToolRuntime[AnalysisContext],
    model: Model = "random_forest", split: Split = "random", test_size: float = 0.2, time_column: str | None = None,
) -> tuple[str, dict]:
    """Estimate a numeric target for one record from explicitly provided values, with held-out error.

    Args:
        target_column: Exact numeric target column name.
        inputs: Exact feature column names mapped to user-supplied values. Only these fields are used.
        model: decision_tree, linear_regression, ridge, or random_forest.
        split: random train/test, chronological train/test, or cross_validation using three shuffled folds.
        test_size: Requested test fraction for random or chronological split.
        time_column: Exact date column for chronological validation.
    """
    _, evidence, _ = assess_model(runtime, target_column, list(inputs), "predict_value", model=model,
                                  split=split, test_size=test_size, time_column=time_column, inputs=inputs)
    estimate = evidence["estimate"]
    warning = f" Categories unseen in the estimate's training data: {', '.join(estimate['unseen_categories'])}." if estimate["unseen_categories"] else ""
    text = (f"Estimated {target_column}: {estimate['value']:.8g} using {estimate['training_rows']:,} training rows.{warning}\n"
            f"{prediction_result(evidence)}\nThe held-out error describes typical performance, not an interval for this record.")
    return text, {"prediction_evaluation": evidence}
