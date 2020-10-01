import json
from typing import Dict, List, Union

import numpy as np
import pandas as pd

_TICK_MARK = "&#10003;"


class DefaultValues:

    # Strings
    NAN_STR = "--"


class MetricColumnTypes:
    AVERAGE_VALUE = "Average Value"
    BASELINE = "Baseline?"
    CONFIDENCE_INTERVAL = "Confidence Interval"
    CREDIBLE_INTERVAL_095 = "0.95 Credible Interval"
    DIFFERENCE = "Difference"
    HIGHLIGHT = "Highlight"
    IMPROVEMENT = "Improvement"
    IS_SIGNIFICANT = "Is Significant?"
    LIFT_POINT_ESTIMATE = "Lift Point Estimate"
    LOWER_IS_BETTER = "Lower Is Better"
    MEAN = "Mean"
    MINIMUM_SAMPLE_SIZE = "Minimum Sample Size"
    NEGATIVE = "Negative"
    NEGATIVE_PROBABILITY = "Probability worse than Control"
    NEUTRAL = "Neutral"
    P_VALUE = "P-Value"
    POSITIVE = "Positive"
    POSITIVE_PROBABILITY = "Probability better than Control"
    RATE = "Rate"
    REJECTED = "Rejected"
    STATISTICAL_SIGNIFICANCE = "Statistical Significance"
    TIMESTAMP = "Timestamp"
    UNITS = "Subjects"
    VALUE = "Value"
    VARIATION = "Variation"
    ZERO_PROBABILITY = "Probability equal to Control"


class Constants:
    """Various constants used in stats and analysis."""

    # Stats.
    ALPHA = "alpha"
    CI_LOWER = "corrected_conf_interval_lower"
    CI_UPPER = "corrected_conf_interval_upper"
    CONVERSION_RATE = "cvr"
    DOF = "degrees_of_freedom"
    FREQUENTIST = "frequentist"
    POINT_ESTIMATE = "lift_estimate"
    POWER = "power"
    P_VALUE = "corrected_p_value"
    REJECTED = "rejected"
    STANDARDIZED_EFFECT_SIZE = "standardized_effect_size"
    STATISTICAL_SIGNIFICANCE = "stats_sig"
    # Bayesian.
    BAYES_FACTOR = "bayes_factor"
    SIGNIFICANCE = "significance"
    IS_SIGNIFICANT = "is_significant"
    POSTERIOR_PROBABILITY = "posterior_probability"
    PRIOR_PROBABILITY = "prior_probability"
    PRIOR_VARIANCE = "prior_variance"

    # Inputs.
    IS_BINARY = "is_binary"
    IS_RELATIVE = "is_relative"
    METRIC_NAME = "metric"
    REFERENCE_VARIATION_ID = "reference_variation_id"
    STATISTIC_TYPE = "statistic_type"
    STRATA = "strata"
    TIMESTAMP = "interval_timestamp"
    UNIT_COUNT = "unit_count"
    UNIT_ID = "visitorId"
    UNIT_OBSERVATION = "unit_observation"
    UNIT_OBSERVATION_SUM = "unit_observation_sum"
    UNIT_OBSERVATION_SUM_OF_SQUARES = "unit_observation_sum_of_squares"
    VARIATION = "variation"
    VARIATION_ID = "variation_id"

    # Derived.
    MIN_SAMPLE_SIZE_CONTROL = "min_sample_size_control"
    MIN_SAMPLE_SIZE_TREATMENT = "min_sample_size_treatment"
    CUMULATIVE_UNIT_COUNT_CONTROL = "cumulative_unit_count_control"
    CUMULATIVE_UNIT_COUNT_TREATMENT = "cumulative_unit_count_treatment"
    CUMULATIVE_UNIT_OBSERVATION_CONTROL = "cumulative_unit_observation_control"
    CUMULATIVE_UNIT_OBSERVATION_TREATMENT = "cumulative_unit_observation_treatment"
    CUMULATIVE_MEAN_CONTROL = "cumulative_mean_control"
    CUMULATIVE_MEAN_TREATMENT = "cumulative_mean_treatment"
    SAMPLE_VARIANCE_CONTROL = "sample_variance_control"
    SAMPLE_VARIANCE_TREATMENT = "sample_variance_treatment"


def isnan(x):
    return x is None or np.isnan(x)


def fmt_difference(x, is_binary, is_relative=True):
    if isnan(x):
        return DefaultValues.NAN_STR

    if is_relative:
        return f"{x * 100:0.2f}%"

    if is_binary:
        return f"{x * 100:0.2f}pp"

    return f"{x:0.2f}"


def fmt_mean(x, is_binary):
    if isnan(x):
        return DefaultValues.NAN_STR

    if is_binary:
        return fmt_percent(x)

    return f"{x:0.2f}"


def fmt_number(x):
    return DefaultValues.NAN_STR if isnan(x) else f"{x:0.0f}"


def fmt_percent(x):
    return DefaultValues.NAN_STR if isnan(x) else f"{x * 100:0.2f}%"


def fmt_probability(x):
    return DefaultValues.NAN_STR if isnan(x) else f"{x:0.3f}"


def fmt_pval(x, floor=0.01):
    if isnan(x):
        return DefaultValues.NAN_STR

    if x < floor:
        return f"<{floor}"

    return f"{x:0.2f}"


def fmt_conf_int(x):
    return x if not pd.isna(x) else DefaultValues.NAN_STR


def fmt_stat_sig(x, ceil=0.99):
    if isnan(x):
        return DefaultValues.NAN_STR

    if x > ceil:
        return f">{fmt_percent(ceil)}"

    return fmt_percent(x)


def _improvement_status(rejected, improvement, lower_is_better):
    """
    Highlights overview table rows for corresponding columns according to improvement.

    Parameters
    ----------
    rejected : bool
    improvement : number
    lower_is_better : bool

    Returns
    -------
    str
        Neutral: If the improvement is a NAN string (a.k.a the Original Variation).
        Positive: If the improvement is positive and lower_is_better is False or if the improvement is negative and lower_is_better is True.
        Negative: Otherwise

    """
    if isinstance(improvement, str) or np.isnan(improvement):
        return MetricColumnTypes.NEUTRAL

    if rejected:
        if (improvement < 0) == lower_is_better:
            return MetricColumnTypes.POSITIVE
        return MetricColumnTypes.NEGATIVE
    return MetricColumnTypes.NEUTRAL


def compute_se_metric_overview_table(
    observations_timeseries: pd.DataFrame,
    statistics: pd.DataFrame,
    reference_variation_id: Union[int, str],
    variation_names: Dict[Union[int, str], str] = None,
) -> dict:
    """Computes a dictionary that contains summary table of experiment results per metric.

    Parameters
    ----------
    observations_timeseries : DataFrame
        DataFrame containing aggregated unit counts and observations per time-bucket.
    statistics : DataFrame
        DataFrame of statistics obtained by analyzing experiment data using Optimizely's Stats Engine.
    reference_variation_id : int or string
        variation_id of the variation to be used as the "baseline"
    variation_names: Dict[int or string, string]
        a dictionary mapping variation_id values to human-readable names

    Returns
    -------
    dict
        Dictionary that contains the overview table for all of the metrics for the given experiment.
        To be dumped to JSON and consumed by the corresponding HTML template (SE_METRIC_OVERVIEW_TABLE).

    Examples
    --------
    from IPython.display import display, HTML

    metric_overview_html = render_se_metric_overview_table(
        observations_timeseries=observations_timeseries,
        statistics=statistics,
        reference_variation_id=reference_variation_id,
        variation_names=variation_names
    )

    display(HTML(metric_overview_html))

    """
    reference_variation = reference_variation_id
    variations = statistics.variation_id.unique()
    treatment_variations = [
        v for v in variations if v != reference_variation_id
    ]

    observations_timeseries = observations_timeseries.reset_index().set_index(
        [Constants.VARIATION_ID, Constants.TIMESTAMP]
    )
    statistics = statistics.reset_index().set_index(
        [Constants.VARIATION_ID, Constants.TIMESTAMP]
    )

    # Create dict containing all metrics details to be consumed by the template.
    metrics_overview_table = {}

    report = observations_timeseries.sum(level=Constants.VARIATION_ID).loc[
        list(variations),
        [Constants.UNIT_COUNT, Constants.UNIT_OBSERVATION_SUM],
    ]
    report = report.rename(
        columns={
            Constants.UNIT_OBSERVATION_SUM: MetricColumnTypes.VALUE,
            Constants.UNIT_COUNT: MetricColumnTypes.UNITS,
        }
    )
    report.insert(0, MetricColumnTypes.BASELINE, "")
    report.loc[reference_variation, MetricColumnTypes.BASELINE] = _TICK_MARK

    is_binary_metric = False
    is_relative_difference = True

    report[MetricColumnTypes.AVERAGE_VALUE] = (
        report[MetricColumnTypes.VALUE] / report[MetricColumnTypes.UNITS]
    )
    report[MetricColumnTypes.AVERAGE_VALUE] = report[
        MetricColumnTypes.AVERAGE_VALUE
    ].apply(lambda x: f"{x:0.2f}")

    for variation_id in treatment_variations:
        try:
            stats = statistics.loc[variation_id]
        except KeyError:
            continue

        max_timestamp = stats.index.max()
        latest_stats = stats.loc[max_timestamp]

        if MetricColumnTypes.REJECTED not in report.columns:
            report[MetricColumnTypes.REJECTED] = None
        is_rejected = latest_stats[Constants.REJECTED]
        report.at[variation_id, MetricColumnTypes.REJECTED] = is_rejected

        report.at[
            variation_id, MetricColumnTypes.LOWER_IS_BETTER
        ] = False  # metric.lower_is_better

        if MetricColumnTypes.IMPROVEMENT not in report.columns:
            report[MetricColumnTypes.IMPROVEMENT] = None
        report.at[variation_id, MetricColumnTypes.IMPROVEMENT] = latest_stats[
            Constants.POINT_ESTIMATE
        ]

        ci_lower = latest_stats[Constants.CI_LOWER]
        ci_upper = latest_stats[Constants.CI_UPPER]
        report.at[variation_id, MetricColumnTypes.CONFIDENCE_INTERVAL] = (
            f"[{fmt_difference(ci_lower, is_binary_metric, is_relative_difference)},"
            f" {fmt_difference(ci_upper, is_binary_metric, is_relative_difference)}]"
        )
        report.at[variation_id, MetricColumnTypes.P_VALUE] = latest_stats[
            Constants.P_VALUE
        ]
        report.at[variation_id, MetricColumnTypes.STATISTICAL_SIGNIFICANCE] = (
            1 - report.loc[variation_id, MetricColumnTypes.P_VALUE]
        )
        report.at[variation_id, MetricColumnTypes.HIGHLIGHT] = _improvement_status(
            report.at[variation_id, MetricColumnTypes.REJECTED],
            report.at[variation_id, MetricColumnTypes.IMPROVEMENT],
            report.at[variation_id, MetricColumnTypes.LOWER_IS_BETTER],
        )

    df = report.reset_index()

    df[MetricColumnTypes.VALUE] = df[MetricColumnTypes.VALUE].apply(
        lambda x: f"{x:0.0f}"
    )
    df[MetricColumnTypes.UNITS] = df[MetricColumnTypes.UNITS].apply(
        lambda x: f"{x:0.0f}"
    )
    df[MetricColumnTypes.IMPROVEMENT] = df[MetricColumnTypes.IMPROVEMENT].apply(
        lambda x: fmt_difference(x, is_binary_metric, is_relative_difference)
    )
    df[MetricColumnTypes.CONFIDENCE_INTERVAL] = df[
        MetricColumnTypes.CONFIDENCE_INTERVAL
    ].apply(fmt_conf_int)
    df[MetricColumnTypes.P_VALUE] = df[MetricColumnTypes.P_VALUE].apply(fmt_pval)
    df[MetricColumnTypes.STATISTICAL_SIGNIFICANCE] = df[
        MetricColumnTypes.STATISTICAL_SIGNIFICANCE
    ].apply(fmt_stat_sig)

    # Replace variation labels if possible
    if variation_names is not None:
        df[Constants.VARIATION_ID] = df[Constants.VARIATION_ID].map(
            lambda v: variation_names[v]
        )

    # Remove 2 columns of Rejected and Lower_Is_Better
    df = df.drop(
        columns=[MetricColumnTypes.REJECTED, MetricColumnTypes.LOWER_IS_BETTER]
    )

    return json.loads(df.to_json(orient="records"))
