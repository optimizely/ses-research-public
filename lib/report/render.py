import os
import pkgutil
import sys
from typing import Dict

import pandas as pd
from lib.report.components import metric_overview_table
from jinja2 import Environment, FunctionLoader

report_module = sys.modules[__name__]

def _load_css_resource() -> str:
    resource = pkgutil.get_data(
        report_module.__name__, os.path.join("static/css", "report.css")
    )
    assert resource is not None
    return resource.decode("utf8")


def _load_html_template(name: str) -> str:
    resource = pkgutil.get_data(
        report_module.__name__, os.path.join("static/templates", name)
    )
    assert resource is not None
    return resource.decode("utf8")


def _get_jinja_environment() -> Environment:
    return Environment(loader=FunctionLoader(_load_html_template))


def _render_report_elements(template: str, **kwargs) -> str:
    css = _load_css_resource()

    return _get_jinja_environment().get_template(template).render(css=css, **kwargs)


def render_se_metric_overview_table(
    statistics: pd.DataFrame,
    observations_timeseries: pd.DataFrame,
    reference_variation_id,
    metric_name,
    variation_names,
    template: str = "se_metric_overview_table.html",
) -> str:
    """Returns an HTML string describing an experiment's metric statistics computed by Optimizely's Stats Engine.

    Parameters
    ----------
    experiment : Experiment
        Metadata about the current experiment (an Optimizely's experiment or a general experiment) to be reported about.
    statistics : DataFrame
        DataFrame of statistics obtained by analyzing experiment data using Optimizely's Stats Engine.
    observations_timeseries : DataFrame
        DataFrame containing aggregated unit counts and observations per time-bucket.
    metric_configs : Dict
        Dictionary which contains the metrics configurations.
        { metric_name: { "is_binary": Boolean, "is_relative": Boolean } }
    metric : Metric
        Object which describes the experiment's metric.
    template: str
        Name of the HTML template of the report table for Stats Engine Analysis.

    Returns
    -------
    str:
        An HTML string describing an experiment's metric statistics computed by Optimizely's Stats Engine.

    """
    overview_table = {
        metric_name :
        metric_overview_table.compute_se_metric_overview_table(
            observations_timeseries=observations_timeseries,
            statistics=statistics,
            reference_variation_id=reference_variation_id,
            variation_names=variation_names
        )
    }
    return _render_report_elements(
        template=template, metric_overview_table=overview_table
    )