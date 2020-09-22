import logging
import os
import pandas as pd
import pprint
import requests

STATS_ENGINE_SERVICE_URI = "https://api.prod.optimizely.com/stats-engine/v0/batch"

logger = logging.getLogger()

def build_ses_headers(optimizely_session_token):
    return {
        "Content-Type": "application/json",
        "account": "0",
        "Authorization": f"Bearer {optimizely_session_token}"
    }

def build_ses_json(metric_observations_dfs, reference_variation_id=None):
    if reference_variation_id is None:
        reference_variation_id = metric_observations_dfs[0].reference_variation_id[0]
    
    metric_observations_dfs = [
        # Convert datetimes to unix timestamps
        df.assign(interval_timestamp=df.interval_timestamp.astype(int) / 10**9)
        for df in metric_observations_dfs
    ]
    
    return {
        "config": {
            "reference_variation_id": str(reference_variation_id),
            "use_stats_resets": True,
        },
        "metrics": [
            {
                "config": {
                    "is_binary": False,
                },
                "data": obs_df.to_dict("records")
            }
            for obs_df in metric_observations_dfs
        ]
    }

def make_ses_request(metric_observations_dfs, reference_variation_id, optimizely_session_token):
    request_headers = build_ses_headers(optimizely_session_token)
    request_data = build_ses_json(metric_observations_dfs, reference_variation_id)
    
    logger.info(f"Making request to Stats Engine Service at {STATS_ENGINE_SERVICE_URI}")
    logger.info("")
    logger.info(f"Headers: {pprint.pformat(request_headers)}")
    logger.info("")
    logger.info(f"JSON: {pprint.pformat(request_data)}")
    logger.info("")
    logger.info("")
    
    ses_response = requests.post(
        STATS_ENGINE_SERVICE_URI, 
        headers=request_headers, 
        json=request_data
    )
    
    display(request_data)

    if ses_response.status_code != 200:
        raise Exception(f"Error: received {ses_response.status_code} from Stats Engine Service ({STATS_ENGINE_SERVICE_URI}): {ses_response.text}")
    
    logger.info(f"Response JSON: {pprint.pformat(ses_response.json())}")

    sequential_stats_dfs = [pd.DataFrame(metric_json) for metric_json in ses_response.json()]
    
    for df in sequential_stats_dfs:
        df.interval_timestamp = pd.to_datetime(df.interval_timestamp, unit='s')
    
    return sequential_stats_dfs
    