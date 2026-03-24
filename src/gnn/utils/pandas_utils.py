import logging
import pandas as pd
from dataclasses import dataclass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphDataFrame:
    node_df: pd.DataFrame
    edge_df: pd.DataFrame


def load_data(node_file_path, edge_file_path, start_node_id, end_node_id):
    node_df = pd.read_csv(node_file_path)
    edge_df = pd.read_csv(edge_file_path)

    if len(node_df) == 0 or len(edge_df) == 0:
        logger.warning(f"No data found for {node_file_path} and {edge_file_path}")
        return None

    if (edge_df[edge_df['Edge_Type'] == 'sun_to_pitchPt']['Distance(m)'] == 0).any():
        logger.warning('Sun to pitchPt distance is 0 ---> Sun Set off ---> file will be passed')
        logger.warning(f"{node_file_path} and {edge_file_path}")
        return None

    node_df = preprocess_node_data(node_df)
    valid_edge_df = preprocess_edge_data(edge_df, start_node_id, end_node_id)

    return GraphDataFrame(node_df, valid_edge_df)


def preprocess_node_data(node_df: pd.DataFrame):
    node_df['열림각도'] = node_df['열림각도'].fillna(0)
    return node_df


def preprocess_edge_data(edge_df: pd.DataFrame, start_node_id, end_node_id):
    # Rename Distance column if needed
    if 'Distance(m)' in edge_df.columns:
        edge_df.rename(columns={'Distance(m)': 'Distance'}, inplace=True)

    # Filter valid edges (panel_to_panel only)
    valid_edge_df = edge_df[edge_df['Edge_Type'] == "panel_to_panel"]
    valid_edge_df = valid_edge_df[
        (valid_edge_df['Source'] >= start_node_id) &
        (valid_edge_df['Source'] <= end_node_id) &
        (valid_edge_df['Target'] >= start_node_id) &
        (valid_edge_df['Target'] <= end_node_id)

        ]
    return valid_edge_df
