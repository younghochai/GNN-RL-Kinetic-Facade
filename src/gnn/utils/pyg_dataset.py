import logging
import pandas as pd
from typing import List, Tuple
import tqdm.auto as tqdm

# utils
from utils.pandas_utils import load_data
from utils.preprocess import CyclicHourEmbedder
from scaler.feature_scaler import GraphTargetScaler

# torch
import torch
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_node_features(
    node_df: pd.DataFrame,
    epw_df: pd.DataFrame,
    start_node_id: int,
    end_node_id: int,
    sun_node_id: int,
    use_node_coord: bool = True,
    use_sun_coord: bool = True,
    return_global_features: bool = True,
):

    # TODO
    node_df['Date_월일'] = pd.to_datetime(node_df['Date_월일'])
    node_df['Date_시간'] = pd.to_datetime(node_df['Date_시간'])

    node_df['hour'] = node_df['Date_시간'].apply(lambda x: x.hour)
    node_df[['hour_sin', 'hour_cos']] = CyclicHourEmbedder().fit_transform(node_df['hour'])

    tgt_hour = node_df['Date_시간'].iloc[0].hour
    tgt_date = node_df['Date_월일'].iloc[0]
    tgt_month = tgt_date.month
    tgt_day = tgt_date.day
    tgt_epw_df = epw_df[(epw_df['Month'] == tgt_month) & (epw_df['Day'] == tgt_day) & (epw_df['Hour'] == tgt_hour)]
    tgt_epw_df = tgt_epw_df[['DNI', 'DHI']]
    DNI, DHI = tgt_epw_df.iloc[0].tolist()

    global_features = {}
    node_features = {}
    for row in node_df.itertuples():
        if start_node_id <= row.Node_ID <= end_node_id:
            if use_node_coord:
                node_features[row.Node_ID] = [
                    row.Node_X, row.Node_Y, row.Node_Z, row.열림각도
                ]
            else:
                node_features[row.Node_ID] = [
                    row.열림각도
                ]
        if row.Node_ID == sun_node_id:
            if use_sun_coord:
                global_features[row.Node_ID] = [
                    row.Node_X, row.Node_Y, row.Node_Z, row.hour_sin, row.hour_cos, DNI, DHI
                ]
            else:
                global_features[row.Node_ID] = [
                    row.hour_sin, row.hour_cos, DNI, DHI
                ]
    if return_global_features:
        return node_features, global_features
    else:
        return node_features


def get_edge_attr(edge_df: pd.DataFrame, use_edge_type: bool = False):
    edge_attr = [
        [row.Edge_Type if use_edge_type else 1, row.Distance]
        for row in edge_df.itertuples()
    ]
    return edge_attr


def get_edge_list(edge_df: pd.DataFrame):
    edge_list = [
        edge_df['Source'].tolist(),
        edge_df['Target'].tolist()
    ]
    return edge_list


def create_graph_data(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    epw_df: pd.DataFrame,
    target_columns: List[str] = ['운동장_태양복사열'],
    start_node_id: int = 2,
    end_node_id: int = 1850,
    sun_node_id: int = 0,
    pitch_node_id: int = 1,
    multi_task: bool = False,
    use_sun_coord: bool = True,
    use_node_coord: bool = True,
    target_scaler=None,
) -> Data:
    if multi_task and len(target_columns) <= 1:
        raise ValueError("target_column must be a list of length greater than 1")

    if not multi_task and len(target_columns) > 1:
        raise ValueError("target_column must be a single column")

    # Extract node features and global features
    node_features, global_features = get_node_features(
        node_df=node_df,
        epw_df=epw_df,
        start_node_id=start_node_id,
        end_node_id=end_node_id,
        sun_node_id=sun_node_id,
        use_sun_coord=use_sun_coord,
        use_node_coord=use_node_coord,
        return_global_features=True,
    )

    # Create edge list
    edge_list = get_edge_list(edge_df)
    edge_list = torch.tensor(edge_list, dtype=torch.long)

    # Create edge attributes
    edge_attr = get_edge_attr(edge_df)

    # Create node ID mapping
    node_ids = sorted(node_features.keys())
    id_map = {old: new for new, old in enumerate(node_ids)}

    # Map edge indices to new node indices

    edge_index = torch.tensor([
        [id_map[n.item()] for n in edge_list[0]],
        [id_map[n.item()] for n in edge_list[1]]
    ], dtype=torch.long)

    # Create node features tensor
    x = torch.tensor(
        [node_features[nid] for nid in node_ids], dtype=torch.float
    )
    global_x = torch.tensor(
        [global_features[sun_node_id]], dtype=torch.float
    )

    # Get target value (graph-level)
    target_value = node_df[target_columns].median()

    # Apply target scaling if scaler is provided
    if target_scaler is not None:
        if target_scaler.is_fitted:
            # Convert to numpy array for scaling
            target_array = target_value.values.reshape(1, -1)
            scaled_target = target_scaler.transform(target_array)
            scaled_target_value = pd.Series(scaled_target.flatten(), index=target_columns)
        else:
            logger.warning("Target scaler is not fitted. Using original target values.")

    # Create PyG Data object
    if target_scaler is not None:
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            y=torch.tensor([target_value], dtype=torch.float),
            scaled_y=torch.tensor([scaled_target_value], dtype=torch.float),
            global_x=global_x,
        )
    else:
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            y=torch.tensor([target_value], dtype=torch.float),
            global_x=global_x,
        )
    return data


def create_multi_graph_dataset(
    node_file_paths: List[str],
    edge_file_paths: List[str],
    epw_file_path: str,
    target_columns: List[str] = ['운동장_태양복사열'],
    start_node_id: int = 2,
    end_node_id: int = 1850,
    sun_node_id: int = 0,
    pitch_node_id: int = 1,
    multi_task: bool = False,
    use_sun_coord: bool = True,
    use_node_coord: bool = True,
    target_scaler: GraphTargetScaler = None,
) -> Tuple[List[Data], List[str]]:
    fail_file_paths = []
    if len(node_file_paths) != len(edge_file_paths):
        raise ValueError(
            "Number of node files must match number of edge files"
        )

    data_list = []
    epw_df = pd.read_csv(epw_file_path)

    time_df = []
    for node_path, edge_path in tqdm.tqdm(zip(node_file_paths, edge_file_paths), total=len(node_file_paths)):
        logger.debug(f"Created graph data for {node_path} and {edge_path}")
        # Load data

        graph_dataframe = load_data(node_path, edge_path, start_node_id, end_node_id)
        if graph_dataframe is None:
            fail_file_paths.append(node_path)
            logger.info(f"Fail files: {node_path}")
            continue
        node_df = graph_dataframe.node_df
        edge_df = graph_dataframe.edge_df

        data = create_graph_data(
            node_df,
            edge_df,
            epw_df,
            target_columns=target_columns,
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            sun_node_id=sun_node_id,
            pitch_node_id=pitch_node_id,
            multi_task=multi_task,
            use_sun_coord=use_sun_coord,
            use_node_coord=use_node_coord,
            target_scaler=target_scaler,
        )
        # data = preprocess_data(data)
        logger.debug(f"Data: {data}")
        data_list.append(data)
        time_df.append((node_df['Date_월일'][0], node_df['Date_시간'][0]))

    logger.info(f"Fail files n: {len(fail_file_paths)}")
    return data_list, fail_file_paths, time_df
