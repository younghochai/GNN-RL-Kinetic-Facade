import pandas as pd
import numpy as np
from typing import List
# from torch_geometric.data import Data  # Unused import


class OneHotEmbedder:
    def __init__(self, handle_unknown: str = 'ignore'):
        """
        Initialize the OneHotEmbedder.

        Args:
            handle_unknown: How to handle unknown categories during transform.
                           'ignore': ignore unknown categories (default)
                           'error': raise error for unknown categories
        """
        self.categories_: List[str] = []
        self.handle_unknown = handle_unknown
        self.is_fitted = False

    def fit(self, data: pd.Series) -> 'OneHotEmbedder':
        """
        Fit the encoder on the training data.

        Args:
            data: pandas Series containing categorical data

        Returns:
            self: Returns self for method chaining
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input must be a pandas Series")

        # Get unique categories, excluding NaN values
        self.categories_ = data.dropna().unique().tolist()
        self.is_fitted = True

        return self
    
    def transform(self, data: pd.Series) -> pd.DataFrame:
        """
        Transform the data using the fitted encoder.

        Args:
            data: pandas Series containing categorical data

        Returns:
            pd.DataFrame: One-hot encoded data
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before transform()")

        if not isinstance(data, pd.Series):
            raise ValueError("Input must be a pandas Series")

        # Create one-hot encoded DataFrame
        result = pd.DataFrame(index=data.index)

        for category in self.categories_:
            if data.name:
                column_name = f"{data.name}_{category}"
            else:
                column_name = str(category)
            result[column_name] = (data == category).astype(int)

        # Handle unknown categories
        if self.handle_unknown == 'error':
            unknown_categories = set(data.dropna()) - set(self.categories_)
            if unknown_categories:
                raise ValueError(
                    f"Unknown categories found: {unknown_categories}"
                )
        elif self.handle_unknown == 'ignore':
            # Unknown categories are simply ignored (not encoded)
            pass

        return result
    
    def fit_transform(self, data: pd.Series) -> pd.DataFrame:
        """
        Fit the encoder and transform the data in one step.

        Args:
            data: pandas Series containing categorical data

        Returns:
            pd.DataFrame: One-hot encoded data
        """
        return self.fit(data).transform(data)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names for the one-hot encoded columns.

        Returns:
            List[str]: List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before get_feature_names()")

        return [f"category_{cat}" for cat in self.categories_]
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        """
        Inverse transform one-hot encoded data back to categorical.

        Args:
            data: One-hot encoded DataFrame

        Returns:
            pd.Series: Original categorical data
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before inverse_transform()")

        # Find the category with the highest value (1) for each row
        result = pd.Series(index=data.index, dtype=object)

        for idx in data.index:
            row = data.loc[idx]
            # Find columns with value 1
            active_categories = row[row == 1].index.tolist()

            if len(active_categories) == 1:
                # Extract category name from column name
                col_name = active_categories[0]
                if '_' in col_name:
                    # Get part after first underscore
                    category = col_name.split('_', 1)[1]
                else:
                    category = col_name
                result.loc[idx] = category
            elif len(active_categories) == 0:
                result.loc[idx] = np.nan
            else:
                # Multiple categories active - shouldn't happen in one-hot
                result.loc[idx] = np.nan

        return result


class CyclicHourEmbedder:
    """
    Cyclic encoder for hour data (0-23).
    Converts hour values to sine and cosine components for cyclic representation.
    """
    
    def __init__(self):
        """Initialize the CyclicHourEmbedder."""
        self.is_fitted = False
    
    def fit(self, data: pd.Series) -> 'CyclicHourEmbedder':
        """
        Fit the encoder (no actual fitting needed for cyclic encoding).

        Args:
            data: pandas Series containing hour values (0-23)

        Returns:
            self: Returns self for method chaining
        """
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.Series) -> pd.DataFrame:
        """
        Transform hour data to cyclic representation.

        Args:
            data: pandas Series containing hour values (0-23)

        Returns:
            pd.DataFrame: DataFrame with 'hour_sin' and 'hour_cos' columns
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before transform()")
        
        if not isinstance(data, pd.Series):
            raise ValueError("Input must be a pandas Series")
        
        # Check for NaN values
        if data.isna().any():
            nan_indices = data[data.isna()].index.tolist()
            raise ValueError(f"NaN values found at indices: {nan_indices}")
        
        # Check for out-of-range values
        out_of_range = data[(data < 0) | (data > 23)]
        if not out_of_range.empty:
            out_indices = out_of_range.index.tolist()
            out_values = out_of_range.tolist()
            raise ValueError(f"Out-of-range values found at indices {out_indices}: {out_values}. Values must be between 0 and 23.")
        
        # Create cyclic encoding
        result = pd.DataFrame(index=data.index)
        
        # Calculate radians: 2 * pi * hour / 24
        radians = 2 * np.pi * data / 24
        
        # Create sine and cosine components
        result['hour_sin'] = np.sin(radians)
        result['hour_cos'] = np.cos(radians)
        
        return result
    
    def fit_transform(self, data: pd.Series) -> pd.DataFrame:
        """
        Fit the encoder and transform the data in one step.

        Args:
            data: pandas Series containing hour values (0-23)

        Returns:
            pd.DataFrame: DataFrame with 'hour_sin' and 'hour_cos' columns
        """
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        """
        Inverse transform cyclic representation back to hour values.

        Args:
            data: DataFrame with 'hour_sin' and 'hour_cos' columns

        Returns:
            pd.Series: Original hour values (0-23)
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before inverse_transform()")
        
        required_cols = ['hour_sin', 'hour_cos']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain {required_cols} columns")
        
        # Calculate angle from sine and cosine
        angle = np.arctan2(data['hour_sin'], data['hour_cos'])
        
        # Convert back to hour: angle * 24 / (2 * pi)
        hour = (angle * 24) / (2 * np.pi)
        
        # Normalize to 0-23 range
        hour = hour % 24
        
        # Handle NaN values
        nan_mask = data['hour_sin'].isna() | data['hour_cos'].isna()
        hour = hour.where(~nan_mask, np.nan)
        
        return pd.Series(hour, index=data.index, name='hour')
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names for the cyclic encoded columns.

        Returns:
            List[str]: List of feature names ['hour_sin', 'hour_cos']
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before get_feature_names()")
        
        return ['hour_sin', 'hour_cos']


def create_onehot_embedder(data: pd.Series,
                           handle_unknown: str = 'ignore') -> OneHotEmbedder:
    """
    Convenience function to create and fit a OneHotEmbedder.

    Args:
        data: pandas Series containing categorical data
        handle_unknown: How to handle unknown categories ('ignore' or 'error')

    Returns:
        OneHotEmbedder: Fitted one-hot embedder
    """
    embedder = OneHotEmbedder(handle_unknown=handle_unknown)
    return embedder.fit(data)


def create_cyclic_hour_embedder(data: pd.Series) -> CyclicHourEmbedder:
    """
    Convenience function to create and fit a CyclicHourEmbedder.

    Args:
        data: pandas Series containing hour values (0-23)

    Returns:
        CyclicHourEmbedder: Fitted cyclic hour embedder
    """
    embedder = CyclicHourEmbedder()
    return embedder.fit(data)


def preprocess_data(data: pd.DataFrame):
    return data


if __name__ == "__main__":
    # Test CyclicHourEmbedder
    import pandas as pd

    # Test with valid data
    valid_hours = pd.Series([0, 6, 12, 18, 23, 3], name='hour')
    embedder = create_cyclic_hour_embedder(valid_hours)
    encoded = embedder.transform(valid_hours)
    print("Valid hours:", valid_hours.tolist())
    print("Encoded data:")
    print(encoded)

    # Test with invalid data (NaN)
    try:
        invalid_hours = pd.Series([0, 6, 12, np.nan], name='hour')
        embedder.transform(invalid_hours)
    except ValueError as e:
        print(f"\nNaN error test: {e}")

    # Test with out-of-range data
    try:
        out_of_range_hours = pd.Series([0, 6, 12, 25], name='hour')
        embedder.transform(out_of_range_hours)
    except ValueError as e:
        print(f"Out-of-range error test: {e}")

    # Inverse transform
    decoded = embedder.inverse_transform(encoded)
    print(f"\nDecoded hours: {decoded.tolist()}")

    # Test OneHotEmbedder
    categories = pd.Series(['A', 'B', 'A', 'C', 'B'], name='category')
    onehot_embedder = create_onehot_embedder(categories)
    onehot_encoded = onehot_embedder.transform(categories)
    print("\nOne-hot encoded data:")
    print(onehot_encoded)

