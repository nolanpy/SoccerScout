"""
ML model for predicting player transfer values using historical data and career progression.

This module enhances the base ML model with:
1. Time-series analysis of historical transfer values
2. Correlations between performance metrics and value changes
3. Value trajectory predictions
4. Age and position-specific value progression patterns
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import pickle
import json
from datetime import datetime
import logging
import database as db
import ml_model as base_ml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup enhanced logging
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Create file handler
log_file = os.path.join(LOG_DIR, f'transfer_ml_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Configure logger
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

# Paths for saving models and metrics
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
METRICS_DIR = os.path.join(os.path.dirname(__file__), 'metrics')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

class TransferValueModel(base_ml.PlayerValueModel):
    """Enhanced ML model for predicting player transfer values using historical data"""
    
    def __init__(self, model_type="random_forest", position_specific=True, age_adjusted=True, time_series=True):
        """Initialize the transfer value model
        
        Args:
            model_type (str): Type of model to use
            position_specific (bool): Whether to train position-specific models
            age_adjusted (bool): Whether to include age-based adjustments
            time_series (bool): Whether to include time-series features
        """
        # Call the parent class constructor
        super().__init__(model_type=model_type, position_specific=position_specific, age_adjusted=age_adjusted)
        
        # Additional settings specific to transfer value model
        self.time_series = time_series
        self.value_history_metrics = {}
        
        # Value trajectory model (for predicting future values)
        self.trajectory_model = None
        self.trajectory_scaler = StandardScaler()
        
        # Define value trajectory features
        self.trajectory_features = [
            'age', 
            'value_growth_1yr', 
            'value_growth_3yr',
            'goals_per90',
            'assists_per90', 
            'xg_per90', 
            'xa_per90',
            'minutes_played'
        ]
        
        # Map of position to career length and peak attributes
        self.position_career_attributes = {
            'GK': {'typical_peak': 31, 'career_length': 20, 'value_stability': 0.9},
            'CB': {'typical_peak': 29, 'career_length': 18, 'value_stability': 0.85},
            'LB': {'typical_peak': 27, 'career_length': 16, 'value_stability': 0.8},
            'RB': {'typical_peak': 27, 'career_length': 16, 'value_stability': 0.8},
            'CDM': {'typical_peak': 28, 'career_length': 17, 'value_stability': 0.8},
            'CM': {'typical_peak': 27, 'career_length': 16, 'value_stability': 0.75},
            'CAM': {'typical_peak': 26, 'career_length': 15, 'value_stability': 0.7},
            'LW': {'typical_peak': 25, 'career_length': 14, 'value_stability': 0.65},
            'RW': {'typical_peak': 25, 'career_length': 14, 'value_stability': 0.65},
            'CF': {'typical_peak': 26, 'career_length': 15, 'value_stability': 0.7}
        }
    
    def _get_value_history_data(self, seasons=None):
        """Get player data with historical transfer values
        
        Args:
            seasons (list): List of seasons to include
            
        Returns:
            pd.DataFrame: Player data with transfer value history
        """
        # Get data with market value history
        df = db.get_players_with_market_value_history()
        
        if df.empty:
            logger.warning("No player data with market value history found")
            return pd.DataFrame()
            
        # Add value change metrics
        for season_idx in range(1, len(seasons) if seasons else 6):
            prev_season = f"market_value_{seasons[season_idx-1].replace('-', '_')}" if seasons else f"market_value_20{17+season_idx:02d}_20{18+season_idx:02d}"
            curr_season = f"market_value_{seasons[season_idx].replace('-', '_')}" if seasons else f"market_value_20{18+season_idx:02d}_20{19+season_idx:02d}"
            
            # Calculate value change and percent change
            change_col = f"value_change_{curr_season[-9:]}"
            pct_change_col = f"value_pct_change_{curr_season[-9:]}"
            
            # Calculate changes where both values exist
            df[change_col] = np.where(
                (df[prev_season].notnull()) & (df[curr_season].notnull()),
                df[curr_season] - df[prev_season],
                np.nan
            )
            
            # Calculate percent changes (avoiding division by zero)
            df[pct_change_col] = np.where(
                (df[prev_season].notnull()) & (df[curr_season].notnull()) & (df[prev_season] > 0),
                (df[curr_season] - df[prev_season]) / df[prev_season] * 100,
                np.nan
            )
        
        # Filter to selected seasons if provided
        if seasons:
            season_cols = [
                f"market_value_{season.replace('-', '_')}" 
                for season in seasons
            ]
            cols_to_keep = list(set(df.columns) - set([
                col for col in df.columns 
                if col.startswith("market_value_") and col not in season_cols
            ]))
            df = df[cols_to_keep]
        
        return df
    
    def train_value_trajectory_model(self, seasons=None):
        """Train a model to predict future value trajectories
        
        Args:
            seasons (list): List of seasons to include
            
        Returns:
            dict: Metrics for trajectory model
        """
        logger.info("Training value trajectory model...")
        
        # Get data with value history
        df = self._get_value_history_data(seasons)
        
        if df.empty:
            logger.error("No data available for training value trajectory model")
            return None
        
        # Prepare training data
        # We're going to predict next season's value percent change
        # Use the most recent complete season pair (2021-2022 to 2022-2023)
        X_columns = [
            'age', 'goals_per90', 'assists_per90', 'xg_per90', 'xa_per90',
            'minutes_played', 'games_played', 'value_growth_1yr'
        ]
        
        # Make sure all required columns exist
        missing_cols = [col for col in X_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns for value trajectory model: {missing_cols}")
            return None
        
        # Target will be the value percent change from 2022-2023 to 2023-2024
        y_column = 'value_pct_change_2023_2024'
        
        # Filter to rows with non-null values for features and target
        valid_rows = df[X_columns + [y_column]].dropna()
        
        if len(valid_rows) < 10:  # Minimum number of samples for training
            logger.error(f"Insufficient data for training value trajectory model: {len(valid_rows)} valid rows")
            return None
        
        logger.info(f"Training value trajectory model on {len(valid_rows)} players")
        
        # Split data
        X = valid_rows[X_columns]
        y = valid_rows[y_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.trajectory_scaler.fit_transform(X_train)
        X_test_scaled = self.trajectory_scaler.transform(X_test)
        
        # Train the model
        if self.model_type == "random_forest":
            self.trajectory_model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == "gradient_boosting":
            self.trajectory_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            self.trajectory_model = Ridge(alpha=1.0)
        
        self.trajectory_model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred_train = self.trajectory_model.predict(X_train_scaled)
        y_pred_test = self.trajectory_model.predict(X_test_scaled)
        
        metrics = {
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "test_r2": r2_score(y_test, y_pred_test),
            "data_size": len(valid_rows),
            "feature_count": len(X_columns)
        }
        
        # Feature importance
        if hasattr(self.trajectory_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_columns,
                'importance': self.trajectory_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Value trajectory model - Top features:")
            for i, row in feature_importance.head(5).iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.4f}")
                
            metrics['feature_importance'] = feature_importance.to_dict()
        
        logger.info(f"Value trajectory model - Test R²: {metrics['test_r2']:.4f}, MAE: {metrics['test_mae']:.2f}")
        
        # Save trajectory model
        self._save_trajectory_model()
        
        return metrics
    
    def predict_future_values(self, player_id=None, seasons_ahead=3):
        """Predict future transfer values for players
        
        Args:
            player_id (int): Player ID to predict for (None = all players)
            seasons_ahead (int): Number of future seasons to predict
            
        Returns:
            pd.DataFrame: Predicted future values
        """
        # Load trajectory model if not available
        if self.trajectory_model is None:
            self._load_trajectory_model()
            
            if self.trajectory_model is None:
                logger.error("No trajectory model available. Training one now...")
                self.train_value_trajectory_model()
                
                if self.trajectory_model is None:
                    logger.error("Failed to train trajectory model")
                    return None
        
        # Get player data with value history
        current_season = "2023-2024"  # Current season
        df = db.get_players_with_market_value_history(player_id=player_id, current_season=current_season)
        
        if df.empty:
            logger.error("No players found for future value prediction")
            return None
        
        # Feature columns for prediction
        feature_cols = [col for col in self.trajectory_features if col in df.columns]
        
        # For missing columns, try to compute them
        if 'value_growth_1yr' not in df.columns or df['value_growth_1yr'].isnull().all():
            logger.warning("value_growth_1yr missing, computing from available data")
            
            # Check if we have the required columns
            if 'market_value_2022_2023' in df.columns and 'market_value_2023_2024' in df.columns:
                df['value_growth_1yr'] = df.apply(
                    lambda row: (row['market_value_2023_2024'] - row['market_value_2022_2023']) / row['market_value_2022_2023'] * 100
                    if row['market_value_2022_2023'] > 0 and not pd.isnull(row['market_value_2022_2023']) and not pd.isnull(row['market_value_2023_2024']) 
                    else 0, 
                    axis=1
                )
        
        # Fill missing values with medians
        for col in feature_cols:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
        
        # Prepare results dataframe
        result_df = df[['id', 'name', 'age', 'position', 'club', 'market_value']].copy()
        
        # Get the most recent market value for each player
        result_df['current_value'] = df['market_value']
        
        # Add columns for future seasons
        current_year = 2023  # Base year for current season 2023-2024
        
        # Generate future predictions for each season
        for season_offset in range(1, seasons_ahead + 1):
            future_season = f"{current_year + season_offset}-{current_year + season_offset + 1}"
            future_column = f"predicted_value_{current_year + season_offset}"
            
            logger.info(f"Predicting values for season {future_season}")
            
            # Update age for this future season
            df['future_age'] = df['age'] + season_offset
            
            # Create feature matrix for this prediction
            X_future = df[feature_cols].copy()
            X_future['age'] = df['future_age']  # Use future age
            
            # Scale features
            X_future_scaled = self.trajectory_scaler.transform(X_future)
            
            # Predict value percent change
            predicted_pct_change = self.trajectory_model.predict(X_future_scaled)
            
            # Calculate new values based on percent change (compounding from previous prediction)
            if season_offset == 1:
                # First future season is based on current value
                base_values = df['market_value']
            else:
                # Subsequent seasons compound from previous prediction
                prev_column = f"predicted_value_{current_year + season_offset - 1}"
                base_values = result_df[prev_column]
            
            # Apply value changes with age and position adjustments
            future_values = []
            
            for idx, row in df.iterrows():
                player_position = row['position']
                future_age = row['future_age']
                current_value = base_values.iloc[idx]
                
                # Get position-specific career attributes
                career_attrs = self.position_career_attributes.get(
                    player_position, 
                    {'typical_peak': 27, 'career_length': 15, 'value_stability': 0.75}  # Default values
                )
                
                # Raw predicted change from the model
                raw_pct_change = predicted_pct_change[idx]
                
                # Adjust change based on age relative to peak
                years_from_peak = abs(future_age - career_attrs['typical_peak'])
                age_factor = max(0, 1 - (years_from_peak / career_attrs['career_length']))
                
                # For players past peak, we reduce the positive growth or amplify the negative
                if future_age > career_attrs['typical_peak']:
                    if raw_pct_change > 0:
                        # Reduce positive growth for older players
                        adjusted_pct_change = raw_pct_change * age_factor
                    else:
                        # Amplify decline for older players
                        adjusted_pct_change = raw_pct_change * (2 - age_factor)
                else:
                    # For younger players, we can see more dramatic changes
                    youth_boost = max(0, 1 - (future_age / career_attrs['typical_peak']))
                    if raw_pct_change > 0:
                        # Young players can see faster growth
                        adjusted_pct_change = raw_pct_change * (1 + youth_boost)
                    else:
                        # Young players are less likely to see big drops
                        adjusted_pct_change = raw_pct_change * (1 - youth_boost)
                
                # Calculate final value
                future_value = current_value * (1 + adjusted_pct_change / 100)
                
                # Ensure value doesn't go too low
                future_value = max(future_value, current_value * 0.3)
                
                # Add some randomness/variability (market factors)
                variability = np.random.uniform(0.9, 1.1)
                future_value = future_value * variability
                
                future_values.append(int(future_value))
            
            # Add to results
            result_df[future_column] = future_values
        
        return result_df
    
    def _save_trajectory_model(self):
        """Save trajectory model to disk"""
        try:
            # Create paths
            model_path = os.path.join(MODEL_DIR, f"value_trajectory_model.pkl")
            scaler_path = os.path.join(MODEL_DIR, f"value_trajectory_scaler.pkl")
            
            # Save model and scaler
            with open(model_path, 'wb') as f:
                pickle.dump(self.trajectory_model, f)
                
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.trajectory_scaler, f)
                
            logger.info(f"Value trajectory model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving trajectory model: {e}")
            return False
    
    def _load_trajectory_model(self):
        """Load trajectory model from disk"""
        try:
            # Create paths
            model_path = os.path.join(MODEL_DIR, f"value_trajectory_model.pkl")
            scaler_path = os.path.join(MODEL_DIR, f"value_trajectory_scaler.pkl")
            
            # Check if files exist
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                logger.warning("Trajectory model files not found")
                return False
            
            # Load model and scaler
            with open(model_path, 'rb') as f:
                self.trajectory_model = pickle.load(f)
                
            with open(scaler_path, 'rb') as f:
                self.trajectory_scaler = pickle.load(f)
                
            logger.info(f"Value trajectory model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading trajectory model: {e}")
            return False
    
    def analyze_value_change_factors(self):
        """Analyze which factors correlate most with transfer value changes
        
        Returns:
            dict: Analysis results with correlations
        """
        # Get data with value history
        df = self._get_value_history_data()
        
        if df.empty:
            logger.error("No data available for value change factor analysis")
            return None
        
        # Get all value change columns
        value_change_cols = [col for col in df.columns if col.startswith('value_pct_change_')]
        
        if not value_change_cols:
            logger.error("No value change columns found in data")
            return None
        
        # Potential factors to analyze
        factor_groups = {
            'performance': ['goals_per90', 'assists_per90', 'xg_per90', 'xa_per90', 
                            'sca_per90', 'gca_per90', 'pass_completion_pct'],
            'playing_time': ['minutes_played', 'games_played'],
            'player_profile': ['age', 'position']
        }
        
        # Result structure
        results = {
            'overall': {},
            'by_position': {},
            'by_age_group': {}
        }
        
        # Calculate overall correlations
        logger.info("Analyzing overall value change correlations...")
        
        correlation_data = []
        
        for change_col in value_change_cols:
            season = change_col.replace('value_pct_change_', '')
            
            # Extract factors available for this season
            valid_factors = []
            for group, factors in factor_groups.items():
                for factor in factors:
                    if factor in df.columns and not df[factor].isnull().all():
                        valid_factors.append(factor)
            
            # Calculate correlations for this season's value change
            for factor in valid_factors:
                corr = df[[factor, change_col]].corr().iloc[0, 1]
                if not pd.isnull(corr):
                    correlation_data.append({
                        'season': season,
                        'factor': factor,
                        'correlation': corr
                    })
        
        # Convert to DataFrame for easier analysis
        corr_df = pd.DataFrame(correlation_data)
        
        if not corr_df.empty:
            # Calculate average correlation across seasons for each factor
            avg_correlations = corr_df.groupby('factor')['correlation'].mean().sort_values(ascending=False)
            
            # Add to results
            results['overall'] = avg_correlations.to_dict()
            
            # Top positive correlations
            logger.info("Top factors positively correlated with value increases:")
            for factor, corr in avg_correlations[avg_correlations > 0].head(5).items():
                logger.info(f"{factor}: {corr:.4f}")
            
            # Top negative correlations
            logger.info("Top factors negatively correlated with value increases:")
            for factor, corr in avg_correlations[avg_correlations < 0].head(5).items():
                logger.info(f"{factor}: {corr:.4f}")
        
        # Position-specific analysis
        logger.info("Analyzing position-specific value change correlations...")
        
        # Group players by position
        for position in df['position'].unique():
            position_df = df[df['position'] == position]
            
            if len(position_df) < 5:  # Need enough players for meaningful correlations
                continue
                
            position_corr_data = []
            
            for change_col in value_change_cols:
                season = change_col.replace('value_pct_change_', '')
                
                # Calculate correlations for performance factors for this position
                for group, factors in factor_groups.items():
                    for factor in factors:
                        if factor in position_df.columns and not position_df[factor].isnull().all():
                            corr = position_df[[factor, change_col]].corr().iloc[0, 1]
                            if not pd.isnull(corr):
                                position_corr_data.append({
                                    'season': season,
                                    'factor': factor,
                                    'correlation': corr
                                })
            
            # Convert to DataFrame for this position
            pos_corr_df = pd.DataFrame(position_corr_data)
            
            if not pos_corr_df.empty:
                # Calculate average correlation across seasons for each factor
                pos_avg_correlations = pos_corr_df.groupby('factor')['correlation'].mean().sort_values(ascending=False)
                
                # Add to results
                results['by_position'][position] = pos_avg_correlations.to_dict()
                
                # Log top factors for this position
                logger.info(f"Top 3 value drivers for {position} players:")
                for factor, corr in pos_avg_correlations.head(3).items():
                    logger.info(f"{factor}: {corr:.4f}")
        
        # Age group analysis
        logger.info("Analyzing age-specific value change correlations...")
        
        # Create age groups
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 21, 25, 29, 33, 100],
            labels=['U21', '22-25', '26-29', '30-33', '34+']
        )
        
        # Group players by age group
        for age_group in df['age_group'].unique():
            age_df = df[df['age_group'] == age_group]
            
            if len(age_df) < 5:  # Need enough players for meaningful correlations
                continue
                
            age_corr_data = []
            
            for change_col in value_change_cols:
                season = change_col.replace('value_pct_change_', '')
                
                # Calculate correlations for factors for this age group
                for group, factors in factor_groups.items():
                    for factor in factors:
                        if factor in age_df.columns and not age_df[factor].isnull().all():
                            corr = age_df[[factor, change_col]].corr().iloc[0, 1]
                            if not pd.isnull(corr):
                                age_corr_data.append({
                                    'season': season,
                                    'factor': factor,
                                    'correlation': corr
                                })
            
            # Convert to DataFrame for this age group
            age_corr_df = pd.DataFrame(age_corr_data)
            
            if not age_corr_df.empty:
                # Calculate average correlation across seasons for each factor
                age_avg_correlations = age_corr_df.groupby('factor')['correlation'].mean().sort_values(ascending=False)
                
                # Add to results
                results['by_age_group'][str(age_group)] = age_avg_correlations.to_dict()
                
                # Log top factors for this age group
                logger.info(f"Top 3 value drivers for {age_group} players:")
                for factor, corr in age_avg_correlations.head(3).items():
                    logger.info(f"{factor}: {corr:.4f}")
        
        # Save correlation results to file
        try:
            results_path = os.path.join(METRICS_DIR, f"value_change_correlations.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Value change correlation analysis saved to {results_path}")
        except Exception as e:
            logger.error(f"Error saving value change correlations: {e}")
        
        return results
    
    def identify_undervalued_players(self, min_performance_score=60, min_value_potential=20):
        """Identify undervalued players based on performance and value trajectory
        
        Args:
            min_performance_score (float): Minimum performance score (0-100)
            min_value_potential (float): Minimum projected value increase percentage
            
        Returns:
            pd.DataFrame: Undervalued players with scores and projections
        """
        logger.info("Identifying undervalued players...")
        
        # 1. Get current performance scores from base ML model
        current_predictions = base_ml.predict_current_values(
            position_specific=True, 
            age_adjusted=True
        )
        
        if current_predictions is None or current_predictions.empty:
            logger.error("Failed to get current player valuations")
            return None
        
        # Keep only players with sufficient performance scores
        performance_filtered = current_predictions[
            current_predictions['performance_score'] >= min_performance_score
        ].copy()
        
        if performance_filtered.empty:
            logger.warning(f"No players meet the minimum performance score of {min_performance_score}")
            return None
        
        logger.info(f"Found {len(performance_filtered)} players with performance score >= {min_performance_score}")
        
        # 2. Get future value projections
        future_predictions = self.predict_future_values(seasons_ahead=2)
        
        if future_predictions is None or future_predictions.empty:
            logger.error("Failed to predict future values")
            return None
        
        # 3. Merge performance data with future projections
        merged_data = performance_filtered.merge(
            future_predictions[['id', 'predicted_value_2024', 'predicted_value_2025']], 
            on='id', 
            how='inner'
        )
        
        if merged_data.empty:
            logger.error("No matching players found after merging predictions")
            return None
        
        # 4. Calculate value potential (2-year growth projection)
        merged_data['value_potential'] = (
            (merged_data['predicted_value_2025'] - merged_data['market_value']) / 
            merged_data['market_value'] * 100
        )
        
        # 5. Calculate combined score (performance + potential)
        # Normalize value potential for scoring (0-100 scale)
        max_potential = merged_data['value_potential'].max()
        min_potential = merged_data['value_potential'].min()
        
        if max_potential > min_potential:
            potential_range = max_potential - min_potential
            merged_data['potential_score'] = (
                (merged_data['value_potential'] - min_potential) / potential_range * 100
            )
        else:
            merged_data['potential_score'] = 50  # Default if no range
            
        # Combined score: 60% performance, 40% potential
        merged_data['undervalued_score'] = (
            merged_data['performance_score'] * 0.6 + 
            merged_data['potential_score'] * 0.4
        )
        
        # 6. Filter by minimum value potential
        undervalued = merged_data[
            merged_data['value_potential'] >= min_value_potential
        ].copy()
        
        if undervalued.empty:
            logger.warning(f"No players meet the minimum value potential of {min_value_potential}%")
            return None
        
        logger.info(f"Found {len(undervalued)} undervalued players with value potential >= {min_value_potential}%")
        
        # 7. Sort by undervalued score
        undervalued = undervalued.sort_values('undervalued_score', ascending=False)
        
        # Add investment attractiveness rating
        def investment_rating(row):
            score = row['undervalued_score']
            if score >= 85:
                return "A+"
            elif score >= 80:
                return "A"
            elif score >= 75:
                return "A-"
            elif score >= 70:
                return "B+"
            elif score >= 65:
                return "B"
            elif score >= 60:
                return "B-"
            elif score >= 55:
                return "C+"
            elif score >= 50:
                return "C"
            else:
                return "C-"
        
        undervalued['investment_rating'] = undervalued.apply(investment_rating, axis=1)
        
        # Select relevant columns for output
        result = undervalued[[
            'id', 'name', 'age', 'position', 'club', 'league',
            'market_value', 'performance_score', 'value_ratio',
            'predicted_value_2024', 'predicted_value_2025',
            'value_potential', 'undervalued_score', 'investment_rating'
        ]]
        
        return result

    def get_value_progression_comparison(self, player_ids, seasons_ahead=3):
        """Compare value progression between multiple players
        
        Args:
            player_ids (list): List of player IDs to compare
            seasons_ahead (int): Number of future seasons to predict
            
        Returns:
            dict: Comparison data for the specified players
        """
        logger.info(f"Comparing value progression for {len(player_ids)} players")
        
        # Get player info
        player_info = {}
        for player_id in player_ids:
            # Get current player data
            player_df = db.get_players_with_market_value_history(player_id=player_id)
            
            if player_df.empty:
                logger.warning(f"No data found for player ID {player_id}")
                continue
                
            # Get basic player info
            player_row = player_df.iloc[0]
            player_info[player_id] = {
                'id': player_id,
                'name': player_row['name'],
                'age': player_row['age'],
                'position': player_row['position'],
                'club': player_row['club'],
                'current_value': player_row['market_value'],
                'historical_values': {},
                'future_values': {}
            }
            
            # Add historical values
            seasons = ["2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"]
            for season in seasons:
                col_name = f"market_value_{season.replace('-', '_')}"
                if col_name in player_df.columns:
                    value = player_df[col_name].iloc[0]
                    if pd.notnull(value):
                        player_info[player_id]['historical_values'][season] = int(value)
        
        # Get future value predictions
        future_df = self.predict_future_values(seasons_ahead=seasons_ahead)
        
        if future_df is not None and not future_df.empty:
            for player_id in player_ids:
                if player_id not in player_info:
                    continue
                    
                player_future = future_df[future_df['id'] == player_id]
                
                if not player_future.empty:
                    # Get predicted values
                    current_year = 2023  # Base year
                    for season_offset in range(1, seasons_ahead + 1):
                        future_season = f"{current_year + season_offset}-{current_year + season_offset + 1}"
                        col_name = f"predicted_value_{current_year + season_offset}"
                        
                        if col_name in player_future.columns:
                            value = player_future[col_name].iloc[0]
                            player_info[player_id]['future_values'][future_season] = int(value)
        
        # Calculate career peak and current progress
        for player_id, info in player_info.items():
            # Combine historical and future values
            all_values = {**info['historical_values'], **info['future_values']}
            
            if all_values:
                # Find max value and its season
                max_value = max(all_values.values())
                max_seasons = [s for s, v in all_values.items() if v == max_value]
                
                info['peak_value'] = max_value
                info['peak_season'] = max_seasons[0] if max_seasons else None
                
                # Calculate current progress as percentage of peak
                if 'current_value' in info and info['peak_value'] > 0:
                    info['peak_progress'] = (info['current_value'] / info['peak_value']) * 100
                else:
                    info['peak_progress'] = None
        
        return {
            'players': list(player_info.values()),
            'seasons': {
                'historical': ["2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"],
                'future': [f"{2023 + i}-{2024 + i}" for i in range(1, seasons_ahead + 1)]
            }
        }

# Utility functions for the API endpoints

def train_transfer_value_model(model_type="random_forest", tag="transfer_value_baseline"):
    """Train a transfer value model and return metrics
    
    Args:
        model_type (str): Type of model to use
        tag (str): Tag for metrics tracking
        
    Returns:
        dict: Training metrics
    """
    # Create and train the model
    model = TransferValueModel(model_type=model_type)
    
    # Train base value prediction model
    base_metrics = model.train(tag=tag)
    
    if base_metrics is None:
        logger.error("Failed to train base value prediction model")
        return None
    
    # Train trajectory model
    trajectory_metrics = model.train_value_trajectory_model()
    
    if trajectory_metrics is None:
        logger.warning("Failed to train value trajectory model")
    
    # Analyze value change factors
    value_factors = model.analyze_value_change_factors()
    
    # Combine all metrics
    all_metrics = {
        "base_model": base_metrics,
        "trajectory_model": trajectory_metrics,
        "value_factors": value_factors is not None
    }
    
    return all_metrics

def get_top_investment_opportunities(min_performance=60, min_potential=20, max_results=10):
    """Get top investment opportunities based on undervalued players
    
    Args:
        min_performance (float): Minimum performance score (0-100)
        min_potential (float): Minimum value growth potential (%)
        max_results (int): Maximum number of results to return
        
    Returns:
        pd.DataFrame: Top investment opportunities
    """
    # Create transfer value model
    model = TransferValueModel()
    
    # Try to load existing models
    model.load_model()
    model._load_trajectory_model()
    
    # Find undervalued players
    opportunities = model.identify_undervalued_players(
        min_performance_score=min_performance,
        min_value_potential=min_potential
    )
    
    if opportunities is None or opportunities.empty:
        logger.error("No investment opportunities found")
        return None
    
    # Return top results
    return opportunities.head(max_results)

def get_stat_value_impact_analysis():
    """Analyze which stats most impact player market value
    
    Returns:
        dict: Analysis of stat impact on value
    """
    # Create transfer value model
    model = TransferValueModel()
    
    # Analyze factors
    return model.analyze_value_change_factors()

def predict_player_future_values(player_id=None, seasons_ahead=3):
    """Predict future market values for players
    
    Args:
        player_id (int): Player ID (None = all players)
        seasons_ahead (int): Number of seasons to predict ahead
        
    Returns:
        pd.DataFrame: Predicted future values
    """
    # Create transfer value model
    model = TransferValueModel()
    
    # Try to load trajectory model
    model._load_trajectory_model()
    
    # Predict future values
    return model.predict_future_values(player_id=player_id, seasons_ahead=seasons_ahead)

def compare_players_value_progression(player_ids, seasons_ahead=3):
    """Compare value progression between players
    
    Args:
        player_ids (list): List of player IDs to compare
        seasons_ahead (int): Number of future seasons to predict
        
    Returns:
        dict: Comparison data
    """
    # Create transfer value model
    model = TransferValueModel()
    
    # Try to load trajectory model
    model._load_trajectory_model()
    
    # Compare players
    return model.get_value_progression_comparison(player_ids, seasons_ahead=seasons_ahead)

# Main function for testing
if __name__ == "__main__":
    # Example usage
    model = TransferValueModel()
    
    # 1. Train model with historical values
    # metrics = model.train(tag="transfer_value_baseline")
    # print(f"Base model metrics: {metrics['test_r2']:.4f} R², {metrics['test_rmse']:.2f} RMSE")
    
    # 2. Train trajectory model
    # trajectory_metrics = model.train_value_trajectory_model()
    # print(f"Trajectory model metrics: {trajectory_metrics['test_r2']:.4f} R², {trajectory_metrics['test_mae']:.2f} MAE")
    
    # 3. Analyze which factors impact value changes
    # factors = model.analyze_value_change_factors()
    # print("Top value change factors:", factors['overall'])
    
    # 4. Identify undervalued players
    # undervalued = model.identify_undervalued_players(min_performance_score=60, min_value_potential=15)
    # if undervalued is not None:
    #     print(f"Found {len(undervalued)} undervalued players")
    #     print(undervalued[['name', 'age', 'position', 'market_value', 'value_potential', 'investment_rating']].head(10))
    
    # 5. Predict future values
    # future_values = model.predict_future_values(seasons_ahead=3)
    # if future_values is not None:
    #     print(f"Predicted future values for {len(future_values)} players")
    #     print(future_values[['name', 'age', 'current_value', 'predicted_value_2024', 'predicted_value_2025', 'predicted_value_2026']].head(10))
    
    # 6. Compare value progression between players
    # comparison = model.get_value_progression_comparison([1, 2, 3, 4, 5], seasons_ahead=3)
    # print(f"Compared value progression for {len(comparison['players'])} players")