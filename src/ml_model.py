"""
ML model for predicting player market values and identifying undervalued players.

This module implements machine learning models to:
1. Predict player market values based on performance statistics
2. Analyze historical accuracy of predictions
3. Identify the most important features for market value prediction
4. Adjust weights based on model learning
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import pickle
import json
from datetime import datetime
import database as db
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging

# Setup enhanced logging
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Create file handler
log_file = os.path.join(LOG_DIR, f'ml_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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

# Path for saving models and metrics
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
METRICS_DIR = os.path.join(os.path.dirname(__file__), 'metrics')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def save_metrics_to_json(metrics, model_type="random_forest", tag=None):
    """Save model performance metrics to a JSON file for easy tracking
    
    Args:
        metrics (dict): Dictionary of model metrics
        model_type (str): Type of model
        tag (str): Optional tag to identify this metrics snapshot (e.g., "baseline", "more_data")
    """
    try:
        # Create a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a filename that includes model type, tag if provided, and timestamp
        if tag:
            filename = f"{model_type}_{tag}_{timestamp}_metrics.json"
        else:
            filename = f"{model_type}_{timestamp}_metrics.json"
        
        file_path = os.path.join(METRICS_DIR, filename)
        
        # Add metadata
        metrics_with_meta = {
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "tag": tag,
            "metrics": metrics
        }
        
        # Write metrics to file
        with open(file_path, 'w') as f:
            json.dump(metrics_with_meta, f, indent=2)
            
        logger.info(f"Metrics saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving metrics to JSON: {e}")
        return None

def load_latest_metrics(model_type="random_forest", tag=None):
    """Load the most recent metrics for a given model type and optional tag
    
    Args:
        model_type (str): Type of model
        tag (str): Optional tag to filter metrics
        
    Returns:
        dict: Metrics dictionary or None if not found
    """
    try:
        # List all metric files for this model type
        files = [f for f in os.listdir(METRICS_DIR) 
                 if f.startswith(model_type) and f.endswith("_metrics.json")]
        
        # Filter by tag if provided
        if tag:
            files = [f for f in files if tag in f]
            
        if not files:
            logger.warning(f"No metric files found for {model_type}" + 
                          (f" with tag {tag}" if tag else ""))
            return None
            
        # Sort by timestamp (part of filename)
        files.sort(reverse=True)
        
        # Load the most recent file
        latest_file = os.path.join(METRICS_DIR, files[0])
        with open(latest_file, 'r') as f:
            metrics = json.load(f)
            
        logger.info(f"Loaded metrics from {latest_file}")
        return metrics
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        return None

def compare_metrics(current_metrics, previous_metrics=None, model_type="random_forest"):
    """Compare current metrics with previous metrics
    
    Args:
        current_metrics (dict): Current performance metrics
        previous_metrics (dict): Previous performance metrics to compare against
        model_type (str): Type of model
        
    Returns:
        dict: Comparison results with improvement percentages
    """
    if previous_metrics is None:
        # Try to load most recent metrics
        previous_data = load_latest_metrics(model_type)
        if previous_data:
            previous_metrics = previous_data["metrics"]
        else:
            logger.warning("No previous metrics found for comparison")
            return None
    
    if not previous_metrics:
        logger.warning("No previous metrics provided for comparison")
        return None
        
    # Extract test metrics for comparison
    current_test_metrics = {
        "r2": current_metrics["test_r2"],
        "rmse": current_metrics["test_rmse"],
        "mae": current_metrics["test_mae"]
    }
    
    previous_test_metrics = {
        "r2": previous_metrics["test_r2"],
        "rmse": previous_metrics["test_rmse"],
        "mae": previous_metrics["test_mae"]
    }
    
    # Calculate changes and percent improvements
    comparison = {}
    
    # R2 improvement (higher is better)
    r2_change = current_test_metrics["r2"] - previous_test_metrics["r2"]
    if previous_test_metrics["r2"] != 0:
        r2_pct_change = (r2_change / abs(previous_test_metrics["r2"])) * 100
    else:
        r2_pct_change = float('inf') if r2_change > 0 else float('-inf') if r2_change < 0 else 0
    
    # RMSE improvement (lower is better)
    rmse_change = previous_test_metrics["rmse"] - current_test_metrics["rmse"]
    rmse_pct_change = (rmse_change / previous_test_metrics["rmse"]) * 100 if previous_test_metrics["rmse"] != 0 else 0
    
    # MAE improvement (lower is better)
    mae_change = previous_test_metrics["mae"] - current_test_metrics["mae"]
    mae_pct_change = (mae_change / previous_test_metrics["mae"]) * 100 if previous_test_metrics["mae"] != 0 else 0
    
    comparison = {
        "current": current_test_metrics,
        "previous": previous_test_metrics,
        "changes": {
            "r2": {
                "absolute": r2_change,
                "percentage": r2_pct_change,
                "improved": r2_change > 0
            },
            "rmse": {
                "absolute": rmse_change,
                "percentage": rmse_pct_change,
                "improved": rmse_change > 0
            },
            "mae": {
                "absolute": mae_change,
                "percentage": mae_pct_change,
                "improved": mae_change > 0
            }
        },
        "summary": {
            "overall_improved": (r2_change > 0 and rmse_change > 0 and mae_change > 0)
        }
    }
    
    # Log the comparison results
    logger.info("===== MODEL PERFORMANCE COMPARISON =====")
    logger.info(f"R² score: {current_test_metrics['r2']:.4f} vs {previous_test_metrics['r2']:.4f} " + 
               f"({'↑' if r2_change > 0 else '↓'} {abs(r2_pct_change):.2f}%)")
    logger.info(f"RMSE: {current_test_metrics['rmse']:.2f} vs {previous_test_metrics['rmse']:.2f} " + 
               f"({'↓' if rmse_change > 0 else '↑'} {abs(rmse_pct_change):.2f}%)")
    logger.info(f"MAE: {current_test_metrics['mae']:.2f} vs {previous_test_metrics['mae']:.2f} " + 
               f"({'↓' if mae_change > 0 else '↑'} {abs(mae_pct_change):.2f}%)")
    logger.info(f"Overall: {'IMPROVED' if comparison['summary']['overall_improved'] else 'MIXED OR DEGRADED'}")
    logger.info("========================================")
    
    return comparison

class PlayerValueModel:
    """Machine learning model for predicting player market values"""
    
    def __init__(self, model_type="random_forest", position_specific=True, age_adjusted=True):
        """Initialize the model
        
        Args:
            model_type (str): Type of model to use
                Options: "random_forest", "gradient_boosting", "ridge", "lasso"
            position_specific (bool): Whether to train position-specific models
            age_adjusted (bool): Whether to include age-based adjustments
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        self.metrics = {}
        self.position_specific = position_specific
        self.age_adjusted = age_adjusted
        
        # Position-specific models
        self.position_models = {
            'forward': None,
            'midfielder': None,
            'defender': None,
            'goalkeeper': None
        }
        
        # Position categories - mapping specific positions to general categories
        self.position_categories = {
            # Forwards
            'CF': 'forward', 'ST': 'forward', 'LW': 'forward', 'RW': 'forward', 
            'SS': 'forward', 'LF': 'forward', 'RF': 'forward',
            # Midfielders
            'CAM': 'midfielder', 'CM': 'midfielder', 'CDM': 'midfielder', 
            'LM': 'midfielder', 'RM': 'midfielder', 'AM': 'midfielder', 'DM': 'midfielder',
            # Defenders
            'CB': 'defender', 'LB': 'defender', 'RB': 'defender', 
            'LWB': 'defender', 'RWB': 'defender', 'SW': 'defender',
            # Goalkeepers
            'GK': 'goalkeeper'
        }
        
        # Define feature groups
        self.feature_groups = {
            "offensive": ['goals', 'assists', 'xg', 'xa', 'npxg', 'sca', 'gca',
                         'shots', 'shots_on_target', 'progressive_carries', 
                         'progressive_passes', 'penalty_box_touches'],
            "possession": ['passes_completed', 'passes_attempted', 'pass_completion_pct',
                          'progressive_passes_received', 'dribbles_completed', 
                          'dribbles_attempted', 'ball_recoveries'],
            "defensive": ['tackles', 'tackles_won', 'interceptions', 'blocks',
                         'clearances', 'pressures', 'pressure_success_rate',
                         'aerial_duels_won', 'aerial_duels_total'],
            "physical": ['minutes_played', 'games_played', 'distance_covered',
                        'high_intensity_runs', 'yellow_cards', 'red_cards'],
            "per90": ['goals_per90', 'assists_per90', 'xg_per90', 'xa_per90',
                     'npxg_per90', 'sca_per90', 'gca_per90']
        }
        
        # Position-specific feature weights
        self.position_weights = {
            'forward': {
                'offensive': 1.5,   # Higher weight for offensive stats
                'possession': 1.0,
                'defensive': 0.5,   # Lower weight for defensive stats
                'physical': 1.0,
                'per90': 1.5
            },
            'midfielder': {
                'offensive': 1.0,
                'possession': 1.5,  # Higher weight for possession stats
                'defensive': 1.0,
                'physical': 1.0,
                'per90': 1.0
            },
            'defender': {
                'offensive': 0.5,   # Lower weight for offensive stats
                'possession': 1.0,
                'defensive': 1.5,   # Higher weight for defensive stats
                'physical': 1.0,
                'per90': 0.8
            },
            'goalkeeper': {
                'offensive': 0.2,
                'possession': 0.7,
                'defensive': 1.2,
                'physical': 1.0,
                'per90': 0.5
            }
        }
        
        # Age adjustment factors
        # Market value typically peaks at around age 27 for most players
        self.age_adjustment = {
            # Age: adjustment factor
            16: 0.8,  # Young players with potential
            17: 0.9,
            18: 1.0,
            19: 1.1,
            20: 1.2,
            21: 1.3,
            22: 1.4, 
            23: 1.5,  # Rising value as approaching prime
            24: 1.6,
            25: 1.7,
            26: 1.8,
            27: 1.9,  # Peak value
            28: 1.8,
            29: 1.6,
            30: 1.4,  # Declining value after 30
            31: 1.2,
            32: 1.0,
            33: 0.8,
            34: 0.6,
            35: 0.4,
            36: 0.3,
            37: 0.2,
            38: 0.15,
            39: 0.1,
            40: 0.05
        }
        
        # Define non-feature columns but add age as a feature
        self.non_features = ['id', 'name', 'nationality', 'club', 'league',
                            'height', 'weight', 'preferred_foot', 'market_value', 
                            'position', 'position_category', 'age_group', 'season', 'last_updated']
    
    def _get_features(self, data):
        """Extract features from data"""
        # Get all features by excluding non-feature columns
        features = [col for col in data.columns if col not in self.non_features and not col.endswith('_weighted')]
        logger.info(f"Selected {len(features)} features for model: {features}")
        return features
    
    def train(self, seasons=None, test_size=0.2, save_model=True, tag=None):
        """Train the model on historical data
        
        Args:
            seasons (list): List of seasons to include
            test_size (float): Proportion of data to use for testing
            save_model (bool): Whether to save the trained model
            tag (str): Optional tag to identify this training run (e.g., "baseline", "more_data")
            
        Returns:
            dict: Training metrics
        """
        # Log training details
        logger.info("====================================================")
        logger.info(f"STARTING TRAINING: {self.model_type.upper()} MODEL")
        if tag:
            logger.info(f"Training run tag: {tag}")
        logger.info(f"Position-specific: {self.position_specific}, Age-adjusted: {self.age_adjusted}")
        logger.info("====================================================")
        
        # Record training start time
        training_start_time = datetime.now()
        
        # Get training data
        df = self._get_training_data(seasons)
        
        if df.empty:
            logger.error("No training data available")
            return None
        
        # Log data metrics
        logger.info(f"Training data: {len(df)} players across {df['season'].nunique()} seasons")
        if seasons:
            logger.info(f"Seasons included: {seasons}")
        else:
            logger.info(f"Seasons included: {df['season'].unique().tolist()}")
        
        # Add position category feature
        df['position_category'] = df['position'].apply(
            lambda pos: self.position_categories.get(pos, 'midfielder')  # Default to midfielder if unknown
        )
        
        # Add age-related features
        if self.age_adjusted:
            # Age group (binned)
            df['age_group'] = pd.cut(
                df['age'], 
                bins=[15, 21, 25, 29, 33, 40], 
                labels=['youth', 'developing', 'prime', 'experienced', 'veteran']
            )
            
            # Age adjustment factor
            df['age_factor'] = df['age'].apply(
                lambda age: self.age_adjustment.get(age, 1.0)  # Default to 1.0 if age not in mapping
            )
            
            # Years to peak (positive = before peak, negative = after peak)
            PEAK_AGE = 27
            df['years_to_peak'] = PEAK_AGE - df['age']
            
            # Years of career left (rough estimate)
            df['estimated_years_left'] = 35 - df['age']
            df.loc[df['estimated_years_left'] < 0, 'estimated_years_left'] = 0
        
        # If training position-specific models
        if self.position_specific:
            position_metrics = {}
            position_counts = df['position_category'].value_counts().to_dict()
            logger.info(f"Position distribution: {position_counts}")
            
            # Train a separate model for each position category
            for position, position_df in df.groupby('position_category'):
                if len(position_df) < 10:  # Skip if too few samples
                    logger.warning(f"Not enough samples ({len(position_df)}) for position {position}. Skipping.")
                    continue
                
                logger.info(f"Training position-specific model for {position} ({len(position_df)} players)")
                
                # Get position-specific features and target
                features = self._get_features(position_df)
                
                # Check for and handle any missing values
                missing_values = position_df[features].isnull().sum()
                features_with_missing = missing_values[missing_values > 0]
                
                if not features_with_missing.empty:
                    logger.warning(f"Position {position} - Features with missing values: \n{features_with_missing}")
                    logger.info(f"Position {position} - Filling missing values with median values...")
                    for feature in features_with_missing.index:
                        position_df[feature] = position_df[feature].fillna(position_df[feature].median())
                
                # Verify all features are numeric
                non_numeric_features = []
                for feature in features:
                    if not pd.api.types.is_numeric_dtype(position_df[feature]):
                        non_numeric_features.append(feature)
                        logger.warning(f"Position {position} - Feature '{feature}' is not numeric: {position_df[feature].dtype}")
                
                if non_numeric_features:
                    logger.warning(f"Position {position} - Removing non-numeric features: {non_numeric_features}")
                    # Remove non-numeric features
                    features = [f for f in features if f not in non_numeric_features]
                    if not features:
                        logger.error(f"Position {position} - No numeric features left for training!")
                        continue
                
                X = position_df[features]
                y = position_df['market_value']
                
                # Split into train and test sets
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=min(test_size, 0.3), random_state=42
                    )
                except ValueError as e:
                    logger.error(f"Error splitting data for position {position}: {e}")
                    continue
                
                # Scale features
                position_scaler = StandardScaler()
                X_train_scaled = position_scaler.fit_transform(X_train)
                X_test_scaled = position_scaler.transform(X_test)
                
                # Create position-specific model
                position_model = self._create_model()
                
                if position_model is None:
                    continue
                
                # Train the model
                position_model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred_train = position_model.predict(X_train_scaled)
                y_pred_test = position_model.predict(X_test_scaled)
                
                # Calculate metrics
                pos_metrics = {
                    "train_mae": mean_absolute_error(y_train, y_pred_train),
                    "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    "train_r2": r2_score(y_train, y_pred_train),
                    "test_mae": mean_absolute_error(y_test, y_pred_test),
                    "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    "test_r2": r2_score(y_test, y_pred_test),
                    "data_size": len(position_df),
                    "feature_count": len(features)
                }
                
                logger.info(f"Position {position} model: Test R² score: {pos_metrics['test_r2']:.4f}, RMSE: {pos_metrics['test_rmse']:.2f}")
                
                # Store model and metrics
                self.position_models[position] = {
                    'model': position_model,
                    'scaler': position_scaler,
                    'metrics': pos_metrics,
                    'features': features
                }
                
                position_metrics[position] = pos_metrics
            
            # Also train a general model for fallback
            logger.info("Training general model for all positions")
        
        # Get features for general model
        features = self._get_features(df)
        
        # Check for and handle any missing values
        logger.info("Checking for missing values in features...")
        missing_values = df[features].isnull().sum()
        features_with_missing = missing_values[missing_values > 0]
        
        if not features_with_missing.empty:
            logger.warning(f"Features with missing values: \n{features_with_missing}")
            logger.info("Filling missing values with median values...")
            for feature in features_with_missing.index:
                df[feature] = df[feature].fillna(df[feature].median())
        
        # Verify all features are numeric
        non_numeric_features = []
        for feature in features:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                non_numeric_features.append(feature)
                logger.warning(f"Feature '{feature}' is not numeric: {df[feature].dtype}")
        
        if non_numeric_features:
            logger.error(f"Found non-numeric features that must be removed: {non_numeric_features}")
            # Remove non-numeric features
            features = [f for f in features if f not in non_numeric_features]
            if not features:
                logger.error("No numeric features left for training!")
                return None
        
        X = df[features]
        y = df['market_value']
        
        # Log available features
        logger.info(f"Training with {len(features)} features: {features}")
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Log target variable distribution
        logger.info(f"Market value range: €{y.min()/1000000:.2f}M - €{y.max()/1000000:.2f}M")
        logger.info(f"Market value mean: €{y.mean()/1000000:.2f}M, median: €{y.median()/1000000:.2f}M")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        logger.info(f"Training {self.model_type} model")
        self.model = self._create_model()
        
        if self.model is None:
            return None
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "test_r2": r2_score(y_test, y_pred_test),
            "data_size": len(df),
            "feature_count": len(features),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "market_value_min": float(y.min()),
            "market_value_max": float(y.max()),
            "market_value_mean": float(y.mean()),
            "market_value_median": float(y.median()),
            "training_date": datetime.now().isoformat(),
        }
        
        # Calculate percentage error
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_errors = np.abs(y_pred_test - y_test) / y_test * 100
            # Replace infinity and NaN with a high value (1000%)
            pct_errors = np.nan_to_num(pct_errors, nan=1000, posinf=1000)
            pct_error = np.mean(pct_errors)
            metrics["test_pct_error"] = float(pct_error)
        
        if self.position_specific:
            metrics['position_metrics'] = position_metrics
        
        self.metrics = metrics
        
        # Log detailed results
        logger.info("====================================================")
        logger.info("TRAINING RESULTS:")
        logger.info(f"Training R² score: {metrics['train_r2']:.4f}")
        logger.info(f"Test R² score: {metrics['test_r2']:.4f}")
        logger.info(f"Test RMSE: {metrics['test_rmse']:.2f}")
        logger.info(f"Test MAE: {metrics['test_mae']:.2f}")
        logger.info(f"Test Percentage Error: {metrics['test_pct_error']:.2f}%")
        logger.info("====================================================")
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 important features:")
            for i, (feature, importance) in enumerate(zip(
                self.feature_importance['feature'].head(10),
                self.feature_importance['importance'].head(10)
            )):
                logger.info(f"{i+1}. {feature}: {importance:.4f}")
        
        # Save model
        if save_model:
            self.save_model()
        
        # Save metrics
        metrics_path = save_metrics_to_json(metrics, self.model_type, tag)
        
        # Try to compare with previous metrics
        try:
            comparison = compare_metrics(metrics, model_type=self.model_type)
            if comparison:
                metrics['comparison'] = comparison
        except Exception as e:
            logger.warning(f"Could not compare with previous metrics: {e}")
        
        # Calculate training time
        training_end_time = datetime.now()
        training_time = (training_end_time - training_start_time).total_seconds()
        logger.info(f"Total training time: {training_time:.2f} seconds")
        metrics['training_time_seconds'] = training_time
        
        return metrics
        
    def _create_model(self):
        """Create a model instance based on model_type"""
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=100, random_state=42
            )
        elif self.model_type == "ridge":
            return Ridge(alpha=1.0, random_state=42)
        elif self.model_type == "lasso":
            return Lasso(alpha=0.1, random_state=42)
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            return None
    
    def _get_training_data(self, seasons=None):
        """Get training data from database
        
        Args:
            seasons (list): List of seasons to include
            
        Returns:
            pd.DataFrame: Training data
        """
        # Get all players with stats
        if seasons is None:
            # Use all seasons except the current one
            all_seasons = self._get_all_seasons()
            if all_seasons:
                # Use all but the most recent season for training
                seasons = all_seasons[:-1]
            else:
                seasons = ["2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023"]
                
        logger.info(f"Training on seasons: {seasons}")
        
        # Combine data from all seasons
        all_data = []
        for season in seasons:
            season_data = db.get_players_with_stats(season)
            if not season_data.empty:
                season_data['season'] = season
                all_data.append(season_data)
                logger.info(f"Added {len(season_data)} players from season {season}")
            else:
                logger.warning(f"No data found for season {season}")
        
        if not all_data:
            logger.warning("No data found for selected seasons")
            return pd.DataFrame()
            
        df = pd.concat(all_data, ignore_index=True)
        
        # Log data shape and columns before returning
        logger.info(f"Training data shape: {df.shape}")
        logger.info(f"Training data columns: {df.columns.tolist()}")
        logger.info(f"Data types: {df.dtypes}")
        
        return df
    
    def _get_all_seasons(self):
        """Get all available seasons from database"""
        conn = sqlite3.connect(db.DB_PATH)
        query = "SELECT DISTINCT season FROM player_stats ORDER BY season"
        seasons = pd.read_sql_query(query, conn)['season'].tolist()
        conn.close()
        return seasons
    
    def predict(self, player_data):
        """Predict market value for players
        
        Args:
            player_data (pd.DataFrame): Player data with features
            
        Returns:
            pd.Series: Predicted market values
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return None
        
        # Create a copy to avoid modifying the original
        df = player_data.copy()
        
        # Add position category feature
        if 'position_category' not in df.columns:
            df['position_category'] = df['position'].apply(
                lambda pos: self.position_categories.get(pos, 'midfielder')  # Default to midfielder if unknown
            )
        
        # Add age-related features if not present and age_adjusted is enabled
        if self.age_adjusted:
            if 'age_group' not in df.columns:
                # Age group (binned)
                df['age_group'] = pd.cut(
                    df['age'], 
                    bins=[15, 21, 25, 29, 33, 40], 
                    labels=['youth', 'developing', 'prime', 'experienced', 'veteran']
                )
                
            if 'age_factor' not in df.columns:
                # Age adjustment factor
                df['age_factor'] = df['age'].apply(
                    lambda age: self.age_adjustment.get(age, 1.0)  # Default to 1.0 if age not in mapping
                )
                
            if 'years_to_peak' not in df.columns:
                # Years to peak (positive = before peak, negative = after peak)
                PEAK_AGE = 27
                df['years_to_peak'] = PEAK_AGE - df['age']
                
            if 'estimated_years_left' not in df.columns:
                # Years of career left (rough estimate)
                df['estimated_years_left'] = 35 - df['age']
                df.loc[df['estimated_years_left'] < 0, 'estimated_years_left'] = 0
        
        # Initialize predictions array
        predictions = np.zeros(len(df))
        position_used = np.zeros(len(df), dtype=bool)
        
        # If using position-specific models
        if self.position_specific and any(m is not None for m in self.position_models.values()):
            # Group by position category
            for position, position_df in df.groupby('position_category'):
                # If we have a trained model for this position
                if position in self.position_models and self.position_models[position] is not None:
                    position_model = self.position_models[position]['model']
                    position_scaler = self.position_models[position]['scaler']
                    position_features = self.position_models[position]['features']
                    
                    # Get indices for this position
                    position_indices = df.index[df['position_category'] == position]
                    
                    # Verify all features are numeric
                    non_numeric_features = []
                    for feature in position_features:
                        if feature in df.columns and not pd.api.types.is_numeric_dtype(df[feature]):
                            non_numeric_features.append(feature)
                            logger.warning(f"Position {position} - Feature '{feature}' is not numeric: {df[feature].dtype}")
                    
                    if non_numeric_features:
                        logger.warning(f"Position {position} - Removing non-numeric features for prediction: {non_numeric_features}")
                        # Remove non-numeric features
                        position_features = [f for f in position_features if f not in non_numeric_features]
                        if not position_features:
                            logger.error(f"Position {position} - No numeric features left for prediction!")
                            continue
                    
                    # Extract features for this position
                    X_pos = df.loc[position_indices, position_features]
                    
                    # Scale features
                    X_pos_scaled = position_scaler.transform(X_pos)
                    
                    # Make position-specific predictions
                    pos_predictions = position_model.predict(X_pos_scaled)
                    
                    # Store predictions
                    predictions[position_indices] = pos_predictions
                    position_used[position_indices] = True
                    
                    logger.info(f"Used position-specific model for {position}: {len(position_indices)} players")
        
        # For any players that didn't use a position-specific model, use the general model
        if not position_used.all():
            # Get indices where position-specific model wasn't used
            general_indices = df.index[~position_used]
            
            # Extract features for general model
            features = self._get_features(df)
            
            # Verify all features are numeric
            non_numeric_features = []
            for feature in features:
                if feature in df.columns and not pd.api.types.is_numeric_dtype(df[feature]):
                    non_numeric_features.append(feature)
                    logger.warning(f"Feature '{feature}' is not numeric: {df[feature].dtype}")
            
            if non_numeric_features:
                logger.warning(f"Removing non-numeric features for prediction: {non_numeric_features}")
                # Remove non-numeric features
                features = [f for f in features if f not in non_numeric_features]
                if not features:
                    logger.error("No numeric features left for prediction!")
                    return np.zeros(len(df))
            
            X_general = df.loc[general_indices, features]
            
            # Scale features
            X_general_scaled = self.scaler.transform(X_general)
            
            # Make general predictions
            general_predictions = self.model.predict(X_general_scaled)
            
            # Store predictions
            predictions[general_indices] = general_predictions
            
            logger.info(f"Used general model for {len(general_indices)} players")
        
        # Apply age adjustments to the predictions if enabled
        if self.age_adjusted:
            # The age factor represents the player's value relative to their peak age value
            # A younger player with high potential will have a higher predicted value in the future
            # An older player will see their value decline over time
            # We use this to adjust our current prediction to get a more accurate present value
            
            # Age adjustment - multiply by age factor to account for age profile
            predictions_with_age = predictions * df['age_factor'].values
            
            # For young players (< 23), boost the value if stats are already good (indicating high potential)
            young_boost_indices = df.index[df['age'] < 23]
            if len(young_boost_indices) > 0:
                # For young players with good stats, there's high potential
                # We apply a "potential multiplier" that boosts value based on current performance
                # The better they perform at a young age, the higher their value
                potential_multiplier = np.ones(len(df))
                
                # Calculate performance score here to boost young players with high stats
                performance_score = np.zeros(len(df))
                # (in a real system, we'd have a better performance calculation here)
                
                # Apply the potential boost
                potential_multiplier[young_boost_indices] = 1.0 + (
                    0.2 * performance_score[young_boost_indices]
                )
                
                # Apply the multiplier
                predictions_with_age = predictions_with_age * potential_multiplier
            
            # Now, predictions_with_age has the age-adjusted values
            return predictions_with_age
        
        return predictions
    
    def save_model(self):
        """Save trained model to disk"""
        if self.model is None:
            logger.error("No model to save")
            return False
            
        model_path = os.path.join(MODEL_DIR, f"{self.model_type}_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{self.model_type}_scaler.pkl")
        metrics_path = os.path.join(MODEL_DIR, f"{self.model_type}_metrics.pkl")
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
                
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            with open(metrics_path, 'wb') as f:
                pickle.dump(self.metrics, f)
                
            logger.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self):
        """Load trained model from disk"""
        model_path = os.path.join(MODEL_DIR, f"{self.model_type}_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{self.model_type}_scaler.pkl")
        metrics_path = os.path.join(MODEL_DIR, f"{self.model_type}_metrics.pkl")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
            
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                
            with open(metrics_path, 'rb') as f:
                self.metrics = pickle.load(f)
                
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_optimal_weights(self):
        """Calculate optimal stat weights based on feature importance
        
        Returns:
            dict: Optimal weights for each feature
        """
        if self.feature_importance is None:
            logger.error("No feature importance available")
            return None
            
        # Normalize importance to sum to 1.0
        normalized = self.feature_importance.copy()
        normalized['importance'] = normalized['importance'] / normalized['importance'].sum()
        
        # Convert to dictionary
        weights = dict(zip(normalized['feature'], normalized['importance']))
        
        # Scale up the values to make them more intuitive (0-5 range)
        max_importance = max(weights.values())
        for key in weights:
            weights[key] = (weights[key] / max_importance) * 5
        
        return weights
    
    def analyze_model_performance(self, plot=False):
        """Analyze model performance on historical data
        
        Args:
            plot (bool): Whether to generate visualization plots
            
        Returns:
            pd.DataFrame: Performance metrics by season
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return None
            
        # Get all seasons
        seasons = self._get_all_seasons()
        
        results = []
        for season in seasons:
            try:
                # Get data for this season
                season_data = db.get_players_with_stats(season)
                
                if season_data.empty:
                    logger.warning(f"No data found for season {season}. Skipping.")
                    continue
                
                # Add missing age-related features if needed
                if self.age_adjusted:
                    # Add position category
                    if 'position_category' not in season_data.columns:
                        season_data['position_category'] = season_data['position'].apply(
                            lambda pos: self.position_categories.get(pos, 'midfielder')
                        )
                    
                    # Add age group
                    if 'age_group' not in season_data.columns:
                        season_data['age_group'] = pd.cut(
                            season_data['age'], 
                            bins=[15, 21, 25, 29, 33, 40], 
                            labels=['youth', 'developing', 'prime', 'experienced', 'veteran']
                        )
                    
                    # Add age factor
                    if 'age_factor' not in season_data.columns:
                        season_data['age_factor'] = season_data['age'].apply(
                            lambda age: self.age_adjustment.get(age, 1.0)
                        )
                    
                    # Add years to peak and estimated years left
                    if 'years_to_peak' not in season_data.columns:
                        PEAK_AGE = 27
                        season_data['years_to_peak'] = PEAK_AGE - season_data['age']
                    
                    if 'estimated_years_left' not in season_data.columns:
                        season_data['estimated_years_left'] = 35 - season_data['age']
                        season_data.loc[season_data['estimated_years_left'] < 0, 'estimated_years_left'] = 0
                
                # Get features for this season's data
                features = self._get_features(season_data)
                
                # Verify all model features are present
                # This is crucial - the features must match those used during training
                model_features = []
                if hasattr(self.model, 'feature_names_in_'):
                    model_features = self.model.feature_names_in_
                    logger.info(f"Model was trained with {len(model_features)} features")
                    
                    # Check for missing features
                    missing_features = [f for f in model_features if f not in features]
                    if missing_features:
                        logger.warning(f"Missing features in evaluation data: {missing_features}")
                        logger.warning("Skipping this season due to feature mismatch")
                        continue
                
                # Remove non-numeric features
                non_numeric_features = []
                for feature in features:
                    if not pd.api.types.is_numeric_dtype(season_data[feature]):
                        non_numeric_features.append(feature)
                
                if non_numeric_features:
                    logger.warning(f"Removing non-numeric features: {non_numeric_features}")
                    features = [f for f in features if f not in non_numeric_features]
                
                # Verify we have features left
                if not features:
                    logger.error(f"No valid features for season {season}. Skipping.")
                    continue
                
                logger.info(f"Evaluating model on season {season} with {len(features)} features")
                
                # Get feature data and actual values
                X = season_data[features]
                y_actual = season_data['market_value']
                
                # Handle missing values
                if X.isnull().any().any():
                    logger.warning(f"Found missing values in features for season {season}. Filling with median.")
                    X = X.fillna(X.median())
                
                # Scale features
                X_scaled = self.scaler.transform(X)
                
                # Make predictions
                y_pred = self.model.predict(X_scaled)
            except Exception as e:
                logger.error(f"Error analyzing season {season}: {str(e)}")
                continue
            
            try:
                # Calculate metrics
                mae = mean_absolute_error(y_actual, y_pred)
                rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
                r2 = r2_score(y_actual, y_pred)
                
                # Average % error (with protection against division by zero)
                with np.errstate(divide='ignore', invalid='ignore'):
                    pct_errors = np.abs(y_pred - y_actual) / y_actual * 100
                    # Replace infinity and NaN with a high value (1000%)
                    pct_errors = np.nan_to_num(pct_errors, nan=1000, posinf=1000)
                    pct_error = np.mean(pct_errors)
                
                logger.info(f"Season {season} metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, Error: {pct_error:.2f}%")
                
                results.append({
                    'season': season,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'pct_error': pct_error,
                    'sample_count': len(y_actual)
                })
            except Exception as e:
                logger.error(f"Error calculating metrics for season {season}: {str(e)}")
        
        # Check if we have any results
        if not results:
            logger.error("No valid results for any season. Analysis failed.")
            return None
        
        df_results = pd.DataFrame(results)
        
        # Try to create visualization plots
        if plot and not df_results.empty:
            try:
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                
                plt.figure(figsize=(12, 8))
                
                # Plot R² by season
                plt.subplot(2, 2, 1)
                plt.plot(df_results['season'], df_results['r2'], marker='o')
                plt.title('R² by Season')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # Plot % Error by season
                plt.subplot(2, 2, 2)
                plt.plot(df_results['season'], df_results['pct_error'], marker='o', color='red')
                plt.title('% Error by Season')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # Plot MAE by season
                plt.subplot(2, 2, 3)
                plt.plot(df_results['season'], df_results['mae'], marker='o', color='green')
                plt.title('MAE by Season')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # Plot sample count by season
                plt.subplot(2, 2, 4)
                plt.bar(df_results['season'], df_results['sample_count'], color='blue', alpha=0.7)
                plt.title('Sample Count by Season')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save the figure
                plot_path = os.path.join(MODEL_DIR, 'model_performance.png')
                plt.savefig(plot_path)
                logger.info(f"Performance plots saved to {plot_path}")
            except Exception as e:
                logger.error(f"Error creating performance plots: {str(e)}")
        
        return df_results

# Helper functions
def train_multiple_models():
    """Train and compare multiple ML models"""
    try:
        model_types = ["random_forest", "gradient_boosting", "ridge", "lasso"]
        results = {}
        successful_models = []
        
        for model_type in model_types:
            try:
                logger.info(f"Training {model_type} model...")
                model = PlayerValueModel(model_type=model_type)
                metrics = model.train(save_model=True)
                
                if metrics is not None:
                    results[model_type] = metrics
                    successful_models.append(model_type)
                    logger.info(f"{model_type} model trained successfully. Test R²: {metrics['test_r2']:.4f}")
                else:
                    logger.warning(f"Failed to train {model_type} model, no metrics returned")
            except Exception as e:
                logger.error(f"Error training {model_type} model: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        if not results:
            logger.error("No models trained successfully")
            return None
            
        # Compare results for successful models
        try:
            comparison_data = {}
            for model_type in successful_models:
                metrics = results[model_type]
                comparison_data[model_type] = {
                    'Test MAE': metrics['test_mae'],
                    'Test RMSE': metrics['test_rmse'],
                    'Test R²': metrics['test_r2']
                }
            
            comparison = pd.DataFrame(comparison_data)
            
            logger.info("Model comparison:")
            logger.info(comparison)
            
            # Save comparison
            comparison_path = os.path.join(MODEL_DIR, 'model_comparison.csv')
            comparison.to_csv(comparison_path)
            logger.info(f"Model comparison saved to {comparison_path}")
            
            # Identify best model
            best_model = max(successful_models, key=lambda m: results[m]['test_r2'])
            logger.info(f"Best model: {best_model} (R²: {results[best_model]['test_r2']:.4f})")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    except Exception as e:
        logger.error(f"Error in train_multiple_models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def predict_current_values(position_specific=True, age_adjusted=True, season=None):
    """Predict market values for current season players and calculate value ratio
    
    Args:
        position_specific (bool): Whether to use position-specific models
        age_adjusted (bool): Whether to use age adjustments
        season (str): Season to predict for (defaults to most recent)
        
    Returns:
        pd.DataFrame: Player data with predictions and value ratios
    """
    try:
        # Use the best model (random forest or gradient boosting typically performs best)
        model = PlayerValueModel(
            model_type="random_forest", 
            position_specific=position_specific, 
            age_adjusted=age_adjusted
        )
        
        # Try to load existing model, train if not available
        if not model.load_model():
            logger.info("No pre-trained model found. Training new model...")
            model.train()
        
        # Determine current season if not specified
        if season is None:
            all_seasons = model._get_all_seasons()
            if all_seasons:
                season = all_seasons[-1]  # Most recent season
            else:
                season = "2023-2024"  # Default
        
        logger.info(f"Making predictions for season: {season}")
        
        # Get current season data
        player_data = db.get_players_with_stats(season)
        
        if player_data.empty:
            logger.error(f"No data for season {season}")
            return None
            
        # Add debugging logs
        logger.info(f"Player data columns: {player_data.columns.tolist()}")
        logger.info(f"Feature types: {player_data.dtypes}")
        logger.info(f"Position values: {player_data['position'].unique()}")
        
        # Add position category if not present
        if 'position_category' not in player_data.columns:
            player_data['position_category'] = player_data['position'].apply(
                lambda pos: model.position_categories.get(pos, 'midfielder')
            )
            
        # Add age-related features if needed
        if age_adjusted:
            # Add age group
            if 'age_group' not in player_data.columns:
                player_data['age_group'] = pd.cut(
                    player_data['age'], 
                    bins=[15, 21, 25, 29, 33, 40], 
                    labels=['youth', 'developing', 'prime', 'experienced', 'veteran']
                )
                
            # Add age factor
            if 'age_factor' not in player_data.columns:
                player_data['age_factor'] = player_data['age'].apply(
                    lambda age: model.age_adjustment.get(age, 1.0)
                )
                
            # Add years to peak and estimated years left
            if 'years_to_peak' not in player_data.columns:
                PEAK_AGE = 27
                player_data['years_to_peak'] = PEAK_AGE - player_data['age']
                
            if 'estimated_years_left' not in player_data.columns:
                player_data['estimated_years_left'] = 35 - player_data['age']
                player_data.loc[player_data['estimated_years_left'] < 0, 'estimated_years_left'] = 0
        
        # Make predictions
        logger.info("Making value predictions...")
        predicted_values = model.predict(player_data)
        
        # Add predictions to dataframe
        player_data['predicted_value'] = predicted_values
        
    except Exception as e:
        logger.error(f"Error predicting values: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    try:
        # Calculate value ratio (predicted/actual)
        # A ratio > 1 means the player is predicted to be worth more than their current market value (undervalued)
        # A ratio < 1 means the player is predicted to be worth less than their current market value (overvalued)
        with np.errstate(divide='ignore', invalid='ignore'):
            player_data['value_ratio'] = player_data['predicted_value'] / player_data['market_value']
            # Replace inf and NaN with 0 for zero market values
            player_data['value_ratio'] = player_data['value_ratio'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Calculate the percentage difference
        with np.errstate(divide='ignore', invalid='ignore'):
            player_data['value_difference_pct'] = (
                (player_data['predicted_value'] - player_data['market_value']) / 
                player_data['market_value'] * 100
            )
            # Replace inf and NaN
            player_data['value_difference_pct'] = player_data['value_difference_pct'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Identify undervalued/overvalued players
        player_data['status'] = 'fair value'
        player_data.loc[player_data['value_ratio'] > 1.5, 'status'] = 'undervalued'
        player_data.loc[player_data['value_ratio'] < 0.7, 'status'] = 'overvalued'
        
        # Select relevant columns (handle case where columns might be missing)
        result_columns = [
            'id', 'name', 'age', 'position', 'nationality', 
            'club', 'league', 'market_value', 'predicted_value', 
            'value_ratio', 'value_difference_pct', 'status'
        ]
        
        # Add optional columns if they exist
        if 'age_group' in player_data.columns:
            result_columns.append('age_group')
        if 'position_category' in player_data.columns:
            result_columns.append('position_category')
        
        # Make sure all columns exist
        available_columns = [col for col in result_columns if col in player_data.columns]
        
        # Select available columns
        result = player_data[available_columns]
        
        # Sort by value ratio (descending)
        result = result.sort_values('value_ratio', ascending=False)
        
        logger.info(f"Prediction complete: Found {len(result)} players with predictions")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating value metrics: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def update_stat_weights_from_model():
    """Update the statistical weights in app.py based on model feature importance"""
    try:
        model = PlayerValueModel(model_type="random_forest")
        
        # Try to load existing model, train if not available
        if not model.load_model():
            logger.info("No pre-trained model found. Training new model...")
            training_result = model.train()
            if training_result is None:
                logger.error("Failed to train model for weight optimization")
                return False
        
        # Get optimal weights
        weights = model.get_optimal_weights()
        
        if not weights:
            logger.error("Failed to get optimal weights")
            return False
        
        # Format weights for Python code
        weights_str = "{\n"
        for feature, weight in sorted(weights.items()):
            weights_str += f"    '{feature}': {weight:.2f},\n"
        weights_str += "}"
        
        logger.info(f"Generated {len(weights)} new feature weights")
        
        # Save weights to a file for reference
        weights_path = os.path.join(MODEL_DIR, 'optimal_weights.py')
        with open(weights_path, 'w') as f:
            f.write(f"# Auto-generated weights from ML model\n# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nstat_weights = {weights_str}\n")
        
        logger.info(f"Weights saved to {weights_path}")
        
        # Also save as JSON for easier consumption by other systems
        import json
        json_path = os.path.join(MODEL_DIR, 'optimal_weights.json')
        with open(json_path, 'w') as f:
            json.dump(weights, f, indent=2)
        
        logger.info(f"Weights also saved as JSON to {json_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating weights from model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Example usage
    # train_multiple_models()
    # update_stat_weights_from_model()
    
    # Predict current values and show top undervalued players
    predictions = predict_current_values()
    if predictions is not None:
        print("Top 10 Undervalued Players:")
        print(predictions.head(10))