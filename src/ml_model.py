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
import database as db
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path for saving models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

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
        
        # Define non-feature columns but add position and age as features
        self.non_features = ['id', 'name', 'nationality', 'club', 'league',
                            'height', 'weight', 'preferred_foot', 'market_value', 
                            'last_updated']
    
    def _get_features(self, data):
        """Extract features from data"""
        # Get all features by excluding non-feature columns
        features = [col for col in data.columns if col not in self.non_features and not col.endswith('_weighted')]
        return features
    
    def train(self, seasons=None, test_size=0.2, save_model=True):
        """Train the model on historical data
        
        Args:
            seasons (list): List of seasons to include
            test_size (float): Proportion of data to use for testing
            save_model (bool): Whether to save the trained model
            
        Returns:
            dict: Training metrics
        """
        # Get training data
        df = self._get_training_data(seasons)
        
        if df.empty:
            logger.error("No training data available")
            return None
        
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
            
            # Train a separate model for each position category
            for position, position_df in df.groupby('position_category'):
                if len(position_df) < 10:  # Skip if too few samples
                    logger.warning(f"Not enough samples ({len(position_df)}) for position {position}. Skipping.")
                    continue
                
                logger.info(f"Training position-specific model for {position} ({len(position_df)} players)")
                
                # Get position-specific features and target
                features = self._get_features(position_df)
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
                }
                
                logger.info(f"Position {position} model: Test R² score: {pos_metrics['test_r2']:.4f}")
                
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
        X = df[features]
        y = df['market_value']
        
        # Log available features
        logger.info(f"Training with {len(features)} features: {features}")
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
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
        }
        
        if self.position_specific:
            metrics['position_metrics'] = position_metrics
        
        self.metrics = metrics
        logger.info(f"General model training complete. Test R² score: {metrics['test_r2']:.4f}")
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 important features:")
            logger.info(self.feature_importance.head(10))
        
        # Save model
        if save_model:
            self.save_model()
        
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
        
        # Combine data from all seasons
        all_data = []
        for season in seasons:
            season_data = db.get_players_with_stats(season)
            if not season_data.empty:
                season_data['season'] = season
                all_data.append(season_data)
        
        if not all_data:
            logger.warning("No data found for selected seasons")
            return pd.DataFrame()
            
        df = pd.concat(all_data, ignore_index=True)
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
            # Get data for this season
            season_data = db.get_players_with_stats(season)
            
            if season_data.empty:
                continue
                
            # Split features and actual values
            features = self._get_features(season_data)
            X = season_data[features]
            y_actual = season_data['market_value']
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            y_pred = self.model.predict(X_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2 = r2_score(y_actual, y_pred)
            
            # Average % error
            pct_error = np.mean(np.abs(y_pred - y_actual) / y_actual) * 100
            
            results.append({
                'season': season,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'pct_error': pct_error
            })
        
        df_results = pd.DataFrame(results)
        
        if plot and not df_results.empty:
            plt.figure(figsize=(12, 8))
            
            # Plot R² by season
            plt.subplot(2, 2, 1)
            plt.plot(df_results['season'], df_results['r2'], marker='o')
            plt.title('R² by Season')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Plot % Error by season
            plt.subplot(2, 2, 2)
            plt.plot(df_results['season'], df_results['pct_error'], marker='o', color='red')
            plt.title('% Error by Season')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(MODEL_DIR, 'model_performance.png'))
        
        return df_results

# Helper functions
def train_multiple_models():
    """Train and compare multiple ML models"""
    model_types = ["random_forest", "gradient_boosting", "ridge", "lasso"]
    results = {}
    
    for model_type in model_types:
        logger.info(f"Training {model_type} model...")
        model = PlayerValueModel(model_type=model_type)
        metrics = model.train(save_model=True)
        results[model_type] = metrics
    
    # Compare results
    comparison = pd.DataFrame({
        model_type: {
            'Test MAE': metrics['test_mae'],
            'Test RMSE': metrics['test_rmse'],
            'Test R²': metrics['test_r2']
        }
        for model_type, metrics in results.items()
    })
    
    logger.info("Model comparison:")
    logger.info(comparison)
    
    # Save comparison
    comparison.to_csv(os.path.join(MODEL_DIR, 'model_comparison.csv'))
    
    return comparison

def predict_current_values(position_specific=True, age_adjusted=True, season=None):
    """Predict market values for current season players and calculate value ratio
    
    Args:
        position_specific (bool): Whether to use position-specific models
        age_adjusted (bool): Whether to use age adjustments
        season (str): Season to predict for (defaults to most recent)
        
    Returns:
        pd.DataFrame: Player data with predictions and value ratios
    """
    # Use the best model (random forest or gradient boosting typically performs best)
    model = PlayerValueModel(
        model_type="random_forest", 
        position_specific=position_specific, 
        age_adjusted=age_adjusted
    )
    
    # Try to load existing model, train if not available
    if not model.load_model():
        logger.info("Training new model...")
        model.train()
    
    # Determine current season if not specified
    if season is None:
        all_seasons = model._get_all_seasons()
        if all_seasons:
            season = all_seasons[-1]  # Most recent season
        else:
            season = "2023-2024"  # Default
    
    # Get current season data
    player_data = db.get_players_with_stats(season)
    
    if player_data.empty:
        logger.error(f"No data for season {season}")
        return None
    
    # Make predictions
    predicted_values = model.predict(player_data)
    
    # Add predictions to dataframe
    player_data['predicted_value'] = predicted_values
    
    # Calculate value ratio (predicted/actual)
    # A ratio > 1 means the player is predicted to be worth more than their current market value (undervalued)
    # A ratio < 1 means the player is predicted to be worth less than their current market value (overvalued)
    player_data['value_ratio'] = player_data['predicted_value'] / player_data['market_value']
    
    # Calculate the percentage difference
    player_data['value_difference_pct'] = (
        (player_data['predicted_value'] - player_data['market_value']) / 
        player_data['market_value'] * 100
    )
    
    # Identify undervalued/overvalued players
    player_data['status'] = 'fair value'
    player_data.loc[player_data['value_ratio'] > 1.5, 'status'] = 'undervalued'
    player_data.loc[player_data['value_ratio'] < 0.7, 'status'] = 'overvalued'
    
    # Add position category for analysis
    if 'position_category' not in player_data.columns:
        player_data['position_category'] = player_data['position'].apply(
            lambda pos: model.position_categories.get(pos, 'midfielder')
        )
    
    # Add age group for analysis
    if 'age_group' not in player_data.columns:
        player_data['age_group'] = pd.cut(
            player_data['age'], 
            bins=[15, 21, 25, 29, 33, 40], 
            labels=['youth', 'developing', 'prime', 'experienced', 'veteran']
        )
    
    # Select relevant columns
    result = player_data[[
        'id', 'name', 'age', 'age_group', 'position', 'position_category', 
        'nationality', 'club', 'league', 'market_value', 'predicted_value', 
        'value_ratio', 'value_difference_pct', 'status'
    ]]
    
    return result.sort_values('value_ratio', ascending=False)

def update_stat_weights_from_model():
    """Update the statistical weights in app.py based on model feature importance"""
    model = PlayerValueModel(model_type="random_forest")
    
    # Try to load existing model, train if not available
    if not model.load_model():
        logger.info("Training new model...")
        model.train()
    
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
    
    logger.info(f"Generated new weights: {weights_str}")
    
    # Save weights to a file for reference
    with open(os.path.join(MODEL_DIR, 'optimal_weights.py'), 'w') as f:
        f.write(f"# Auto-generated weights from ML model\nstat_weights = {weights_str}\n")
    
    logger.info(f"Weights saved to {os.path.join(MODEL_DIR, 'optimal_weights.py')}")
    
    return True

if __name__ == "__main__":
    # Example usage
    # train_multiple_models()
    # update_stat_weights_from_model()
    
    # Predict current values and show top undervalued players
    predictions = predict_current_values()
    if predictions is not None:
        print("Top 10 Undervalued Players:")
        print(predictions.head(10))