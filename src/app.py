# backend/app.py
import pandas as pd
import numpy as np
import json
import math
import logging
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
import database as db

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NumPy types and data cleaning
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle NaN, Infinity, etc.
            if np.isnan(obj):
                return None
            if np.isinf(obj):
                return None if obj < 0 else 1e38  # Use a very large number for +inf, null for -inf
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)
    
    def encode(self, obj):
        # Clean up any Python native NaN/inf values before encoding
        if isinstance(obj, dict):
            cleaned_dict = {}
            for k, v in obj.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    cleaned_dict[k] = None
                else:
                    cleaned_dict[k] = v
            return super(NumpyEncoder, self).encode(cleaned_dict)
        elif isinstance(obj, list):
            cleaned_list = []
            for item in obj:
                if isinstance(item, dict):
                    cleaned_dict = {}
                    for k, v in item.items():
                        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                            cleaned_dict[k] = None
                        else:
                            cleaned_dict[k] = v
                    cleaned_list.append(cleaned_dict)
                else:
                    cleaned_list.append(item)
            return super(NumpyEncoder, self).encode(cleaned_list)
        return super(NumpyEncoder, self).encode(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder
CORS(app, resources={r"/*": {"origins": "*"}})

# Serve index.html at the root URL
@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')

# Define weights for each statistic (adjust as needed)
stat_weights = {
    # Offensive metrics
    'goals': 1.0,
    'assists': 0.8,
    'xg': 0.9,
    'xa': 0.7,
    'npxg': 0.8,
    'sca': 0.6,
    'gca': 0.9,
    'shots': 0.3,
    'shots_on_target': 0.5,
    'progressive_carries': 0.5,
    'progressive_passes': 0.5,
    'penalty_box_touches': 0.4,
    
    # Possession metrics
    'passes_completed': 0.3,
    'passes_attempted': 0.1,
    'pass_completion_pct': 0.4,
    'progressive_passes_received': 0.4,
    'dribbles_completed': 0.5,
    
    # Defensive metrics
    'tackles': 0.4,
    'tackles_won': 0.5,
    'interceptions': 0.4,
    'blocks': 0.3,
    'pressures': 0.3,
    'pressure_success_rate': 0.4,
    'aerial_duels_won': 0.3,
    
    # Per 90 metrics (these get higher weights)
    'goals_per90': 1.5,
    'assists_per90': 1.2,
    'xg_per90': 1.4,
    'xa_per90': 1.1,
    'npxg_per90': 1.3,
    'sca_per90': 1.0,
    'gca_per90': 1.4
}

def calculate_player_score(player_df):
    """Calculate a player's score based on weighted statistics"""
    # Initialize score columns
    player_df['total_score'] = 0
    
    # Group statistics by category for more balanced scoring
    stat_categories = {
        'offensive': ['goals', 'assists', 'xg', 'xa', 'npxg', 'sca', 'gca', 
                    'shots', 'shots_on_target', 'penalty_box_touches'],
        'possession': ['passes_completed', 'pass_completion_pct', 'progressive_passes',
                    'progressive_carries', 'progressive_passes_received', 
                    'dribbles_completed'],
        'defensive': ['tackles', 'tackles_won', 'interceptions', 'blocks', 
                    'pressures', 'pressure_success_rate', 'aerial_duels_won'],
        'per90': ['goals_per90', 'assists_per90', 'xg_per90', 'xa_per90', 
                'npxg_per90', 'sca_per90', 'gca_per90']
    }
    
    # Calculate score for each category separately
    for category, stats in stat_categories.items():
        category_score = 0
        available_stats = [s for s in stats if s in player_df.columns]
        
        if not available_stats:
            continue
            
        # Apply weights to each statistic in this category
        for stat in available_stats:
            if stat in stat_weights:
                # Normalize the stat value using min-max scaling within its category
                min_val = player_df[stat].min()
                max_val = player_df[stat].max()
                
                # Avoid division by zero
                if max_val == min_val:
                    normalized_value = 0
                else:
                    normalized_value = (player_df[stat] - min_val) / (max_val - min_val)
                
                # Add weighted statistic to category score
                weight = stat_weights.get(stat, 0.5)  # Default weight if not specified
                player_df[f'{stat}_weighted'] = normalized_value * weight
                category_score += player_df[f'{stat}_weighted']
        
        # Store category score
        if len(available_stats) > 0:
            # Normalize by number of stats to avoid bias toward categories with more stats
            player_df[f'{category}_score'] = category_score / len(available_stats) * 10
            player_df['total_score'] += player_df[f'{category}_score']
    
    # Scale the final score to a 0-100 range for easier interpretation
    min_score = player_df['total_score'].min()
    max_score = player_df['total_score'].max()
    
    if max_score > min_score:
        player_df['performance_score'] = 50 + 50 * (player_df['total_score'] - min_score) / (max_score - min_score)
    else:
        player_df['performance_score'] = 50
    
    return player_df['performance_score']

def calculate_market_value_ratio(row):
    """Calculate ratio of performance score to market value
    
    A higher ratio means the player is potentially undervalued:
    - ratio > 2.0: Significantly undervalued 
    - 0.5 <= ratio <= 2.0: Fair value
    - ratio < 0.5: Overvalued
    """
    if row['market_value'] == 0:  # Avoid division by zero
        return 0
    
    # Calculate expected market value based on performance score
    # Use a baseline of €1M per point for performance scores around 50
    expected_value = (row['performance_score'] ** 1.5) * 200000
    
    # Ratio of expected value to actual market value
    # Values > 1 mean player is potentially undervalued
    ratio = expected_value / row['market_value']
    
    return ratio

@app.route('/players')
def get_players():
    try:
        # Get player data with statistics from the database
        player_data = db.get_players_with_stats()
        
        if player_data.empty:
            return jsonify([])
        
        # Calculate performance score based on weighted statistics
        player_data['performance_score'] = calculate_player_score(player_data)
        
        # Use the unified ML model for value predictions
        import unified_ml_model
        
        # First check if we have analysis results from transfer value analysis
        METRICS_DIR = os.path.join(os.path.dirname(__file__), 'metrics')
        analysis_files = [f for f in os.listdir(METRICS_DIR) if f.startswith('value_analysis_')]
        
        if analysis_files:
            # Get the most recent analysis file
            latest_analysis = max(
                [os.path.join(METRICS_DIR, f) for f in analysis_files],
                key=os.path.getmtime
            )
            
            # Load the analysis results
            try:
                with open(latest_analysis, 'r') as f:
                    analysis = json.load(f)
                    
                # Convert to DataFrame
                if 'all_players' in analysis and analysis['all_players']:
                    import pandas as pd
                    predictions_df = pd.DataFrame(analysis['all_players'])
                    logger.info(f"Loaded predictions from existing analysis: {latest_analysis}")
                else:
                    # Generate new predictions
                    predictions_df = unified_ml_model.predict_current_values(
                        position_specific=True,
                        age_adjusted=True
                    )
            except Exception as e:
                logger.error(f"Error loading analysis file: {e}")
                # Fall back to generating new predictions
                predictions_df = unified_ml_model.predict_current_values(
                    position_specific=True,
                    age_adjusted=True
                )
        else:
            # Generate new predictions
            predictions_df = unified_ml_model.predict_current_values(
                position_specific=True,
                age_adjusted=True
            )
        
        if predictions_df is not None and not predictions_df.empty:
            # Merge predictions with player data
            merge_columns = ['id', 'predicted_value', 'value_ratio', 'value_difference']
            player_data = player_data.merge(
                predictions_df[merge_columns],
                on='id',
                how='left'
            )
        else:
            # Fall back to simpler calculation if ML predictions fail
            player_data['value_ratio'] = player_data.apply(calculate_market_value_ratio, axis=1)
            player_data['predicted_value'] = player_data['market_value'] * player_data['value_ratio']
            player_data['value_difference'] = player_data['predicted_value'] - player_data['market_value']
        
        # Calculate a combined score
        player_data['combined_score'] = (
            player_data['performance_score'] * 0.7 +  # 70% performance
            player_data['value_ratio'] * 0.3          # 30% value for money
        )
        
        # Select and rename columns for the API response
        result_df = player_data[['id', 'name', 'age', 'position', 'club', 'league', 
                               'market_value', 'performance_score', 'value_ratio', 
                               'combined_score', 'goals', 'assists', 'xg', 'xa',
                               'sca', 'gca', 'tackles', 'interceptions', 'predicted_value']]
        
        # Sort by combined score (descending)
        sorted_players = result_df.sort_values(by='combined_score', ascending=False)
        
        # Convert to JSON
        players_json = sorted_players.to_dict(orient='records')
        
        return jsonify(players_json)
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Error getting player data", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/player/<int:player_id>')
def get_player_detail(player_id):
    # Get detailed player data
    player_info = db.get_all_players()[db.get_all_players()['id'] == player_id]
    player_stats = db.get_player_stats(player_id)
    
    if player_info.empty:
        return jsonify({"error": "Player not found"}), 404
    
    # Combine player info with stats
    player_info = player_info.to_dict(orient='records')[0]
    player_stats = player_stats.to_dict(orient='records')
    
    result = {
        "player_info": player_info,
        "player_stats": player_stats
    }
    
    return jsonify(result)

@app.route('/top-undervalued')
def get_top_undervalued():
    try:
        # Get player data with statistics
        player_data = db.get_players_with_stats()
        
        if player_data.empty:
            return jsonify([])
        
        # Calculate performance score based on weighted statistics
        player_data['performance_score'] = calculate_player_score(player_data)
        
        # Use the unified ML model for value predictions
        import unified_ml_model
        
        # First check if we have analysis results from transfer value analysis
        METRICS_DIR = os.path.join(os.path.dirname(__file__), 'metrics')
        analysis_files = [f for f in os.listdir(METRICS_DIR) if f.startswith('value_analysis_')]
        
        if analysis_files:
            # Get the most recent analysis file
            latest_analysis = max(
                [os.path.join(METRICS_DIR, f) for f in analysis_files],
                key=os.path.getmtime
            )
            
            # Load the analysis results
            try:
                with open(latest_analysis, 'r') as f:
                    analysis = json.load(f)
                    
                # Convert to DataFrame
                if 'all_players' in analysis and analysis['all_players']:
                    import pandas as pd
                    predictions_df = pd.DataFrame(analysis['all_players'])
                    logger.info(f"Loaded predictions from existing analysis: {latest_analysis}")
                else:
                    # Generate new predictions
                    predictions_df = unified_ml_model.predict_current_values(
                        position_specific=True,
                        age_adjusted=True
                    )
            except Exception as e:
                logger.error(f"Error loading analysis file: {e}")
                # Fall back to generating new predictions
                predictions_df = unified_ml_model.predict_current_values(
                    position_specific=True,
                    age_adjusted=True
                )
        else:
            # Generate new predictions
            predictions_df = unified_ml_model.predict_current_values(
                position_specific=True,
                age_adjusted=True
            )
        
        if predictions_df is not None and not predictions_df.empty:
            # Get undervalued players from predictions
            undervalued_players = predictions_df[predictions_df['status'] == 'Undervalued'].copy()
            
            if len(undervalued_players) >= 5:
                # Merge with original player data to get performance scores
                undervalued_players = undervalued_players.merge(
                    player_data[['id', 'performance_score']], 
                    on='id',
                    how='left'
                )
                
                # Filter by average performance
                avg_performance = player_data['performance_score'].mean()
                undervalued_players = undervalued_players[
                    undervalued_players['performance_score'] >= avg_performance
                ]
            else:
                # Fall back to original method if not enough players from ML
                player_data['value_ratio'] = player_data.apply(calculate_market_value_ratio, axis=1)
                avg_performance = player_data['performance_score'].mean()
                undervalued_players = player_data[
                    (player_data['value_ratio'] > 1.5) & 
                    (player_data['performance_score'] >= avg_performance)
                ]
        else:
            # Fall back to original method if ML fails
            player_data['value_ratio'] = player_data.apply(calculate_market_value_ratio, axis=1)
            avg_performance = player_data['performance_score'].mean()
            undervalued_players = player_data[
                (player_data['value_ratio'] > 1.5) & 
                (player_data['performance_score'] >= avg_performance)
            ]
        
        # If we don't have enough players that meet the strict criteria, fall back to top value ratios
        if len(undervalued_players) < 5:
            # Just get players with above average performance, sorted by value ratio
            undervalued_players = player_data[player_data['performance_score'] >= avg_performance]
        
        # Sort by value ratio (highest first - most undervalued)
        top_undervalued = undervalued_players.sort_values(by='value_ratio', ascending=False).head(10)
        
        # Required columns for the response
        needed_columns = ['id', 'name', 'age', 'position', 'club', 'league',
                         'market_value', 'performance_score', 'value_ratio', 'predicted_value']
        
        # Ensure all required columns exist
        for col in needed_columns:
            if col not in top_undervalued.columns:
                if col == 'predicted_value' and 'market_value' in top_undervalued.columns and 'value_ratio' in top_undervalued.columns:
                    top_undervalued['predicted_value'] = top_undervalued['market_value'] * top_undervalued['value_ratio']
                else:
                    top_undervalued[col] = None
        
        # Select columns for the API response
        result_df = top_undervalued[needed_columns]
        
        # Convert to JSON
        undervalued_json = result_df.to_dict(orient='records')
        
        return jsonify(undervalued_json)
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Error getting undervalued players", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/stats-distribution')
def get_stats_distribution():
    """Return statistics distribution for visualization"""
    player_stats = db.get_player_stats()
    
    if player_stats.empty:
        return jsonify([])
    
    # Get distribution for key metrics
    key_metrics = ['goals', 'assists', 'xg', 'xa', 'sca', 'gca', 
                   'pass_completion_pct', 'tackles', 'interceptions']
    
    result = {}
    for metric in key_metrics:
        if metric in player_stats.columns:
            # Calculate percentiles
            percentiles = [10, 25, 50, 75, 90]
            values = [player_stats[metric].quantile(p/100) for p in percentiles]
            
            result[metric] = {
                'min': float(player_stats[metric].min()),
                'max': float(player_stats[metric].max()),
                'mean': float(player_stats[metric].mean()),
                'median': float(player_stats[metric].median()),
                'percentiles': {str(p): float(v) for p, v in zip(percentiles, values)}
            }
    
    return jsonify(result)

@app.route('/ml-predictions')
def get_ml_predictions():
    """Get player market value predictions from ML model"""
    try:
        # Import unified ML module
        import unified_ml_model
        from flask import request
        
        # Get query parameters
        position_specific = request.args.get('position_specific', 'true').lower() == 'true'
        age_adjusted = request.args.get('age_adjusted', 'true').lower() == 'true'
        
        # Get predictions with specified parameters
        predictions = unified_ml_model.predict_current_values(
            position_specific=position_specific,
            age_adjusted=age_adjusted
        )
        
        if predictions is None:
            return jsonify({"error": "Failed to generate predictions"}), 500
            
        # Convert to JSON
        predictions_json = predictions.to_dict(orient='records')
        
        # Add metadata about the prediction settings
        response = {
            "settings": {
                "position_specific": position_specific,
                "age_adjusted": age_adjusted
            },
            "predictions": predictions_json
        }
        
        return jsonify(response)
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Error generating predictions", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/train-model')
def train_model():
    """Train the ML model and return performance metrics"""
    try:
        # Import unified ML module
        import unified_ml_model
        from flask import request
        
        # Get optional parameters
        tag = request.args.get('tag', 'baseline')
        model_type = request.args.get('model_type', 'random_forest')
        position_specific = request.args.get('position_specific', 'true').lower() == 'true'
        age_adjusted = request.args.get('age_adjusted', 'true').lower() == 'true'
        time_series = request.args.get('time_series', 'true').lower() == 'true'
        
        # Create model with specified parameters
        model = unified_ml_model.UnifiedPlayerValueModel(
            model_type=model_type,
            position_specific=position_specific,
            age_adjusted=age_adjusted,
            time_series=time_series
        )
        
        # Train model with tag for tracking
        metrics = model.train(tag=tag)
        
        if metrics is None:
            return jsonify({"error": "Failed to train model"}), 500
            
        # Return metrics and performance
        result = {
            "training_settings": {
                "tag": tag,
                "model_type": model_type,
                "position_specific": position_specific,
                "age_adjusted": age_adjusted,
                "time_series": time_series
            },
            "training_metrics": metrics
        }
        
        # Try to retrieve and include comparison with previous runs
        try:
            previous_metrics = unified_ml_model.load_latest_metrics(model_type, tag="baseline")
            if previous_metrics and previous_metrics.get("metrics"):
                # Simple comparison of key metrics
                current = metrics
                previous = previous_metrics.get("metrics")
                
                comparison = {
                    "current": {
                        "r2": current.get("test_r2"),
                        "rmse": current.get("test_rmse"),
                        "mae": current.get("test_mae")
                    },
                    "previous": {
                        "r2": previous.get("test_r2"),
                        "rmse": previous.get("test_rmse"),
                        "mae": previous.get("test_mae")
                    },
                    "improvement": {
                        "r2": (current.get("test_r2", 0) - previous.get("test_r2", 0)) / max(abs(previous.get("test_r2", 1)), 0.001) * 100,
                        "rmse": (previous.get("test_rmse", 0) - current.get("test_rmse", 0)) / max(previous.get("test_rmse", 1), 0.001) * 100,
                        "mae": (previous.get("test_mae", 0) - current.get("test_mae", 0)) / max(previous.get("test_mae", 1), 0.001) * 100
                    }
                }
                
                result["comparison_with_baseline"] = comparison
        except Exception as e:
            import traceback
            result["comparison_error"] = str(e)
            result["comparison_traceback"] = traceback.format_exc()
        
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Error training model", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/compare-models')
def compare_models():
    """Train and compare multiple ML models"""
    try:
        # Import ML module
        import ml_model
        
        # Train and compare models
        comparison = ml_model.train_multiple_models()
        
        if comparison is None:
            return jsonify({"error": "Failed to compare models"}), 500
            
        # Convert to JSON
        result = comparison.to_dict()
        
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Error comparing models", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/update-weights')
def update_weights():
    """Update statistical weights based on ML model feature importance"""
    try:
        # Import ML module
        import ml_model
        
        # Update weights
        success = ml_model.update_stat_weights_from_model()
        
        if not success:
            return jsonify({"error": "Failed to update weights"}), 500
            
        return jsonify({"success": True, "message": "Weights updated successfully"})
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Error updating weights", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/compare-model-runs')
def compare_model_runs():
    """Compare ML model performance across different runs/data versions"""
    try:
        # Import ML module
        import ml_model
        from flask import request
        
        # Get query parameters
        model_type = request.args.get('model_type', 'random_forest')
        baseline_tag = request.args.get('baseline_tag', 'baseline')
        comparison_tag = request.args.get('comparison_tag')
        
        if not comparison_tag:
            return jsonify({"error": "comparison_tag parameter is required"}), 400
            
        # Load baseline and comparison metrics
        baseline_metrics = ml_model.load_latest_metrics(model_type, tag=baseline_tag)
        comparison_metrics = ml_model.load_latest_metrics(model_type, tag=comparison_tag)
        
        if not baseline_metrics:
            return jsonify({"error": f"No baseline metrics found for model {model_type} with tag {baseline_tag}"}), 404
            
        if not comparison_metrics:
            return jsonify({"error": f"No comparison metrics found for model {model_type} with tag {comparison_tag}"}), 404
            
        # Generate comparison
        comparison = ml_model.compare_metrics(
            comparison_metrics["metrics"], 
            baseline_metrics["metrics"],
            model_type
        )
        
        if not comparison:
            return jsonify({"error": "Failed to generate comparison"}), 500
            
            
        # Create response with detailed metrics
        response = {
            "settings": {
                "model_type": model_type,
                "baseline_tag": baseline_tag,
                "comparison_tag": comparison_tag,
                "baseline_timestamp": baseline_metrics["timestamp"],
                "comparison_timestamp": comparison_metrics["timestamp"]
            },
            "baseline_metrics": {
                "r2": baseline_metrics["metrics"]["test_r2"],
                "rmse": baseline_metrics["metrics"]["test_rmse"],
                "mae": baseline_metrics["metrics"]["test_mae"],
                "pct_error": baseline_metrics["metrics"].get("test_pct_error", "N/A"),
                "data_size": baseline_metrics["metrics"].get("data_size", "N/A"),
                "feature_count": baseline_metrics["metrics"].get("feature_count", "N/A")
            },
            "comparison_metrics": {
                "r2": comparison_metrics["metrics"]["test_r2"],
                "rmse": comparison_metrics["metrics"]["test_rmse"],
                "mae": comparison_metrics["metrics"]["test_mae"],
                "pct_error": comparison_metrics["metrics"].get("test_pct_error", "N/A"),
                "data_size": comparison_metrics["metrics"].get("data_size", "N/A"),
                "feature_count": comparison_metrics["metrics"].get("feature_count", "N/A")
            },
            "comparison": comparison
        }
        
        return jsonify(response)
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Error comparing model runs", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/transfer-value-predictions')
def get_transfer_value_predictions():
    """Get market value predictions using the enhanced transfer value model"""
    try:
        from flask import request
        import json
        import os
        import pandas as pd
        import numpy as np

        # See if we have a recent analysis
        METRICS_DIR = os.path.join(os.path.dirname(__file__), 'metrics')
        analysis_files = [f for f in os.listdir(METRICS_DIR) if f.startswith('value_analysis_')]
        
        if not analysis_files:
            # No analysis files found, let's run the analysis
            import unified_ml_model
            analysis_results = unified_ml_model.analyze_transfer_values()
            
            if analysis_results is None:
                return jsonify({"error": "Failed to generate transfer value analysis"}), 500
                
            # Check again for analysis files
            analysis_files = [f for f in os.listdir(METRICS_DIR) if f.startswith('value_analysis_')]
        
        if not analysis_files:
            return jsonify({"error": "No transfer value analysis found"}), 500
        
        # Get the most recent analysis file
        latest_analysis = max(
            [os.path.join(METRICS_DIR, f) for f in analysis_files],
            key=os.path.getmtime
        )
        
        # Load the analysis results
        with open(latest_analysis, 'r') as f:
            analysis = json.load(f)
        
        # Process results for the API
        undervalued_players = analysis.get('undervalued', [])
        overvalued_players = analysis.get('overvalued', [])
        fair_value_players = analysis.get('fair_value', [])
        all_players = analysis.get('all_players', [])
        stats = analysis.get('stats', {})
        
        # Enhance players with additional stats if missing
        if all_players:
            # Get player data from database for the current season
            player_data = db.get_players_with_stats()
            
            # Create a dictionary to map player IDs to their stat data
            player_stats = {}
            for _, player in player_data.iterrows():
                player_id = player.get('id')
                if player_id is not None:
                    player_stats[player_id] = player.to_dict()
            
            # Enhance each player with additional stats if missing
            for player in all_players:
                player_id = player.get('id')
                if player_id in player_stats:
                    # Add missing stats
                    for stat in ['goals', 'assists', 'xg', 'xa', 'sca', 'gca', 'tackles', 'interceptions', 'performance_score']:
                        if stat not in player and stat in player_stats[player_id]:
                            player[stat] = player_stats[player_id][stat]
        
        # Get query parameters
        status_filter = request.args.get('status', '').lower()  # 'undervalued', 'overvalued', 'fair'
        position = request.args.get('position', '').upper()     # 'CF', 'CM', etc.
        
        # Apply filters
        if status_filter == 'undervalued':
            filtered_players = undervalued_players
        elif status_filter == 'overvalued':
            filtered_players = overvalued_players
        elif status_filter == 'fair':
            filtered_players = fair_value_players
        else:
            filtered_players = all_players
            
        # Filter by position if specified
        if position:
            filtered_players = [p for p in filtered_players if p.get('position') == position]
        
        # Sort players by value ratio (most extreme differences first)
        if status_filter == 'overvalued':
            # For overvalued, lower ratio is more extreme
            sorted_players = sorted(filtered_players, key=lambda p: p.get('value_ratio', 1.0))
        else:
            # For undervalued and default, higher ratio is more extreme
            sorted_players = sorted(filtered_players, key=lambda p: p.get('value_ratio', 1.0), reverse=True)
        
        # Create response
        response = {
            "filters": {
                "status": status_filter,
                "position": position
            },
            "stats": stats,
            "analysis_date": analysis.get("analysis_date", 
                             os.path.basename(latest_analysis).split('_', 1)[1].split('.')[0]),  # Extract timestamp
            "players": sorted_players
        }
        
        return jsonify(response)
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Error getting transfer value predictions", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/analyze-transfer-values')
def analyze_transfer_values():
    """Run analysis on market values to find undervalued and overvalued players"""
    try:
        # Import unified ML module
        import unified_ml_model
        
        # Get force parameter (to force retraining)
        from flask import request
        force = request.args.get('force', 'false').lower() == 'true'
        
        if force:
            # If force is true, always run a new analysis
            analysis_results = unified_ml_model.analyze_transfer_values()
            analysis_success = analysis_results is not None
        else:
            # Check if we have a recent analysis (less than 1 hour old)
            METRICS_DIR = os.path.join(os.path.dirname(__file__), 'metrics')
            analysis_files = [f for f in os.listdir(METRICS_DIR) if f.startswith('value_analysis_')]
            
            if not analysis_files:
                # No analysis files found, run the analysis
                analysis_results = unified_ml_model.analyze_transfer_values()
                analysis_success = analysis_results is not None
            else:
                # Get the most recent analysis file
                latest_analysis = max(
                    [os.path.join(METRICS_DIR, f) for f in analysis_files],
                    key=os.path.getmtime
                )
                
                # Check if it's less than 1 hour old
                import time
                age_in_hours = (time.time() - os.path.getmtime(latest_analysis)) / 3600
                
                if age_in_hours > 1:
                    # Analysis is more than 1 hour old, run a new one
                    analysis_results = unified_ml_model.analyze_transfer_values()
                    analysis_success = analysis_results is not None
                else:
                    # Analysis is recent, use it
                    analysis_success = True
        
        if not analysis_success:
            return jsonify({"error": "Failed to analyze transfer values"}), 500
            
        # Redirect to the transfer-value-predictions endpoint
        from flask import redirect
        return redirect('/transfer-value-predictions')
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Error analyzing transfer values", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/compare-players')
def compare_players():
    """Compare players within the same position and age group"""
    try:
        # Import unified ML module and request
        import unified_ml_model
        from flask import request
        
        # Get query parameters
        position = request.args.get('position')
        age_group = request.args.get('age_group')
        player_id = request.args.get('player_id')
        
        if not (position or age_group or player_id):
            return jsonify({"error": "At least one filter parameter (position, age_group, or player_id) is required"}), 400
        
        # Get predictions with position and age adjustments
        predictions = unified_ml_model.predict_current_values(
            position_specific=True,
            age_adjusted=True
        )
        
        if predictions is None:
            return jsonify({"error": "Failed to generate predictions"}), 500
            
        # Filter by position category if specified
        if position:
            if position in predictions['position'].values:
                # Exact position match
                predictions = predictions[predictions['position'] == position]
            elif 'position_category' in predictions.columns and position in predictions['position_category'].values:
                # Position category match
                predictions = predictions[predictions['position_category'] == position]
            else:
                return jsonify({"error": f"Position '{position}' not found"}), 404
        
        # Filter by age group if specified
        if age_group:
            if 'age_group' not in predictions.columns:
                # Create age groups if not present
                predictions['age_group'] = pd.cut(
                    predictions['age'], 
                    bins=[15, 21, 25, 29, 33, 40], 
                    labels=['youth', 'developing', 'prime', 'experienced', 'veteran']
                )
                
            if age_group not in predictions['age_group'].values:
                return jsonify({"error": f"Age group '{age_group}' not found"}), 404
            predictions = predictions[predictions['age_group'] == age_group]
        
        # Get reference player if specified
        reference_player = None
        if player_id:
            try:
                player_id = int(player_id)
                reference_player = predictions[predictions['id'] == player_id]
                if reference_player.empty:
                    return jsonify({"error": f"Player with ID {player_id} not found"}), 404
                reference_player = reference_player.iloc[0].to_dict()
                
                # Convert any non-serializable values
                for k, v in list(reference_player.items()):
                    if isinstance(v, (pd.Series, pd.DataFrame)):
                        reference_player[k] = None
                    elif isinstance(v, np.ndarray):
                        reference_player[k] = v.tolist()
                    elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        reference_player[k] = None
            except ValueError:
                return jsonify({"error": f"Invalid player ID: {player_id}"}), 400
        
        # Sort by value ratio (most undervalued first)
        sorted_players = predictions.sort_values('value_ratio', ascending=False)
        
        # Convert to JSON
        players_json = sorted_players.to_dict(orient='records')
        
        # Create response
        response = {
            "filters": {
                "position": position,
                "age_group": age_group,
                "player_id": player_id
            },
            "reference_player": reference_player,
            "players": players_json
        }
        
        return jsonify(response)
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Error comparing players", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # Create database if it doesn't exist
    if not os.path.exists(db.DB_PATH):
        db.create_database()
        db.populate_database()
        
    app.run(debug=True)
