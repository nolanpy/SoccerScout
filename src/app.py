# backend/app.py
import pandas as pd
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
import database as db

app = Flask(__name__)
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
    score = 0
    
    # Apply weights to each statistic
    for stat, weight in stat_weights.items():
        if stat in player_df.columns:
            # Normalize the stat value using min-max scaling within its category
            min_val = player_df[stat].min()
            max_val = player_df[stat].max()
            
            # Avoid division by zero
            if max_val == min_val:
                normalized_value = 0
            else:
                normalized_value = (player_df[stat] - min_val) / (max_val - min_val)
            
            # Add weighted statistic to score
            player_df[f'{stat}_weighted'] = normalized_value * weight
            score += player_df[f'{stat}_weighted']
    
    return score

def calculate_market_value_ratio(row):
    """Calculate ratio of performance score to market value
    
    A higher ratio means the player is potentially undervalued:
    - ratio > 2.0: Significantly undervalued 
    - 0.5 <= ratio <= 2.0: Fair value
    - ratio < 0.5: Overvalued
    """
    if row['market_value'] == 0:  # Avoid division by zero
        return 0
    
    # Calculate ratio (higher ratio = potentially undervalued player)
    # Normalize market value to millions for more intuitive ratios
    return row['performance_score'] / (row['market_value'] / 1000000)

@app.route('/players')
def get_players():
    # Get player data with statistics from the database
    player_data = db.get_players_with_stats()
    
    if player_data.empty:
        return jsonify([])
    
    # Calculate performance score based on weighted statistics
    player_data['performance_score'] = calculate_player_score(player_data)
    
    # Calculate market value ratio to identify undervalued/overvalued players
    player_data['value_ratio'] = player_data.apply(calculate_market_value_ratio, axis=1)
    
    # Calculate a combined score
    player_data['combined_score'] = (
        player_data['performance_score'] * 0.7 +  # 70% performance
        player_data['value_ratio'] * 0.3          # 30% value for money
    )
    
    # Select and rename columns for the API response
    result_df = player_data[['id', 'name', 'age', 'position', 'club', 'league', 
                           'market_value', 'performance_score', 'value_ratio', 
                           'combined_score', 'goals', 'assists', 'xg', 'xa',
                           'sca', 'gca', 'tackles', 'interceptions']]
    
    # Sort by combined score (descending)
    sorted_players = result_df.sort_values(by='combined_score', ascending=False)
    
    # Convert to JSON
    players_json = sorted_players.to_dict(orient='records')
    
    return jsonify(players_json)

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
    # Get player data with statistics
    player_data = db.get_players_with_stats()
    
    if player_data.empty:
        return jsonify([])
    
    # Calculate performance score based on weighted statistics
    player_data['performance_score'] = calculate_player_score(player_data)
    
    # Calculate market value ratio (higher = more undervalued)
    player_data['value_ratio'] = player_data.apply(calculate_market_value_ratio, axis=1)
    
    # Consider a player undervalued if:
    # 1. Their value ratio is greater than 2.0 (same as the frontend threshold)
    # 2. They have at least average performance
    avg_performance = player_data['performance_score'].mean()
    undervalued = player_data[
        (player_data['value_ratio'] > 2.0) & 
        (player_data['performance_score'] >= avg_performance)
    ]
    
    # If we don't have enough players that meet the strict criteria, fall back to top value ratios
    if len(undervalued) < 5:
        # Just get players with above average performance, sorted by value ratio
        undervalued = player_data[player_data['performance_score'] >= avg_performance]
    
    # Sort by value ratio (highest first - most undervalued)
    top_undervalued = undervalued.sort_values(by='value_ratio', ascending=False).head(10)
    
    # Select columns for the API response
    result_df = top_undervalued[['id', 'name', 'age', 'position', 'club', 'league', 
                               'market_value', 'performance_score', 'value_ratio']]
    
    # Convert to JSON
    undervalued_json = result_df.to_dict(orient='records')
    
    return jsonify(undervalued_json)

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

if __name__ == '__main__':
    # Create database if it doesn't exist
    if not os.path.exists(db.DB_PATH):
        db.create_database()
        db.populate_database()
        
    app.run(debug=True)
