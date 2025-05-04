"""Market value prediction utility for SoccerScout

This script will:
1. Add the missing get_players_with_market_value_history function to database.py
2. Create a ML model to predict market values based on player stats
3. Update the player_market_values table with predicted values
"""

import os
import sqlite3
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
import database as db
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f'market_values_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.path.join(os.path.dirname(__file__), 'soccer_scout.db')

def add_missing_database_function():
    """Add the get_players_with_market_value_history function to database.py if not present"""
    logger.info("Adding get_players_with_market_value_history function to database.py...")
    
    db_file_path = os.path.join(os.path.dirname(__file__), 'database.py')
    
    # Check if the function already exists in database.py
    with open(db_file_path, 'r') as f:
        db_content = f.read()
        
    if "get_players_with_market_value_history" in db_content:
        logger.info("Function already exists in database.py")
        return True
    
    # Function to add
    function_code = """
def get_players_with_market_value_history(player_id=None, current_season="2023-2024"):
    \"\"\"Retrieve players with their market value history across seasons
    
    Args:
        player_id (int): Optional player ID to filter by
        current_season (str): Current season to use as reference
        
    Returns:
        pd.DataFrame: Player data with market value history
    \"\"\"
    conn = sqlite3.connect(DB_PATH)
    
    # Build the base player data query
    base_query = f'''
    SELECT 
        p.id, p.name, p.age, p.nationality, p.position, p.club, p.league, 
        p.market_value,
        ps.goals, ps.assists, ps.xg, ps.xa, ps.npxg, ps.sca, ps.gca,
        ps.shots, ps.shots_on_target, ps.progressive_carries, ps.progressive_passes,
        ps.penalty_box_touches, ps.passes_completed, ps.passes_attempted,
        ps.pass_completion_pct, ps.progressive_passes_received,
        ps.final_third_passes_completed, ps.final_third_passes_attempted,
        ps.dribbles_completed, ps.dribbles_attempted, ps.ball_recoveries,
        ps.tackles, ps.tackles_won, ps.interceptions, ps.blocks, ps.clearances,
        ps.pressures, ps.pressure_success_rate, ps.aerial_duels_won,
        ps.aerial_duels_total, ps.minutes_played, ps.games_played,
        ps.distance_covered, ps.high_intensity_runs, ps.yellow_cards, ps.red_cards,
        ps.goals_per90, ps.assists_per90, ps.xg_per90, ps.xa_per90,
        ps.npxg_per90, ps.sca_per90, ps.gca_per90
    FROM players p
    LEFT JOIN player_stats ps ON p.id = ps.player_id AND ps.season = '{current_season}'
    '''
    
    # Add player_id filter if provided
    if player_id:
        base_query += f" WHERE p.id = {player_id}"
    
    # Get base player data
    base_df = pd.read_sql_query(base_query, conn)
    
    if base_df.empty:
        conn.close()
        return pd.DataFrame()
    
    # Get market value history for all seasons
    market_values_query = f'''
    SELECT 
        player_id, 
        season, 
        market_value
    FROM player_market_values
    '''
    
    if player_id:
        market_values_query += f" WHERE player_id = {player_id}"
    
    market_values_df = pd.read_sql_query(market_values_query, conn)
    
    if market_values_df.empty:
        conn.close()
        return base_df
    
    # Pivot the market values to get a column for each season
    pivoted_values = market_values_df.pivot(
        index='player_id', 
        columns='season', 
        values='market_value'
    ).reset_index()
    
    # Rename the season columns
    for col in pivoted_values.columns:
        if col != 'player_id':
            pivoted_values.rename(columns={col: f'market_value_{col.replace("-", "_")}'}, inplace=True)
    
    # Merge base data with market value history
    result_df = base_df.merge(pivoted_values, left_on='id', right_on='player_id', how='left')
    
    # Calculate value growth metrics
    # 1-year growth (previous season to current)
    prev_season = "2022_2023"  # Example: for current_season="2023-2024"
    
    if f'market_value_{prev_season}' in result_df.columns:
        result_df['value_growth_1yr'] = result_df.apply(
            lambda row: ((row['market_value'] - row[f'market_value_{prev_season}']) / row[f'market_value_{prev_season}'] * 100) 
            if pd.notnull(row['market_value']) and pd.notnull(row[f'market_value_{prev_season}']) and row[f'market_value_{prev_season}'] > 0 
            else 0,
            axis=1
        )
    
    # 3-year growth
    season_3yr_ago = "2020_2021"  # Example: for current_season="2023-2024"
    
    if f'market_value_{season_3yr_ago}' in result_df.columns:
        result_df['value_growth_3yr'] = result_df.apply(
            lambda row: ((row['market_value'] - row[f'market_value_{season_3yr_ago}']) / row[f'market_value_{season_3yr_ago}'] * 100)
            if pd.notnull(row['market_value']) and pd.notnull(row[f'market_value_{season_3yr_ago}']) and row[f'market_value_{season_3yr_ago}'] > 0
            else 0,
            axis=1
        )
    
    conn.close()
    return result_df
"""
    
    # Add the function to database.py
    with open(db_file_path, 'a') as f:
        f.write(function_code)
    
    logger.info("Added get_players_with_market_value_history function to database.py")
    return True

def train_simple_market_value_model():
    """Train a simple market value prediction model based on player stats"""
    logger.info("Training market value prediction model...")
    
    # Get player data with stats (from the most recent season)
    season = "2023-2024"
    players_df = db.get_players_with_stats(season)
    
    if players_df.empty:
        logger.error(f"No player data found for season {season}")
        return None
    
    logger.info(f"Got data for {len(players_df)} players from {season} season")
    
    # Check for players with market values that aren't NULL
    non_null_market_values = players_df[players_df['market_value'].notnull()]
    
    if len(non_null_market_values) < 5:
        logger.warning("Not enough players with non-NULL market values for training")
        logger.info("Using simulated market values based on player stats...")
        
        # Generate synthetic market values based on player stats
        # This is a fallback for initial setup when no real values exist
        players_df['synthetic_value'] = players_df.apply(
            lambda row: (
                (row['goals'] * 2000000 if 'goals' in row else 0) +
                (row['assists'] * 1500000 if 'assists' in row else 0) +
                (row['minutes_played'] * 10000 if 'minutes_played' in row else 0) +
                (row['games_played'] * 500000 if 'games_played' in row else 0) +
                (row['xg'] * 1000000 if 'xg' in row else 0) +
                (row['xa'] * 800000 if 'xa' in row else 0) +
                (max(0, 35 - row['age']) * 1000000 if 'age' in row else 0)
            ),
            axis=1
        )
        
        # Use synthetic values for training
        players_df['target_value'] = players_df['synthetic_value']
    else:
        logger.info(f"Using {len(non_null_market_values)} players with real market values for training")
        # Use real market values
        players_df['target_value'] = players_df['market_value']
    
    # Select features for the model
    features = [
        'age', 'goals', 'assists', 'goals_per90', 'assists_per90',
        'xg', 'xa', 'sca', 'gca', 'minutes_played', 'games_played',
        'progressive_carries', 'progressive_passes', 'penalty_box_touches',
        'pass_completion_pct', 'tackles', 'interceptions'
    ]
    
    # Make sure all selected features exist in the dataframe
    available_features = [f for f in features if f in players_df.columns]
    
    if len(available_features) < 5:
        logger.error("Not enough features available for training")
        return None
    
    logger.info(f"Using {len(available_features)} features for model training")
    
    # Handle missing values
    for feature in available_features:
        if players_df[feature].isnull().any():
            players_df[feature] = players_df[feature].fillna(players_df[feature].median())
    
    # Prepare data for training
    X = players_df[available_features]
    y = players_df['target_value']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        "train_mae": mean_absolute_error(y_train, y_pred_train),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "train_r2": r2_score(y_train, y_pred_train),
        "test_mae": mean_absolute_error(y_test, y_pred_test) if len(y_test) > 0 else 0,
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)) if len(y_test) > 0 else 0,
        "test_r2": r2_score(y_test, y_pred_test) if len(y_test) > 0 else 0,
        "feature_importances": dict(zip(available_features, model.feature_importances_))
    }
    
    logger.info(f"Model trained. Test R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.2f}")
    
    # Save metrics
    metrics_dir = os.path.join(os.path.dirname(__file__), 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_file = os.path.join(metrics_dir, 
                               f"market_value_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_metrics.json")
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Return the trained model and scaler
    return {
        "model": model,
        "scaler": scaler,
        "features": available_features,
        "metrics": metrics
    }

def predict_market_values(model_data):
    """Predict market values for all players with NULL values"""
    logger.info("Predicting market values for players with NULL values...")
    
    if not model_data or "model" not in model_data:
        logger.error("No valid model provided for predictions")
        return False
    
    model = model_data["model"]
    scaler = model_data["scaler"]
    features = model_data["features"]
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get seasons from player_market_values table
    cursor.execute("SELECT DISTINCT season FROM player_market_values ORDER BY season")
    seasons = [row[0] for row in cursor.fetchall()]
    
    if not seasons:
        logger.error("No seasons found in player_market_values table")
        conn.close()
        return False
    
    logger.info(f"Found {len(seasons)} seasons: {seasons}")
    
    # Process each season
    predictions_count = 0
    
    for season in seasons:
        logger.info(f"Processing market values for season {season}...")
        
        # Get player data for this season
        players_df = db.get_players_with_stats(season)
        
        if players_df.empty:
            logger.warning(f"No player data found for season {season}")
            continue
        
        # Check which players have NULL market values in player_market_values table
        cursor.execute("""
        SELECT player_id FROM player_market_values 
        WHERE season = ? AND market_value IS NULL
        """, (season,))
        
        players_with_null_values = [row[0] for row in cursor.fetchall()]
        
        if not players_with_null_values:
            logger.info(f"No NULL market values found for season {season}")
            continue
        
        logger.info(f"Found {len(players_with_null_values)} players with NULL market values for season {season}")
        
        # Filter to players with NULL values
        players_to_predict = players_df[players_df['id'].isin(players_with_null_values)]
        
        if players_to_predict.empty:
            logger.warning(f"No matching players found for prediction in season {season}")
            continue
        
        logger.info(f"Predicting market values for {len(players_to_predict)} players in season {season}")
        
        # Make sure all features exist
        available_features = [f for f in features if f in players_to_predict.columns]
        
        if len(available_features) < len(features) * 0.7:  # At least 70% of features should be available
            logger.warning(f"Too few features available for season {season}. Using available player data directly.")
            
            # For early seasons with missing features, we'll create simpler synthetic values
            players_to_predict['predicted_value'] = players_to_predict.apply(
                lambda row: (
                    (row['goals'] * 2000000 if 'goals' in row and pd.notnull(row['goals']) else 0) +
                    (row['assists'] * 1500000 if 'assists' in row and pd.notnull(row['assists']) else 0) +
                    (row['minutes_played'] * 10000 if 'minutes_played' in row and pd.notnull(row['minutes_played']) else 0) +
                    (row['xg'] * 1000000 if 'xg' in row and pd.notnull(row['xg']) else 0) +
                    (max(0, 35 - row['age']) * 1000000 if 'age' in row and pd.notnull(row['age']) else 20000000)
                ),
                axis=1
            )
        else:
            # Handle missing values
            for feature in available_features:
                if players_to_predict[feature].isnull().any():
                    players_to_predict[feature] = players_to_predict[feature].fillna(players_to_predict[feature].median())
            
            # Prepare data for prediction
            X = players_to_predict[available_features]
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            
            # Add predictions to dataframe
            players_to_predict['predicted_value'] = predictions
        
        # Add position-specific and age adjustments
        players_to_predict['adjusted_value'] = players_to_predict.apply(
            lambda row: adjust_value_by_position_and_age(
                row['predicted_value'], 
                row['position'] if 'position' in row else 'CM',
                row['age'] if 'age' in row else 25
            ),
            axis=1
        )
        
        # Add season-specific adjustments
        season_year = int(season.split('-')[0])
        current_year = datetime.now().year
        years_diff = current_year - season_year
        
        # Apply inflation adjustment for older seasons (roughly 5-10% per year)
        if years_diff > 0:
            inflation_factor = 1 + (years_diff * 0.08)  # 8% annual inflation for past seasons
            players_to_predict['adjusted_value'] = players_to_predict['adjusted_value'] / inflation_factor
        
        # Update database with predicted values
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source = "ml_prediction"
        
        for _, row in players_to_predict.iterrows():
            market_value = int(max(1000000, row['adjusted_value']))  # Ensure minimum value
            
            cursor.execute("""
            UPDATE player_market_values
            SET market_value = ?, value_date = ?, source = ?
            WHERE player_id = ? AND season = ? AND market_value IS NULL
            """, (market_value, current_date, source, row['id'], season))
            
            predictions_count += 1
        
        conn.commit()
        logger.info(f"Updated {len(players_to_predict)} market values for season {season}")
    
    # Update player table with latest values from 2023-2024 season
    cursor.execute("""
    UPDATE players
    SET market_value = (
        SELECT mv.market_value
        FROM player_market_values mv
        WHERE mv.player_id = players.id
        AND mv.season = '2023-2024'
        ORDER BY mv.value_date DESC
        LIMIT 1
    ),
    last_updated = ?
    WHERE market_value IS NULL OR market_value = 0
    """, (current_date,))
    
    updated_players = cursor.rowcount
    conn.commit()
    
    logger.info(f"Updated {updated_players} players with latest market values")
    
    conn.close()
    return predictions_count > 0

def adjust_value_by_position_and_age(base_value, position, age):
    """Apply position and age-specific adjustments to market values"""
    # Position multipliers (forwards typically cost more than defenders)
    position_multipliers = {
        'GK': 0.7,    # Goalkeepers typically cheaper
        'CB': 0.85,   # Central defenders
        'LB': 0.8,    # Left backs
        'RB': 0.8,    # Right backs
        'CDM': 0.9,   # Defensive midfielders
        'CM': 1.0,    # Central midfielders (baseline)
        'CAM': 1.1,   # Attacking midfielders
        'LW': 1.15,   # Left wingers
        'RW': 1.15,   # Right wingers
        'CF': 1.2     # Forwards/strikers most expensive
    }
    
    # Age multipliers (peak value around age 27)
    def age_multiplier(player_age):
        if player_age <= 20:
            return 0.8  # Young players with potential
        elif player_age <= 23:
            return 1.0  # Rising stars
        elif player_age <= 27:
            return 1.2  # Peak years approaching
        elif player_age <= 30:
            return 1.0  # Still in prime
        elif player_age <= 33:
            return 0.7  # Declining but valuable
        else:
            return 0.4  # Veterans
    
    # Get position multiplier (default to CM if not found)
    pos_multiplier = position_multipliers.get(position, 1.0)
    
    # Apply adjustments
    adjusted_value = base_value * pos_multiplier * age_multiplier(age)
    
    return adjusted_value

def run_market_value_update():
    """Execute the full market value update process"""
    try:
        # Step 1: Add missing database function
        if not add_missing_database_function():
            logger.error("Failed to add required database function")
            return False
        
        # Step 2: Train market value prediction model
        model_data = train_simple_market_value_model()
        if model_data is None:
            logger.error("Failed to train market value model")
            return False
        
        # Step 3: Predict and update market values
        if not predict_market_values(model_data):
            logger.error("Failed to update market values")
            return False
        
        logger.info("Market value update completed successfully")
        return True
        
    except Exception as e:
        logger.exception(f"Error during market value update: {str(e)}")
        return False

if __name__ == "__main__":
    run_market_value_update()