"""Database reset and repopulation utility for SoccerScout

This script will:
1. Reset all database tables (players, player_stats, player_market_values)
2. Import real player data from FBref
3. Set up empty market values for ML processing
"""

import os
import sqlite3
import logging
import database as db
import fbref_scraper
from datetime import datetime

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f'database_reset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

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

def reset_database():
    """Drop and recreate all database tables"""
    logger.info("Resetting database...")
    
    # Remove existing database if it exists
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        logger.info(f"Removed existing database at {DB_PATH}")
    
    # Create new database with tables
    db.create_database()
    logger.info("Created new database with empty tables")
    
    return True

def create_market_values_table():
    """Create the player_market_values table if it doesn't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create market values table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS player_market_values (
        id INTEGER PRIMARY KEY,
        player_id INTEGER NOT NULL,
        season TEXT NOT NULL,
        market_value INTEGER,
        value_date TIMESTAMP,
        source TEXT,
        FOREIGN KEY (player_id) REFERENCES players(id)
    )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Created player_market_values table")
    
    return True

def populate_with_fbref_data():
    """Populate database with real player data from FBref"""
    logger.info("Populating database with FBref player data...")
    
    # Scrape Manchester United data
    success_count, fail_count = fbref_scraper.scrape_team_data(
        team_id='19538871',  # Manchester United team ID
        team_name='Manchester United',
        start_season=2019,  # Start from the 2019-2020 season for better data quality
        end_season=2022     # Up to 2022-2023 season
    )
    
    logger.info(f"Imported {success_count} players successfully")
    logger.info(f"Failed to import {fail_count} players")
    
    return success_count > 0

def initialize_market_values():
    """Initialize empty market values for all players in the database"""
    logger.info("Initializing empty market values for all players...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all players
    cursor.execute("SELECT id FROM players")
    players = cursor.fetchall()
    
    if not players:
        logger.warning("No players found in database")
        conn.close()
        return False
    
    # Seasons we want to create market value entries for
    seasons = ["2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"]
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Initialize empty market values for each player and season
    values_added = 0
    
    for player_id in [p[0] for p in players]:
        for season in seasons:
            cursor.execute('''
            INSERT INTO player_market_values (player_id, season, market_value, value_date, source)
            VALUES (?, ?, NULL, ?, ?)
            ''', (player_id, season, current_date, "pending_ml_prediction"))
            values_added += 1
    
    conn.commit()
    conn.close()
    
    logger.info(f"Added {values_added} empty market value entries for {len(players)} players")
    return True

def run_database_reset():
    """Execute the full database reset and repopulation process"""
    try:
        # Step 1: Reset the database
        if not reset_database():
            logger.error("Failed to reset database")
            return False
        
        # Step 2: Create market values table
        if not create_market_values_table():
            logger.error("Failed to create market values table")
            return False
        
        # Step 3: Populate with FBref data
        if not populate_with_fbref_data():
            logger.error("Failed to populate database with FBref data")
            return False
        
        # Step 4: Initialize market values
        if not initialize_market_values():
            logger.error("Failed to initialize market values")
            return False
        
        logger.info("Database reset and repopulation completed successfully")
        return True
        
    except Exception as e:
        logger.exception(f"Error during database reset: {str(e)}")
        return False

if __name__ == "__main__":
    run_database_reset()
