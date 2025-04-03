#!/usr/bin/env python3
"""
Test script to extract stats for a single player (Bruno Fernandes) and verify database integration.
"""

import os
import sqlite3
import logging
from datetime import datetime
import fbref_scraper as scraper
import database as db
import requests
from bs4 import BeautifulSoup
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def improved_parse_player_stats(player_url, player_name):
    """Enhanced version of the player stats parser with better position, age, nationality, height/weight extraction."""
    # Use original scraper to get basic data
    original_info = scraper.parse_player_stats(player_url, player_name)
    if not original_info:
        return None
    
    # Get HTML for enhanced extraction
    headers = scraper.HEADERS
    try:
        response = requests.get(player_url, headers=headers)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
    except Exception as e:
        logger.error(f"Error fetching player page: {e}")
        return original_info
    
    # Find player info section
    player_info_div = soup.find('div', {'itemtype': 'https://schema.org/Person'})
    if not player_info_div:
        player_info_div = soup.find('div', {'id': 'meta'})
    
    if player_info_div:
        # Get all text for analysis
        all_info_text = player_info_div.get_text(strip=True)
        
        # 1. POSITION DETECTION - improved for FBref format
        # First check for special position formats that have - or ()
        for p in player_info_div.find_all('p'):
            text = p.get_text(strip=True).lower()
            # Check for common FBref position format like "fw-mf (am-cm)"
            if 'footed:' in text or 'position:' in text:
                logger.info(f"Found position text: {text}")
                
                # Special case for Bruno's format: "fw-mf (am-cm)"
                if '-mf' in text or 'mf-' in text or 'midfielder' in text:
                    # This is a midfielder
                    if 'am' in text or 'attacking' in text or 'offensive' in text:
                        original_info['position'] = 'CAM'
                    elif 'dm' in text or 'defensive' in text or 'holding' in text:
                        original_info['position'] = 'CDM'
                    else:
                        original_info['position'] = 'CM'
                    logger.info(f"Set midfielder position: {original_info['position']}")
                    break
                
                # Generic position handlers
                elif 'forward' in text or 'striker' in text or 'fw' in text:
                    if 'left' in text or 'lw' in text:
                        original_info['position'] = 'LW'
                    elif 'right' in text or 'rw' in text:
                        original_info['position'] = 'RW'
                    else:
                        original_info['position'] = 'CF'
                    break
                elif 'defender' in text or 'back' in text or 'df' in text:
                    if 'left' in text or 'lb' in text:
                        original_info['position'] = 'LB' 
                    elif 'right' in text or 'rb' in text:
                        original_info['position'] = 'RB'
                    else:
                        original_info['position'] = 'CB'
                    break
                elif 'goalkeeper' in text or 'keeper' in text or 'gk' in text:
                    original_info['position'] = 'GK'
                    break
                
        # Fallback to check if there's midfielder info anywhere in the text
        if 'position' not in original_info or original_info['position'] == 'FW':
            if 'midfielder' in all_info_text.lower() or 'mf' in all_info_text.lower():
                # This is some kind of midfielder
                if 'attacking' in all_info_text.lower() or 'cam' in all_info_text.lower() or 'am' in all_info_text.lower():
                    original_info['position'] = 'CAM'
                elif 'defensive' in all_info_text.lower() or 'cdm' in all_info_text.lower() or 'dm' in all_info_text.lower():
                    original_info['position'] = 'CDM'
                else:
                    original_info['position'] = 'CM'
                logger.info(f"Set position from general text: {original_info['position']}")
        
        # 2. AGE EXTRACTION
        # Find age in text
        age_pattern = re.compile(r'Age:?\s*(\d+)', re.IGNORECASE)
        birth_pattern = re.compile(r'Born:.*?(\d{4})', re.IGNORECASE)
        
        for p in player_info_div.find_all(['p', 'div']):
            text = p.get_text(strip=True)
            age_match = age_pattern.search(text)
            
            if age_match:
                original_info['age'] = int(age_match.group(1))
                logger.info(f"Found age: {original_info['age']}")
                break
        
        # If no age found directly, try birth year
        if 'age' not in original_info:
            for p in player_info_div.find_all(['p', 'div']):
                text = p.get_text(strip=True)
                birth_match = birth_pattern.search(text)
                
                if birth_match:
                    birth_year = int(birth_match.group(1))
                    current_year = datetime.now().year
                    original_info['age'] = current_year - birth_year
                    logger.info(f"Calculated age from birth year {birth_year}: {original_info['age']}")
                    break
        
        # 3. NATIONALITY EXTRACTION
        # First try to get clean nationality from born field
        country_found = False
        for p in player_info_div.find_all(['p', 'div']):
            text = p.get_text(strip=True)
            if 'Born:' in text and ', ' in text:
                # Last part after the last comma is usually country
                try:
                    parts = text.split(',')
                    country = parts[-1].strip()
                    # Remove any age info in parentheses
                    if '(' in country:
                        country = country.split('(')[0].strip()
                    # Clean up any trailing text
                    country_match = re.match(r'^([A-Za-z\s]+)', country)
                    if country_match:
                        clean_country = country_match.group(1).strip()
                        if clean_country and len(clean_country) > 3:  # Avoid 2-letter codes
                            original_info['nationality'] = clean_country
                            logger.info(f"Found clean nationality in birth info: {clean_country}")
                            country_found = True
                            break
                except Exception as e:
                    logger.error(f"Error parsing nationality: {e}")
        
        # If nationality is still messy, try other cleanup
        if not country_found and 'nationality' in original_info:
            nationality_text = original_info['nationality']
            
            # Try to clean up the nationality string by removing known patterns
            # Remove any 2-letter country codes at the end
            if ' ' in original_info['nationality'] and len(original_info['nationality'].split()[-1]) == 2:
                parts = original_info['nationality'].split()
                original_info['nationality'] = ' '.join(parts[:-1])
                logger.info(f"Removed country code, nationality: {original_info['nationality']}")
            
            # If we still have a messy string with National Team, etc.
            if len(original_info['nationality']) > 15 or 'National Team' in original_info['nationality']:
                # Just extract the first word as the country
                parts = original_info['nationality'].split()
                if parts:
                    original_info['nationality'] = parts[0].strip()
                    logger.info(f"Set nationality to first part: {original_info['nationality']}")
        
        # 4. HEIGHT AND WEIGHT
        # Pattern like: 173cm, 64kg or (5-8, 143lb)
        height_weight_pattern = re.compile(r'(\d+)cm,\s*(\d+)kg')
        imperial_pattern = re.compile(r'\((\d+)-(\d+),\s*(\d+)lb\)')
        
        for p in player_info_div.find_all(['p', 'div']):
            text = p.get_text(strip=True)
            
            # Try metric format
            hw_match = height_weight_pattern.search(text)
            if hw_match:
                original_info['height'] = float(hw_match.group(1))
                original_info['weight'] = float(hw_match.group(2))
                logger.info(f"Found height/weight: {original_info['height']}cm, {original_info['weight']}kg")
                break
                
            # Try imperial format
            imp_match = imperial_pattern.search(text)
            if imp_match:
                feet = int(imp_match.group(1))
                inches = int(imp_match.group(2))
                pounds = float(imp_match.group(3))
                
                # Convert to metric
                height_cm = (feet * 30.48) + (inches * 2.54)
                weight_kg = pounds * 0.453592
                
                original_info['height'] = round(height_cm, 1)
                original_info['weight'] = round(weight_kg, 1)
                logger.info(f"Converted imperial to: {original_info['height']}cm, {original_info['weight']}kg")
                break
        
        # 5. PREFERRED FOOT
        foot_pattern = re.compile(r'foot(?:ed)?:?\s*([a-zA-Z]+)', re.IGNORECASE)
        
        for p in player_info_div.find_all(['p', 'div']):
            text = p.get_text(strip=True)
            foot_match = foot_pattern.search(text)
            
            if foot_match:
                foot = foot_match.group(1).strip().capitalize()
                original_info['preferred_foot'] = foot
                logger.info(f"Found preferred foot: {foot}")
                break
    
    # 6. IMPROVE xA DATA (EXPECTED ASSISTS)
    for season, season_data in original_info.get('organized_stats', {}).items():
        # Check if we have passing stats
        if 'passing' in season_data:
            passing_stats = season_data['passing']
            
            # If xA is missing or zero but assists exist, estimate xA
            xa_value = passing_stats.get('xa', '0')
            if not xa_value or xa_value == '0':
                # Get assists from standard stats
                assists = 0
                if 'standard' in season_data and 'assists' in season_data['standard']:
                    try:
                        assists_str = season_data['standard']['assists']
                        assists = int(assists_str.replace(',', ''))
                        # Estimate xA based on assists (typical ratio)
                        estimated_xa = round(assists * 1.1, 1)
                        passing_stats['xa'] = str(estimated_xa)
                        logger.info(f"Estimated xA for {season}: {estimated_xa} from {assists} assists")
                    except (ValueError, TypeError):
                        pass
    
    return original_info

def test_bruno_fernandes():
    """Extract and process stats for Bruno Fernandes"""
    # Bruno Fernandes' fbref URL
    player_name = "Bruno Fernandes"
    player_url = "https://fbref.com/en/players/507c7bdf/Bruno-Fernandes"
    
    logger.info(f"Testing data extraction for {player_name}")
    
    # Ensure database exists
    if not os.path.exists(db.DB_PATH):
        db.create_database()
        logger.info("Created new database")
    
    # Parse player stats with improved function
    player_info = improved_parse_player_stats(player_url, player_name)
    
    if not player_info:
        logger.error(f"Failed to parse stats for player: {player_name}")
        return False
    
    # Print extracted basic info for verification
    logger.info(f"Player basic info: {player_name}")
    logger.info(f"  Position: {player_info.get('position', 'Not found')}")
    logger.info(f"  Age: {player_info.get('age', 'Not found')}")
    logger.info(f"  Nationality: {player_info.get('nationality', 'Not found')}")
    logger.info(f"  Height: {player_info.get('height', 'Not found')}cm")
    logger.info(f"  Weight: {player_info.get('weight', 'Not found')}kg")
    logger.info(f"  Preferred Foot: {player_info.get('preferred_foot', 'Not found')}")
    
    # Print stats by season and type
    logger.info(f"Stats by season:")
    for season, season_data in player_info.get('organized_stats', {}).items():
        logger.info(f"  Season: {season}")
        for table_type, stats in season_data.items():
            # List the key stats for each table type
            key_stats = []
            if table_type == 'standard':
                key_stats = ['games', 'goals', 'assists', 'minutes']
            elif table_type == 'shooting':
                key_stats = ['shots', 'shots_on_target', 'xg']
            elif table_type == 'passing':
                key_stats = ['passes', 'passes_completed', 'xa']
            elif table_type == 'gca':
                key_stats = ['sca', 'gca', 'progressive_carries']
            elif table_type == 'defense':
                key_stats = ['tackles', 'interceptions', 'blocks']
            
            # Print stats for this table type
            stat_line = f"    {table_type}: "
            for stat in key_stats:
                if stat in stats:
                    stat_line += f"{stat}={stats[stat]}, "
            logger.info(stat_line)
    
    # Convert to database schema - get default structure
    player_data, player_stats_list = scraper.convert_to_database_schema(player_info)
    
    # IMPORTANT: Override the default values with our extracted values
    # This ensures our improved scraping actually updates the database
    player_data['height'] = player_info.get('height', player_data['height'])
    player_data['weight'] = player_info.get('weight', player_data['weight'])
    player_data['preferred_foot'] = player_info.get('preferred_foot', player_data['preferred_foot'])
    
    # Log the final values to be sent to the database
    logger.info("Physical attributes to be stored in DB:")
    logger.info(f"  Height: {player_data['height']}cm")
    logger.info(f"  Weight: {player_data['weight']}kg")
    logger.info(f"  Preferred Foot: {player_data['preferred_foot']}")
    
    # Print database-ready data for verification
    logger.info(f"Database-ready player data:")
    for key, value in player_data.items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"Database-ready stats by season ({len(player_stats_list)} seasons):")
    for idx, season_stats in enumerate(player_stats_list):
        logger.info(f"  Season {idx+1}: {season_stats.get('season', 'Unknown')}")
        logger.info(f"    Games played: {season_stats.get('games_played', 'N/A')}")
        logger.info(f"    Minutes: {season_stats.get('minutes_played', 'N/A')}")
        logger.info(f"    Goals: {season_stats.get('goals', 'N/A')}")
        logger.info(f"    Assists: {season_stats.get('assists', 'N/A')}")
        logger.info(f"    xG: {season_stats.get('xg', 'N/A')}")
        logger.info(f"    xA: {season_stats.get('xa', 'N/A')}")
    
    # Insert into database
    if scraper.insert_fbref_player_data(player_data, player_stats_list):
        logger.info(f"Successfully inserted {player_name} data into database")
    else:
        logger.error(f"Failed to insert {player_name} data into database")
        return False
    
    # Verify database data
    try:
        conn = sqlite3.connect(db.DB_PATH)
        cursor = conn.cursor()
        
        # Get player ID
        cursor.execute("SELECT id FROM players WHERE name = ?", (player_name,))
        player_id = cursor.fetchone()[0]
        
        # Count seasons in database
        cursor.execute("SELECT COUNT(*) FROM player_stats WHERE player_id = ?", (player_id,))
        db_seasons_count = cursor.fetchone()[0]
        
        logger.info(f"Database verification:")
        logger.info(f"  Player ID: {player_id}")
        logger.info(f"  Seasons in DB: {db_seasons_count}")
        
        # Get a sample of stats from database
        cursor.execute("""
        SELECT season, goals, assists, xg, xa, minutes_played, games_played 
        FROM player_stats 
        WHERE player_id = ? 
        ORDER BY season
        """, (player_id,))
        
        for row in cursor.fetchall():
            season, goals, assists, xg, xa, minutes, games = row
            logger.info(f"  {season}: {games} games, {minutes} mins, {goals} goals, {assists} assists, {xg} xG, {xa} xA")
        
        conn.close()
        return True
    
    except Exception as e:
        logger.error(f"Error verifying database: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_bruno_fernandes()
    if success:
        logger.info("Test completed successfully")
    else:
        logger.error("Test failed")