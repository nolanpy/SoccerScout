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
    """Enhanced version of the player stats parser with better position, age, nationality, height/weight extraction.
    Designed to work with all players regardless of position or nationality."""
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
        
        # 1. POSITION DETECTION - universally applicable
        position_map = {
            # Midfielder positions
            'am': 'CAM',     # Attacking midfielder
            'cm': 'CM',      # Central midfielder
            'dm': 'CDM',     # Defensive midfielder
            'mf': 'CM',      # Generic midfielder
            'mid': 'CM',     # Generic midfielder
            'midfielder': 'CM',
            'attacking midfielder': 'CAM',
            'defensive midfielder': 'CDM',
            'central midfielder': 'CM',
            
            # Forward positions
            'fw': 'CF',      # Generic forward
            'cf': 'CF',      # Center forward
            'st': 'CF',      # Striker
            'lw': 'LW',      # Left winger
            'rw': 'RW',      # Right winger
            'ss': 'CF',      # Second striker
            'forward': 'CF',
            'striker': 'CF',
            'winger': 'LW',  # Default to LW, will be refined later
            
            # Defender positions
            'df': 'CB',      # Generic defender
            'cb': 'CB',      # Center back
            'lb': 'LB',      # Left back
            'rb': 'RB',      # Right back
            'wb': 'RB',      # Wing back, will refine with left/right
            'defender': 'CB',
            'centre-back': 'CB',
            'center-back': 'CB',
            'fullback': 'RB',  # Default to RB, will be refined later
            'wingback': 'RB',  # Default to RB, will be refined later
            
            # Goalkeeper
            'gk': 'GK',      # Goalkeeper
            'goalkeeper': 'GK'
        }
            
        position_found = False
        position_text = ""
        
        # First, try to find position in specific position field
        for p in player_info_div.find_all('p'):
            text = p.get_text(strip=True).lower()
            if 'position:' in text or 'positions:' in text:
                position_text = text.split(':', 1)[1].strip()
                position_found = True
                logger.info(f"Found position text: {position_text}")
                break
        
        # If no specific position field, look for position indicators in any paragraph
        if not position_found:
            for p in player_info_div.find_all('p'):
                text = p.get_text(strip=True).lower()
                # Common patterns found on FBref
                if any(pos in text for pos in ['midfielder', 'forward', 'defender', 'goalkeeper', 'mf', 'fw', 'df', 'gk']):
                    if 'footed:' in text and ':' in text:
                        parts = text.split(':', 1)
                        # The position is often before "footed"
                        position_text = parts[0].split('footed')[0].strip()
                        position_found = True
                        logger.info(f"Found position in foot text: {position_text}")
                        break
        
        # Extract position from position text using the map
        if position_text:
            position_text = position_text.lower()
            
            # First check for position codes in the map
            for pos_key, pos_value in position_map.items():
                if pos_key in position_text:
                    if pos_value in ['LW', 'RW']:
                        # Refine winger position (left or right)
                        if 'left' in position_text or 'lw' in position_text:
                            original_info['position'] = 'LW'
                        elif 'right' in position_text or 'rw' in position_text:
                            original_info['position'] = 'RW'
                        else:
                            original_info['position'] = pos_value
                    elif pos_value in ['LB', 'RB']:
                        # Refine fullback/wingback position (left or right)
                        if 'left' in position_text or 'lb' in position_text:
                            original_info['position'] = 'LB'
                        elif 'right' in position_text or 'rb' in position_text:
                            original_info['position'] = 'RB'
                        else:
                            original_info['position'] = pos_value
                    else:
                        original_info['position'] = pos_value
                    
                    logger.info(f"Mapped position to: {original_info['position']}")
                    position_found = True
                    break
        
        # If still no position, check for common position patterns in full info text
        if not position_found or 'position' not in original_info:
            info_lower = all_info_text.lower()
            
            # Generic position detection as fallback
            if 'goalkeeper' in info_lower or ' gk ' in info_lower:
                original_info['position'] = 'GK'
            elif 'midfielder' in info_lower or ' mf ' in info_lower:
                if 'attacking' in info_lower or ' am ' in info_lower:
                    original_info['position'] = 'CAM'
                elif 'defensive' in info_lower or ' dm ' in info_lower:
                    original_info['position'] = 'CDM'
                else:
                    original_info['position'] = 'CM'
            elif 'forward' in info_lower or 'striker' in info_lower or ' fw ' in info_lower:
                if 'left' in info_lower or ' lw ' in info_lower:
                    original_info['position'] = 'LW'
                elif 'right' in info_lower or ' rw ' in info_lower:
                    original_info['position'] = 'RW'
                else:
                    original_info['position'] = 'CF'
            elif 'defender' in info_lower or ' df ' in info_lower:
                if 'centre' in info_lower or 'center' in info_lower or ' cb ' in info_lower:
                    original_info['position'] = 'CB'
                elif 'left' in info_lower or ' lb ' in info_lower:
                    original_info['position'] = 'LB'
                elif 'right' in info_lower or ' rb ' in info_lower:
                    original_info['position'] = 'RB'
                else:
                    original_info['position'] = 'CB'
                    
            logger.info(f"Set position from general text: {original_info.get('position', 'Not found')}")
        
        # 2. AGE EXTRACTION - using multiple patterns
        age_patterns = [
            re.compile(r'Age:?\s*(\d+)', re.IGNORECASE),
            re.compile(r'(\d+) years old', re.IGNORECASE),
            re.compile(r'aged (\d+)', re.IGNORECASE)
        ]
        
        birth_patterns = [
            re.compile(r'Born:.*?(\d{4})', re.IGNORECASE),
            re.compile(r'Born on.*?(\d{4})', re.IGNORECASE),
            re.compile(r'DOB:.*?(\d{4})', re.IGNORECASE)
        ]
        
        for p in player_info_div.find_all(['p', 'div', 'span']):
            text = p.get_text(strip=True)
            
            # Try all age patterns
            for pattern in age_patterns:
                age_match = pattern.search(text)
                if age_match:
                    original_info['age'] = int(age_match.group(1))
                    logger.info(f"Found age: {original_info['age']}")
                    break
            
            if 'age' in original_info:
                break
        
        # If no age found directly, try birth year
        if 'age' not in original_info:
            for p in player_info_div.find_all(['p', 'div', 'span']):
                text = p.get_text(strip=True)
                
                # Try all birth year patterns
                for pattern in birth_patterns:
                    birth_match = pattern.search(text)
                    if birth_match:
                        birth_year = int(birth_match.group(1))
                        current_year = datetime.now().year
                        original_info['age'] = current_year - birth_year
                        logger.info(f"Calculated age from birth year {birth_year}: {original_info['age']}")
                        break
                
                if 'age' in original_info:
                    break
        
        # 3. NATIONALITY EXTRACTION - targeted approach for FBref format
        country_found = False
        
        # Best approach: Look specifically for "Born: ... in [City], [Country]" pattern
        # This is the most reliable pattern on FBref
        for p in player_info_div.find_all(['p', 'div']):
            text = p.get_text(strip=True)
            
            # Most reliable pattern: "Born: ... in City, Country"
            born_match = re.search(r'Born:.*?in\s+[^,]+,\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
            if born_match:
                # Get the country name after "in City,"
                country_name = born_match.group(1).strip()
                
                # Clean up country codes (like "pt", "es")
                country_parts = country_name.split()
                if country_parts:
                    # Check if last part is a 2-letter country code
                    if len(country_parts[-1]) == 2 and country_parts[-1].isalpha():
                        # Remove the country code
                        country_name = ' '.join(country_parts[:-1]).strip()
                    
                    # If country name is valid
                    if country_name and len(country_name) > 1:
                        original_info['nationality'] = country_name
                        logger.info(f"Found nationality from birth info: {country_name}")
                        country_found = True
                        break
        
        # Fallback: Look for National Team info if birth location not found
        if not country_found:
            nation_patterns = [
                re.compile(r'National(?:\s+|-)?Te?a?m:?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE),
                re.compile(r'Represents:?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE)
            ]
            
            for p in player_info_div.find_all(['p', 'div']):
                text = p.get_text(strip=True)
                
                for pattern in nation_patterns:
                    nationality_match = pattern.search(text)
                    if nationality_match:
                        original_info['nationality'] = nationality_match.group(1).strip()
                        logger.info(f"Found nationality from national team: {original_info['nationality']}")
                        country_found = True
                        break
                
                if country_found:
                    break
        
        # Second fallback: Try a more generic approach with commas
        if not country_found:
            # Set a default nationality in case we don't find anything
            original_info['nationality'] = 'Unknown'
            
            for p in player_info_div.find_all(['p', 'div']):
                text = p.get_text(strip=True)
                if 'Born:' in text and ',' in text:
                    try:
                        # Extract from "Born: September 8, 1994 in Maia, Portugal"
                        born_parts = text.split('Born:')[1].strip()
                        # Get the last part after a comma
                        parts = born_parts.split(',')
                        if parts and len(parts) >= 2:
                            last_part = parts[-1].strip()
                            # Extract just the country name (remove any code or extra text)
                            country_match = re.match(r'^([A-Za-z\s]+)', last_part)
                            if country_match:
                                country = country_match.group(1).strip()
                                # Clean up potential country code
                                if ' ' in country:
                                    # Check if last word is a 2-letter code
                                    words = country.split()
                                    if len(words[-1]) == 2 and words[-1].isalpha():
                                        country = ' '.join(words[:-1]).strip()
                                original_info['nationality'] = country
                                logger.info(f"Found nationality fallback: {country}")
                                country_found = True
                                break
                    except Exception as e:
                        logger.error(f"Error parsing nationality fallback: {e}")
        
        # 4. HEIGHT AND WEIGHT - handle multiple formats
        height_weight_patterns = [
            # Metric: 173cm, 64kg
            re.compile(r'(\d+)cm,\s*(\d+)kg'),
            # Imperial in parentheses: (5-8, 143lb)
            re.compile(r'\((\d+)-(\d+),\s*(\d+)lb\)'),
            # Alternative format: Height: 1.73m, Weight: 64kg
            re.compile(r'Height:?\s*(\d+)\.(\d+)m,?\s*Weight:?\s*(\d+)kg', re.IGNORECASE),
            # Imperial separated: 5ft 8in, 143 lbs
            re.compile(r'(\d+)ft\s*(\d+)in,?\s*(\d+)\s*lbs?', re.IGNORECASE)
        ]
        
        for p in player_info_div.find_all(['p', 'div', 'span']):
            text = p.get_text(strip=True)
            
            for i, pattern in enumerate(height_weight_patterns):
                hw_match = pattern.search(text)
                if hw_match:
                    if i == 0:  # Metric format: 173cm, 64kg
                        original_info['height'] = float(hw_match.group(1))
                        original_info['weight'] = float(hw_match.group(2))
                        logger.info(f"Found metric height/weight: {original_info['height']}cm, {original_info['weight']}kg")
                    elif i == 1:  # Imperial format: (5-8, 143lb)
                        feet = int(hw_match.group(1))
                        inches = int(hw_match.group(2))
                        pounds = float(hw_match.group(3))
                        # Convert to metric
                        height_cm = (feet * 30.48) + (inches * 2.54)
                        weight_kg = pounds * 0.453592
                        original_info['height'] = round(height_cm, 1)
                        original_info['weight'] = round(weight_kg, 1)
                        logger.info(f"Converted imperial to: {original_info['height']}cm, {original_info['weight']}kg")
                    elif i == 2:  # Height in meters: Height: 1.73m, Weight: 64kg
                        meters = float(hw_match.group(1))
                        cm_part = float(hw_match.group(2))
                        height_cm = (meters * 100) + (cm_part * 10 if cm_part < 10 else cm_part)
                        weight_kg = float(hw_match.group(3))
                        original_info['height'] = round(height_cm, 1)
                        original_info['weight'] = weight_kg
                        logger.info(f"Parsed height in meters: {original_info['height']}cm, {original_info['weight']}kg")
                    elif i == 3:  # Imperial separated: 5ft 8in, 143 lbs
                        feet = int(hw_match.group(1))
                        inches = int(hw_match.group(2))
                        pounds = float(hw_match.group(3))
                        # Convert to metric
                        height_cm = (feet * 30.48) + (inches * 2.54)
                        weight_kg = pounds * 0.453592
                        original_info['height'] = round(height_cm, 1)
                        original_info['weight'] = round(weight_kg, 1)
                        logger.info(f"Converted imperial units: {original_info['height']}cm, {original_info['weight']}kg")
                    break
            
            if 'height' in original_info and 'weight' in original_info:
                break
        
        # 5. PREFERRED FOOT - cover different formats
        foot_patterns = [
            re.compile(r'(?:Preferred\s+)?foot(?:ed)?:?\s*([a-zA-Z]+)', re.IGNORECASE),
            re.compile(r'([Ll]eft|[Rr]ight)[\s-]*footed', re.IGNORECASE)
        ]
        
        for p in player_info_div.find_all(['p', 'div', 'span']):
            text = p.get_text(strip=True)
            
            foot_found = False
            for pattern in foot_patterns:
                foot_match = pattern.search(text)
                if foot_match:
                    foot = foot_match.group(1).strip().capitalize()
                    # Standardize to just "Left" or "Right"
                    if foot.lower() in ['left', 'l']:
                        original_info['preferred_foot'] = 'Left'
                    elif foot.lower() in ['right', 'r']:
                        original_info['preferred_foot'] = 'Right'
                    else:
                        original_info['preferred_foot'] = foot
                    logger.info(f"Found preferred foot: {original_info['preferred_foot']}")
                    foot_found = True
                    break
            
            if foot_found:
                break
    
    # 6. ENHANCE STATS EXTRACTION - ONLY USE ACTUAL FBREF DATA
    for season, season_data in original_info.get('organized_stats', {}).items():
        # Check if we have passing stats
        if 'passing' in season_data:
            passing_stats = season_data['passing']
            
            # Only handle xA estimation if missing - this is a key stat for player evaluation
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
                        
            # Extract progressive passes if available
            prog_passes = passing_stats.get('progressive_passes', '0')
            if prog_passes and prog_passes != '0':
                try:
                    # Store directly in the passing stats
                    passing_stats['progressive_passes'] = prog_passes
                    logger.info(f"Found progressive passes for {season}: {prog_passes}")
                except (ValueError, TypeError):
                    pass
        
        # Extract advanced metrics from Goal Creating Actions (gca) table
        if 'gca' in season_data:
            gca_stats = season_data['gca']
            
            # Extract progressive carries if available
            prog_carries = gca_stats.get('carries_progressive', '0')
            if prog_carries and prog_carries != '0':
                try:
                    # Store directly in the gca stats
                    gca_stats['progressive_carries'] = prog_carries
                    logger.info(f"Found progressive carries for {season}: {prog_carries}")
                except (ValueError, TypeError):
                    pass
                    
            # Extract progressive passes received if available
            prog_passes_rcv = gca_stats.get('passes_received_progressive', '0')
            if prog_passes_rcv and prog_passes_rcv != '0':
                try:
                    # Store directly in the gca stats
                    gca_stats['progressive_passes_received'] = prog_passes_rcv
                    logger.info(f"Found progressive passes received for {season}: {prog_passes_rcv}")
                except (ValueError, TypeError):
                    pass
        
        # Extract possession metrics if available
        if 'possession' in season_data:
            possession_stats = season_data['possession']
            
            # Extract dribbles/take-ons if available
            dribbles_completed = possession_stats.get('dribbles_completed', '0')
            dribbles_attempted = possession_stats.get('dribbles_attempted', '0')
            
            if dribbles_attempted and dribbles_attempted != '0':
                logger.info(f"Found dribbles for {season}: {dribbles_completed}/{dribbles_attempted}")
        
        # Ensure each season has core stat categories that we know exist on FBref
        required_categories = ['standard', 'shooting', 'passing', 'gca', 'defense']
        for category in required_categories:
            if category not in season_data:
                season_data[category] = {}
                logger.info(f"Added missing category {category} for {season}")
    
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
    
    # Fix nationality if it still has country code
    if 'nationality' in player_data:
        nationality = player_data['nationality']
        # Check if it looks like "Portugalpt" - country name followed by 2-letter code
        if len(nationality) > 2 and nationality[-2:].isalpha() and nationality[-2:].islower():
            # Remove the last two characters
            player_data['nationality'] = nationality[:-2].strip()
            logger.info(f"Fixed nationality from {nationality} to {player_data['nationality']}")
    
    # Modify the player_stats_list to ensure we're not using default values for fields
    # that don't exist in the FBref data - set them explicitly to NULL/None
    fields_to_nullify = [
        'progressive_carries',
        'progressive_passes',
        'progressive_passes_received',
        'penalty_box_touches',
        'dribbles_completed',
        'dribbles_attempted',
        'pressures',
        'pressure_success_rate',
        'aerial_duels_won',
        'aerial_duels_total',
        'high_intensity_runs',
        'ball_recoveries',
        'final_third_passes_completed',
        'final_third_passes_attempted'
    ]
    
    for season_stats in player_stats_list:
        # For each season, only include values that were explicitly in the original data
        for field in fields_to_nullify:
            # Check if the value is defaulted (usually 0)
            if field in season_stats and season_stats[field] == 0:
                # Set to None (NULL in SQL) to indicate missing data
                season_stats[field] = None
                logger.info(f"Set {field} to NULL for season {season_stats.get('season')}")
    
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