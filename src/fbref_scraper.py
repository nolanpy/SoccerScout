"""
FBref data scraper for SoccerScout application.

This module provides functions to:
1. Scrape historical player data from FBref.com
2. Process and transform the data to match the SoccerScout database schema
3. Import the data into the SoccerScout database
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import random
import sqlite3
import os
import logging
import database as db
import re
from datetime import datetime

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f'fbref_scraping_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
BASE_URL = 'https://fbref.com'

# Team IDs
TEAM_IDS = {
    'manchester_united': '19538871'
}

def get_page_with_retry(url, max_retries=3, base_delay=3):
    """Get a web page with retry logic and random delays to be respectful"""
    for attempt in range(max_retries):
        try:
            # Add a random delay between requests
            delay = base_delay + random.uniform(1, 3)
            logger.info(f"Waiting {delay:.2f} seconds before request")
            time.sleep(delay)
            
            # Make the request
            logger.info(f"Requesting URL: {url}")
            response = requests.get(url, headers=HEADERS)
            
            if response.status_code == 200:
                logger.info(f"Successfully retrieved page: {url}")
                return response.text
            else:
                logger.warning(f"Failed to retrieve page (Status: {response.status_code}): {url}")
                
        except Exception as e:
            logger.error(f"Error retrieving page ({attempt+1}/{max_retries}): {str(e)}")
            
        # Increase delay for next attempt
        base_delay *= 2
    
    logger.error(f"Failed to retrieve page after {max_retries} attempts: {url}")
    return None

def get_season_urls(team_id, start_season=2017, end_season=2023):
    """Get URLs for all seasons for a team"""
    season_urls = []
    
    for year in range(start_season, end_season + 1):
        season = f"{year}-{year+1}"
        url = f"{BASE_URL}/en/squads/{team_id}/{season}/Manchester-United-Stats"
        season_urls.append((season, url))
    
    return season_urls

def get_player_urls_from_roster(roster_url):
    """Extract player URLs from a team's roster page"""
    html = get_page_with_retry(roster_url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    player_urls = []
    
    # Try to find roster table with different possible IDs
    roster_table = soup.find('table', {'id': 'roster'})
    
    # If not found, try alternative IDs
    if not roster_table:
        roster_table = soup.find('table', {'id': 'stats_standard_ks_roster'})
    
    # If still not found, try any table with "roster" in the ID
    if not roster_table:
        for table in soup.find_all('table'):
            table_id = table.get('id', '')
            if 'roster' in table_id.lower():
                roster_table = table
                break
    
    # If still not found, try any table with player stats
    if not roster_table:
        roster_table = soup.find('table', {'class': 'stats_table'})
        
    if not roster_table:
        logger.warning(f"No roster table found at {roster_url}")
        # Try to directly extract player links from the page
        player_links = soup.find_all('a')
        for link in player_links:
            href = link.get('href', '')
            if '/players/' in href and href.endswith('/Manchester-United'):
                player_name = link.get_text(strip=True)
                if player_name and player_name not in [p[0] for p in player_urls]:
                    player_urls.append((player_name, f"{BASE_URL}{href}"))
        
        if player_urls:
            logger.info(f"Found {len(player_urls)} players by direct link extraction")
            return player_urls
        return []
    
    # Extract player links
    player_rows = roster_table.find('tbody').find_all('tr')
    for row in player_rows:
        # Look for any player cell
        player_cell = row.find(['th', 'td'], {'data-stat': 'player'})
        if not player_cell:
            # Try other possible stats that might contain player links
            for cell in row.find_all(['th', 'td']):
                if cell.find('a'):
                    player_cell = cell
                    break
        
        if player_cell and player_cell.find('a'):
            player_link = player_cell.find('a')['href']
            player_name = player_cell.get_text(strip=True)
            player_urls.append((player_name, f"{BASE_URL}{player_link}"))
    
    logger.info(f"Found {len(player_urls)} players on roster page")
    return player_urls

def parse_player_stats(player_url, player_name):
    """Extract and parse player statistics from player page"""
    html = get_page_with_retry(player_url)
    if not html:
        return None
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract player info
    player_info = {
        'name': player_name,
        'url': player_url,
        'stats_by_season': []
    }
    
    # Try to extract nationality, position, and other basic info
    player_info_div = soup.find('div', {'itemtype': 'https://schema.org/Person'})
    if not player_info_div:
        # Try alternative structure
        player_info_div = soup.find('div', {'id': 'meta'})
    
    if player_info_div:
        # Extract position
        position_paragraph = player_info_div.find('p')
        if position_paragraph:
            position_text = position_paragraph.get_text(strip=True)
            position_match = re.search(r'Position:?\s*([A-Za-z, ]+)', position_text)
            if position_match:
                player_info['position'] = position_match.group(1).strip()
                # Map to our schema's position format (take the first position)
                positions = player_info['position'].split(',')[0].strip()
                if 'Forward' in positions:
                    player_info['position'] = 'CF'
                elif 'Midfielder' in positions:
                    player_info['position'] = 'CM'
                elif 'Defender' in positions:
                    player_info['position'] = 'CB'
                elif 'Goalkeeper' in positions:
                    player_info['position'] = 'GK'
        
        # Try alternative position extraction
        if 'position' not in player_info:
            # Look for position in any paragraph
            for paragraph in player_info_div.find_all('p'):
                text = paragraph.get_text(strip=True)
                if 'Position' in text:
                    position_match = re.search(r'Position:?\s*([A-Za-z, ]+)', text)
                    if position_match:
                        player_info['position'] = position_match.group(1).strip()
                        # Map to our schema's position format (take the first position)
                        positions = player_info['position'].split(',')[0].strip()
                        if 'Forward' in positions:
                            player_info['position'] = 'CF'
                        elif 'Midfielder' in positions:
                            player_info['position'] = 'CM'
                        elif 'Defender' in positions:
                            player_info['position'] = 'CB'
                        elif 'Goalkeeper' in positions:
                            player_info['position'] = 'GK'
                    break
        
        # Extract age
        age_strong = player_info_div.find('strong', text=re.compile(r'Age:?'))
        if age_strong:
            age_span = age_strong.parent.find('span')
            if age_span:
                age_match = re.search(r'(\d+)', age_span.get_text(strip=True))
                if age_match:
                    player_info['age'] = int(age_match.group(1))
            else:
                # Try alternative age extraction
                age_text = age_strong.parent.get_text(strip=True)
                age_match = re.search(r'Age:?\s*(\d+)', age_text)
                if age_match:
                    player_info['age'] = int(age_match.group(1))
        
        # Try alternative age extraction
        if 'age' not in player_info:
            for paragraph in player_info_div.find_all('p'):
                text = paragraph.get_text(strip=True)
                if 'Age' in text:
                    age_match = re.search(r'Age:?\s*(\d+)', text)
                    if age_match:
                        player_info['age'] = int(age_match.group(1))
                    break
        
        # Extract nationality
        birthplace_element = player_info_div.find('strong', text=re.compile(r'Birthplace:?|Born:?|Country:?'))
        if birthplace_element:
            birthplace_text = birthplace_element.parent.get_text(strip=True)
            # Extract the last comma-separated item (typically the country)
            birthplace_parts = birthplace_text.split(',')
            if birthplace_parts:
                nationality = birthplace_parts[-1].strip()
                # Remove any text before a colon (like "Birthplace: ")
                if ':' in nationality:
                    nationality = nationality.split(':')[1].strip()
                player_info['nationality'] = nationality
    
    # Try to determine the player's position based on stats if we couldn't extract it
    if 'position' not in player_info:
        # Default to CM if we can't determine
        player_info['position'] = 'CM'
    
    # Find all stats tables on the page
    all_tables = soup.find_all('table')
    relevant_tables = []
    
    for table in all_tables:
        table_id = table.get('id', '')
        table_class = table.get('class', [])
        
        # Check if it's a valid stats table (either by ID or class)
        is_valid = (
            ('stats_table' in table_class) or
            (table_id and (
                'stats_standard' in table_id or
                'stats_shooting' in table_id or
                'stats_passing' in table_id or
                'stats_gca' in table_id or
                'stats_defense' in table_id or
                'all_comps' in table_id or
                'keeper' in table_id
            ))
        )
        
        if is_valid:
            relevant_tables.append(table)
    
    # Log the number of stats tables found
    logger.info(f"Found {len(relevant_tables)} relevant stats tables for {player_name}")
    
    for table in relevant_tables:
        table_id = table.get('id', '')
        
        # Determine table type based on ID
        table_type = ''
        if 'standard' in table_id:
            table_type = 'standard'
        elif 'shooting' in table_id:
            table_type = 'shooting'
        elif 'passing' in table_id:
            table_type = 'passing'
        elif 'gca' in table_id:
            table_type = 'gca'
        elif 'defense' in table_id:
            table_type = 'defense'
        elif 'keeper' in table_id:
            table_type = 'keeper'
        else:
            # Try to determine type from table caption or other elements
            caption = table.find('caption')
            if caption:
                caption_text = caption.get_text(strip=True).lower()
                if 'standard' in caption_text or 'summary' in caption_text:
                    table_type = 'standard'
                elif 'shooting' in caption_text:
                    table_type = 'shooting'
                elif 'passing' in caption_text:
                    table_type = 'passing'
                elif 'creation' in caption_text:
                    table_type = 'gca'
                elif 'defense' in caption_text:
                    table_type = 'defense'
                elif 'keeper' in caption_text or 'goalkeeping' in caption_text:
                    table_type = 'keeper'
        
        # Extract column headers
        headers = []
        header_rows = table.find('thead').find_all('tr')
        
        # Try the last row first, as it usually has the actual column names
        header_row = header_rows[-1] if header_rows else None
        
        if not header_row:
            continue
            
        for header in header_row.find_all(['th', 'td']):
            # Use data-stat attribute if available, otherwise use text
            stat = header.get('data-stat', '')
            if not stat:
                stat = header.get_text(strip=True).lower().replace(' ', '_')
            headers.append(stat)
        
        # Process each row (season)
        if not table.find('tbody'):
            continue
            
        rows = table.find('tbody').find_all('tr')
        for row in rows:
            # Skip non-data rows
            if 'class' in row.attrs and ('thead' in row['class'] or 'divider' in row['class']):
                continue
            
            # Extract season (might be in different locations)
            season_cell = row.find(['th', 'td'], {'data-stat': 'season'})
            if not season_cell:
                # Try other common season identifiers
                season_cell = row.find(['th', 'td'], {'data-stat': 'year_id'})
                
            if not season_cell:
                # Try to find any cell that looks like a season (e.g., "2019-2020")
                for cell in row.find_all(['th', 'td']):
                    text = cell.get_text(strip=True)
                    if re.match(r'\d{4}-\d{4}', text):
                        season_cell = cell
                        break
            
            if not season_cell:
                continue
                
            season = season_cell.get_text(strip=True)
            
            # Extract competition info (we want Premier League stats)
            comp_cell = row.find(['th', 'td'], {'data-stat': 'comp'})
            comp_name = comp_cell.get_text(strip=True) if comp_cell else ""
            
            # Accept Premier League or just league stats if competition not specified
            if comp_name and 'Premier League' not in comp_name and 'league' not in comp_name.lower():
                continue
                
            # Extract all stats from the row
            row_data = {
                'season': season,
                'table_type': table_type
            }
            
            all_cells = row.find_all(['th', 'td'])
            for idx, cell in enumerate(all_cells):
                if idx < len(headers):
                    stat_name = headers[idx]
                    stat_value = cell.get_text(strip=True)
                    row_data[stat_name] = stat_value
            
            # Add only if we have some meaningful stats
            if len(row_data) > 3:  # More than just season and table_type
                player_info['stats_by_season'].append(row_data)
    
    # Log how many stat rows we found
    logger.info(f"Extracted {len(player_info['stats_by_season'])} stat rows for {player_name}")
    
    # Organize stats by season and table type
    organized_stats = {}
    for stat_row in player_info['stats_by_season']:
        season = stat_row['season']
        table_type = stat_row['table_type']
        
        if season not in organized_stats:
            organized_stats[season] = {}
            
        organized_stats[season][table_type] = stat_row
    
    player_info['organized_stats'] = organized_stats
    return player_info

def convert_to_database_schema(player_info):
    """Convert player info into our database schema format"""
    # Setup basic player info
    player_data = {
        'name': player_info['name'],
        'age': player_info.get('age', 25),  # Default to 25 if age not found
        'nationality': player_info.get('nationality', 'Unknown'),
        'position': player_info.get('position', 'CF'),  # Default to Forward if not found
        'club': 'Manchester United',
        'league': 'Premier League',
        'height': 180.0,  # Default values
        'weight': 75.0,
        'preferred_foot': 'Right',
        'market_value': 50000000,  # Default value, we'll need other sources for real values
        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Process stats by season
    seasons_stats = []
    for season, season_data in player_info.get('organized_stats', {}).items():
        # Only process relatively recent seasons
        if not season[0:4].isdigit() or int(season[0:4]) < 2017:
            continue
            
        stats = {}
        stats['season'] = season
        
        # Standard stats
        std_stats = season_data.get('standard', {})
        if std_stats:
            stats['games_played'] = int(std_stats.get('games', '0').replace(',', '')) if std_stats.get('games', '0').replace(',', '').isdigit() else 0
            stats['minutes_played'] = int(std_stats.get('minutes', '0').replace(',', '')) if std_stats.get('minutes', '0').replace(',', '').isdigit() else 0
            stats['goals'] = int(std_stats.get('goals', '0').replace(',', '')) if std_stats.get('goals', '0').replace(',', '').isdigit() else 0
            stats['assists'] = int(std_stats.get('assists', '0').replace(',', '')) if std_stats.get('assists', '0').replace(',', '').isdigit() else 0
            
            # Calculate per 90 metrics
            mins_per90 = stats['minutes_played'] / 90 if stats['minutes_played'] > 0 else 1
            stats['goals_per90'] = round(stats['goals'] / mins_per90, 2) if mins_per90 > 0 else 0
            stats['assists_per90'] = round(stats['assists'] / mins_per90, 2) if mins_per90 > 0 else 0
            
            # Cards
            stats['yellow_cards'] = int(std_stats.get('cards_yellow', '0').replace(',', '')) if std_stats.get('cards_yellow', '0').replace(',', '').isdigit() else 0
            stats['red_cards'] = int(std_stats.get('cards_red', '0').replace(',', '')) if std_stats.get('cards_red', '0').replace(',', '').isdigit() else 0
        
        # Shooting stats
        shoot_stats = season_data.get('shooting', {})
        if shoot_stats:
            stats['shots'] = int(shoot_stats.get('shots', '0').replace(',', '')) if shoot_stats.get('shots', '0').replace(',', '').isdigit() else 0
            stats['shots_on_target'] = int(shoot_stats.get('shots_on_target', '0').replace(',', '')) if shoot_stats.get('shots_on_target', '0').replace(',', '').isdigit() else 0
            
            # Expected goals
            xg_str = shoot_stats.get('xg', '0')
            stats['xg'] = float(xg_str) if xg_str and xg_str != '' else 0
            
            npxg_str = shoot_stats.get('npxg', '0')
            stats['npxg'] = float(npxg_str) if npxg_str and npxg_str != '' else 0
            
            # Per 90 xG
            stats['xg_per90'] = round(stats['xg'] / mins_per90, 2) if mins_per90 > 0 else 0
            stats['npxg_per90'] = round(stats['npxg'] / mins_per90, 2) if mins_per90 > 0 else 0
        
        # Passing stats
        pass_stats = season_data.get('passing', {})
        if pass_stats:
            stats['passes_completed'] = int(pass_stats.get('passes_completed', '0').replace(',', '')) if pass_stats.get('passes_completed', '0').replace(',', '').isdigit() else 0
            stats['passes_attempted'] = int(pass_stats.get('passes', '0').replace(',', '')) if pass_stats.get('passes', '0').replace(',', '').isdigit() else 0
            
            # Pass completion percentage
            if stats['passes_attempted'] > 0:
                stats['pass_completion_pct'] = round((stats['passes_completed'] / stats['passes_attempted']) * 100, 2)
            else:
                stats['pass_completion_pct'] = 0
                
            # Expected assists
            xa_str = pass_stats.get('xa', '0')
            stats['xa'] = float(xa_str) if xa_str and xa_str != '' else 0
            stats['xa_per90'] = round(stats['xa'] / mins_per90, 2) if mins_per90 > 0 else 0
            
            # Progressive passes
            stats['progressive_passes'] = int(pass_stats.get('progressive_passes', '0').replace(',', '')) if pass_stats.get('progressive_passes', '0').replace(',', '').isdigit() else 0
            
            # Final third passes
            final_third_passes = int(pass_stats.get('passes_into_final_third', '0').replace(',', '')) if pass_stats.get('passes_into_final_third', '0').replace(',', '').isdigit() else 0
            stats['final_third_passes_attempted'] = final_third_passes
            stats['final_third_passes_completed'] = int(final_third_passes * stats['pass_completion_pct'] / 100)
        
        # Goal and Shot Creation stats
        gca_stats = season_data.get('gca', {})
        if gca_stats:
            stats['sca'] = int(gca_stats.get('sca', '0').replace(',', '')) if gca_stats.get('sca', '0').replace(',', '').isdigit() else 0
            stats['gca'] = int(gca_stats.get('gca', '0').replace(',', '')) if gca_stats.get('gca', '0').replace(',', '').isdigit() else 0
            
            # Per 90
            stats['sca_per90'] = round(stats['sca'] / mins_per90, 2) if mins_per90 > 0 else 0
            stats['gca_per90'] = round(stats['gca'] / mins_per90, 2) if mins_per90 > 0 else 0
            
            # Progressive carries and passes received
            stats['progressive_carries'] = int(gca_stats.get('carries_progressive', '0').replace(',', '')) if gca_stats.get('carries_progressive', '0').replace(',', '').isdigit() else 0
            stats['progressive_passes_received'] = int(gca_stats.get('passes_received_progressive', '0').replace(',', '')) if gca_stats.get('passes_received_progressive', '0').replace(',', '').isdigit() else 0
        
        # Defensive stats
        def_stats = season_data.get('defense', {})
        if def_stats:
            stats['tackles'] = int(def_stats.get('tackles', '0').replace(',', '')) if def_stats.get('tackles', '0').replace(',', '').isdigit() else 0
            stats['tackles_won'] = int(def_stats.get('tackles_won', '0').replace(',', '')) if def_stats.get('tackles_won', '0').replace(',', '').isdigit() else 0
            stats['interceptions'] = int(def_stats.get('interceptions', '0').replace(',', '')) if def_stats.get('interceptions', '0').replace(',', '').isdigit() else 0
            stats['blocks'] = int(def_stats.get('blocks', '0').replace(',', '')) if def_stats.get('blocks', '0').replace(',', '').isdigit() else 0
            stats['clearances'] = int(def_stats.get('clearances', '0').replace(',', '')) if def_stats.get('clearances', '0').replace(',', '').isdigit() else 0
            
            # Pressures
            stats['pressures'] = int(def_stats.get('pressures', '0').replace(',', '')) if def_stats.get('pressures', '0').replace(',', '').isdigit() else 0
            pressures_success = int(def_stats.get('pressure_regains', '0').replace(',', '')) if def_stats.get('pressure_regains', '0').replace(',', '').isdigit() else 0
            stats['pressure_success_rate'] = round((pressures_success / stats['pressures']) * 100, 2) if stats['pressures'] > 0 else 0
            
            # Aerial duels
            stats['aerial_duels_won'] = int(def_stats.get('aerials_won', '0').replace(',', '')) if def_stats.get('aerials_won', '0').replace(',', '').isdigit() else 0
            stats['aerial_duels_total'] = stats['aerial_duels_won'] * 2  # Estimate total from won
        
        # Add other stats with defaults
        if 'penalty_box_touches' not in stats:
            stats['penalty_box_touches'] = int(stats.get('progressive_carries', 0) / 2)  # Estimate
            
        if 'dribbles_attempted' not in stats:
            stats['dribbles_attempted'] = int(stats.get('progressive_carries', 0) / 2)  # Estimate
            
        if 'dribbles_completed' not in stats:
            stats['dribbles_completed'] = int(stats.get('dribbles_attempted', 0) * 0.6)  # Estimate 60% completion
            
        if 'ball_recoveries' not in stats:
            stats['ball_recoveries'] = stats.get('interceptions', 0) * 3  # Estimate
            
        if 'high_intensity_runs' not in stats:
            stats['high_intensity_runs'] = int(stats.get('progressive_carries', 0) * 1.5)  # Estimate
            
        if 'distance_covered' not in stats:
            # Estimate based on position and minutes
            if stats.get('minutes_played', 0) > 0:
                mins_per_match = stats.get('minutes_played', 0) / max(stats.get('games_played', 1), 1)
                matches_played = stats.get('games_played', 0)
                
                # Estimate distance in km: ~10km per full match
                distance_per_match = mins_per_match / 90 * 10
                stats['distance_covered'] = round(distance_per_match * matches_played, 2)
            else:
                stats['distance_covered'] = 0
        
        seasons_stats.append(stats)
    
    return player_data, seasons_stats

def insert_fbref_player_data(player_data, player_stats_list):
    """Insert player and stats data into the database"""
    try:
        conn = sqlite3.connect(db.DB_PATH)
        cursor = conn.cursor()
        
        # Check if player already exists
        cursor.execute("SELECT id FROM players WHERE name = ?", (player_data['name'],))
        existing_player = cursor.fetchone()
        
        player_id = None
        if existing_player:
            player_id = existing_player[0]
            # Update existing player
            cursor.execute('''
            UPDATE players SET
                age = ?,
                nationality = ?,
                position = ?,
                club = ?,
                league = ?,
                height = ?,
                weight = ?,
                preferred_foot = ?,
                market_value = ?,
                last_updated = ?
            WHERE id = ?
            ''', (
                player_data['age'],
                player_data['nationality'],
                player_data['position'],
                player_data['club'],
                player_data['league'],
                player_data['height'],
                player_data['weight'],
                player_data['preferred_foot'],
                player_data['market_value'],
                player_data['last_updated'],
                player_id
            ))
            logger.info(f"Updated existing player: {player_data['name']} (ID: {player_id})")
        else:
            # Get next ID
            cursor.execute("SELECT MAX(id) FROM players")
            max_id = cursor.fetchone()[0]
            player_id = 1 if max_id is None else max_id + 1
            
            # Insert new player
            cursor.execute('''
            INSERT INTO players (
                id, name, age, nationality, position, club, league,
                height, weight, preferred_foot, market_value, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                player_id,
                player_data['name'],
                player_data['age'],
                player_data['nationality'],
                player_data['position'],
                player_data['club'],
                player_data['league'],
                player_data['height'],
                player_data['weight'],
                player_data['preferred_foot'],
                player_data['market_value'],
                player_data['last_updated']
            ))
            logger.info(f"Inserted new player: {player_data['name']} (ID: {player_id})")
        
        # Insert player stats for each season
        stats_inserted = 0
        stats_updated = 0
        
        for stats in player_stats_list:
            # Check if stats for this season already exist
            cursor.execute(
                "SELECT id FROM player_stats WHERE player_id = ? AND season = ?", 
                (player_id, stats['season'])
            )
            existing_stats = cursor.fetchone()
            
            if existing_stats:
                # Update existing stats
                # Dynamically create the update query based on available fields
                available_fields = [
                    field for field in stats.keys() 
                    if field != 'season' and field in [
                        'goals', 'assists', 'xg', 'xa', 'npxg', 'sca', 'gca', 'shots', 
                        'shots_on_target', 'progressive_carries', 'progressive_passes',
                        'penalty_box_touches', 'passes_completed', 'passes_attempted',
                        'pass_completion_pct', 'progressive_passes_received',
                        'final_third_passes_completed', 'final_third_passes_attempted',
                        'dribbles_completed', 'dribbles_attempted', 'ball_recoveries',
                        'tackles', 'tackles_won', 'interceptions', 'blocks', 'clearances',
                        'pressures', 'pressure_success_rate', 'aerial_duels_won',
                        'aerial_duels_total', 'minutes_played', 'games_played',
                        'distance_covered', 'high_intensity_runs', 'yellow_cards',
                        'red_cards', 'goals_per90', 'assists_per90', 'xg_per90',
                        'xa_per90', 'npxg_per90', 'sca_per90', 'gca_per90'
                    ]
                ]
                
                update_fields = ", ".join([f"{field} = ?" for field in available_fields])
                update_values = [stats[field] for field in available_fields]
                update_values.append(existing_stats[0])  # Add stats ID for WHERE clause
                
                cursor.execute(f'''
                UPDATE player_stats SET
                    {update_fields}
                WHERE id = ?
                ''', update_values)
                
                stats_updated += 1
                logger.debug(f"Updated stats for {player_data['name']} ({stats['season']})")
            else:
                # Get next ID for stats
                cursor.execute("SELECT MAX(id) FROM player_stats")
                max_stats_id = cursor.fetchone()[0]
                stats_id = 1 if max_stats_id is None else max_stats_id + 1
                
                # Create columns and values for the query
                stat_columns = ['id', 'player_id', 'season']
                stat_values = [stats_id, player_id, stats['season']]
                
                for stat_name, stat_value in stats.items():
                    if stat_name != 'season':
                        stat_columns.append(stat_name)
                        stat_values.append(stat_value)
                
                columns_str = ", ".join(stat_columns)
                placeholders = ", ".join(["?"] * len(stat_columns))
                
                cursor.execute(f'''
                INSERT INTO player_stats ({columns_str})
                VALUES ({placeholders})
                ''', stat_values)
                
                stats_inserted += 1
                logger.debug(f"Inserted stats for {player_data['name']} ({stats['season']})")
        
        conn.commit()
        logger.info(f"Processed stats for {player_data['name']}: {stats_inserted} inserted, {stats_updated} updated")
        return True
    
    except Exception as e:
        logger.error(f"Error inserting player data: {str(e)}")
        if conn:
            conn.rollback()
        return False
    
    finally:
        if conn:
            conn.close()

def scrape_team_data(team_id='19538871', team_name='Manchester United', start_season=2017, end_season=2023):
    """Scrape data for all players in a team across multiple seasons"""
    logger.info(f"Starting scraping for {team_name} from {start_season} to {end_season}")
    
    # Get URLs for all seasons
    season_urls = get_season_urls(team_id, start_season, end_season)
    
    all_player_urls = set()  # Use a set to avoid duplicates
    
    # Get player URLs from each season's roster
    for season, season_url in season_urls:
        # Try multiple URL patterns for roster pages
        possible_roster_urls = [
            season_url.replace('Stats', 'roster/Manchester-United-Roster-Details'),
            f"{BASE_URL}/en/squads/{team_id}/{season}/roster/",
            season_url  # Try the season page itself, as it may contain player links
        ]
        
        for roster_url in possible_roster_urls:
            logger.info(f"Getting player URLs from roster: {season} - {roster_url}")
            player_urls = get_player_urls_from_roster(roster_url)
            
            # If we found players, add them and move to next season
            if player_urls:
                for player_name, player_url in player_urls:
                    all_player_urls.add((player_name, player_url))
                break  # Found players, no need to try other URL patterns
    
    logger.info(f"Found {len(all_player_urls)} unique players across all seasons")
    
    # Process each player
    successful_imports = 0
    failed_imports = 0
    
    for player_name, player_url in all_player_urls:
        try:
            logger.info(f"Processing player: {player_name} - {player_url}")
            
            # Parse player stats
            player_info = parse_player_stats(player_url, player_name)
            if not player_info:
                logger.warning(f"Failed to parse stats for player: {player_name}")
                failed_imports += 1
                continue
            
            # Convert to our database schema
            player_data, player_stats_list = convert_to_database_schema(player_info)
            
            # Insert into database
            if insert_fbref_player_data(player_data, player_stats_list):
                successful_imports += 1
            else:
                failed_imports += 1
        
        except Exception as e:
            logger.error(f"Error processing player {player_name}: {str(e)}")
            failed_imports += 1
    
    logger.info(f"Completed scraping for {team_name}")
    logger.info(f"Successfully imported {successful_imports} players")
    logger.info(f"Failed to import {failed_imports} players")
    
    return successful_imports, failed_imports

if __name__ == "__main__":
    # Ensure database exists
    if not os.path.exists(db.DB_PATH):
        db.create_database()
        logger.info("Created new database")
    
    # Scrape Manchester United data from 2017-2018 to 2022-2023
    scrape_team_data(
        team_id='19538871',
        team_name='Manchester United',
        start_season=2017,
        end_season=2022
    )