import sqlite3
import os
import pandas as pd
import random
from datetime import datetime, timedelta

# Database setup
DB_PATH = os.path.join(os.path.dirname(__file__), 'soccer_scout.db')
print(DB_PATH)

def create_database():
    """Create the SQLite database with tables for players and statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Players table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS players (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        nationality TEXT,
        position TEXT,
        club TEXT,
        league TEXT,
        height FLOAT,
        weight FLOAT,
        preferred_foot TEXT,
        market_value INTEGER,
        last_updated TIMESTAMP
    )
    ''')
    
    # Statistics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS player_stats (
        id INTEGER PRIMARY KEY,
        player_id INTEGER,
        season TEXT,
        
        -- Offensive metrics
        goals INTEGER,
        assists INTEGER,
        xg FLOAT,
        xa FLOAT,
        npxg FLOAT,
        sca INTEGER,
        gca INTEGER,
        shots INTEGER,
        shots_on_target INTEGER,
        progressive_carries INTEGER,
        progressive_passes INTEGER,
        penalty_box_touches INTEGER,
        
        -- Possession metrics
        passes_completed INTEGER,
        passes_attempted INTEGER,
        pass_completion_pct FLOAT,
        progressive_passes_received INTEGER,
        final_third_passes_completed INTEGER,
        final_third_passes_attempted INTEGER,
        dribbles_completed INTEGER,
        dribbles_attempted INTEGER,
        ball_recoveries INTEGER,
        
        -- Defensive metrics
        tackles INTEGER,
        tackles_won INTEGER,
        interceptions INTEGER,
        blocks INTEGER,
        clearances INTEGER,
        pressures INTEGER,
        pressure_success_rate FLOAT,
        aerial_duels_won INTEGER,
        aerial_duels_total INTEGER,
        
        -- Physical/tactical metrics
        minutes_played INTEGER,
        games_played INTEGER,
        distance_covered FLOAT,
        high_intensity_runs INTEGER,
        yellow_cards INTEGER,
        red_cards INTEGER,
        
        -- Per 90 metrics
        goals_per90 FLOAT,
        assists_per90 FLOAT,
        xg_per90 FLOAT,
        xa_per90 FLOAT,
        npxg_per90 FLOAT,
        sca_per90 FLOAT,
        gca_per90 FLOAT,
        
        FOREIGN KEY (player_id) REFERENCES players(id)
    )
    ''')
    
    conn.commit()
    conn.close()

def generate_random_player_data():
    """Generate random player data for simulation purposes"""
    # Names, nationalities, clubs, and leagues for simulation
    first_names = ["Lionel", "Cristiano", "Kylian", "Erling", "Kevin", "Robert", "Neymar", "Mohamed", "Bruno", "Virgil", 
                 "Harry", "Joshua", "Trent", "Phil", "Mason", "Jack", "Jude", "Bukayo", "Jadon", "Timo", "Lucas"]
    last_names = ["Messi", "Ronaldo", "Mbappe", "Haaland", "De Bruyne", "Lewandowski", "Jr", "Salah", "Fernandes", "van Dijk", 
                "Kane", "Kimmich", "Alexander-Arnold", "Foden", "Mount", "Grealish", "Bellingham", "Saka", "Sancho", "Werner", "Paqueta"]
    
    nationalities = ["Argentina", "Portugal", "France", "Norway", "Belgium", "Poland", "Brazil", "Egypt", "Portugal", "Netherlands", 
                     "England", "Germany", "England", "England", "England", "England", "England", "England", "England", "Germany", "Brazil"]
    
    positions = ["CF", "CF", "CF", "CF", "CAM", "CF", "LW", "RW", "CAM", "CB", 
                "CF", "CDM", "RB", "CAM", "CAM", "LW", "CM", "RW", "RW", "CF", "CM"]
    
    clubs = ["Inter Miami", "Al-Nassr", "Real Madrid", "Manchester City", "Manchester City", "Barcelona", "Al-Hilal", "Liverpool", 
             "Manchester United", "Liverpool", "Bayern Munich", "Bayern Munich", "Liverpool", "Manchester City", "Chelsea", 
             "Manchester City", "Real Madrid", "Arsenal", "Borussia Dortmund", "Tottenham", "West Ham"]
    
    leagues = ["MLS", "Saudi Pro League", "La Liga", "Premier League", "Premier League", "La Liga", "Saudi Pro League", "Premier League", 
              "Premier League", "Premier League", "Bundesliga", "Bundesliga", "Premier League", "Premier League", "Premier League", 
              "Premier League", "La Liga", "Premier League", "Bundesliga", "Premier League", "Premier League"]
    
    market_values = [
        35000000, 15000000, 180000000, 150000000, 100000000, 30000000, 60000000, 80000000, 75000000, 40000000,
        90000000, 70000000, 65000000, 100000000, 60000000, 65000000, 120000000, 100000000, 60000000, 30000000, 45000000
    ]
    
    heights = [169, 187, 178, 194, 181, 185, 175, 175, 179, 193, 188, 177, 175, 171, 178, 175, 186, 178, 180, 180, 180]
    weights = [72, 85, 73, 88, 70, 81, 68, 71, 69, 92, 86, 70, 69, 70, 74, 68, 75, 70, 76, 75, 72]
    feet = ["Left", "Right", "Right", "Left", "Right", "Right", "Right", "Left", "Right", "Right", "Right", "Right", "Right", "Left", "Right", "Right", "Right", "Left", "Right", "Right", "Left"]
    
    players = []
    for i in range(len(first_names)):
        player = {
            "id": i + 1,
            "name": f"{first_names[i]} {last_names[i]}",
            "age": random.randint(19, 36),
            "nationality": nationalities[i],
            "position": positions[i],
            "club": clubs[i],
            "league": leagues[i],
            "height": heights[i],
            "weight": weights[i],
            "preferred_foot": feet[i],
            "market_value": market_values[i],
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        players.append(player)
    
    return players

def generate_player_stats(player_id, seasons=["2021-2022", "2022-2023", "2023-2024"]):
    """Generate realistic player statistics for simulation"""
    stats_list = []
    
    for season in seasons:
        # Base stats - these will vary by player_id to create different player profiles
        skill_factor = 0.5 + (player_id % 10) / 10  # Create variation between players
        
        # Minutes and games
        games_played = random.randint(25, 38)
        minutes_played = games_played * random.randint(70, 90)
        
        # Create different player types based on player_id
        is_forward = player_id in [1, 2, 3, 4, 7, 8, 10, 20]
        is_midfielder = player_id in [5, 9, 11, 13, 14, 15, 16, 17, 18, 21]
        is_defender = player_id in [6, 12, 19]
        
        # Offensive stats
        if is_forward:
            goals = int(random.randint(15, 30) * skill_factor)
            assists = int(random.randint(5, 15) * skill_factor)
            xg = goals * random.uniform(0.8, 1.2)
            xa = assists * random.uniform(0.8, 1.2)
            shots = int(goals * random.uniform(3, 5))
            shots_on_target = int(shots * random.uniform(0.4, 0.6))
        elif is_midfielder:
            goals = int(random.randint(5, 12) * skill_factor)
            assists = int(random.randint(8, 20) * skill_factor)
            xg = goals * random.uniform(0.8, 1.2)
            xa = assists * random.uniform(0.8, 1.2)
            shots = int(goals * random.uniform(3, 5))
            shots_on_target = int(shots * random.uniform(0.3, 0.5))
        else:  # defender
            goals = int(random.randint(1, 5) * skill_factor)
            assists = int(random.randint(1, 8) * skill_factor)
            xg = goals * random.uniform(0.8, 1.2)
            xa = assists * random.uniform(0.8, 1.2)
            shots = int(goals * random.uniform(2, 4))
            shots_on_target = int(shots * random.uniform(0.3, 0.5))
        
        # More offensive stats
        npxg = xg - (random.randint(0, 5) * 0.76)  # Remove penalties
        sca = int(random.randint(50, 150) * skill_factor)
        gca = int(sca * random.uniform(0.05, 0.15))
        progressive_carries = int(random.randint(50, 200) * skill_factor)
        progressive_passes = int(random.randint(50, 300) * skill_factor)
        penalty_box_touches = int(random.randint(20, 200) * skill_factor)
        
        # Possession stats
        passes_attempted = int(random.randint(1000, 2500) * skill_factor)
        pass_completion_pct = random.uniform(0.7, 0.92)
        passes_completed = int(passes_attempted * pass_completion_pct)
        progressive_passes_received = int(random.randint(50, 250) * skill_factor)
        final_third_passes_attempted = int(passes_attempted * random.uniform(0.2, 0.4))
        final_third_passes_completed = int(final_third_passes_attempted * random.uniform(0.6, 0.85))
        dribbles_attempted = int(random.randint(50, 200) * skill_factor)
        dribbles_completed = int(dribbles_attempted * random.uniform(0.4, 0.75))
        ball_recoveries = int(random.randint(100, 300) * skill_factor)
        
        # Defensive stats
        if is_defender:
            tackles = int(random.randint(70, 120) * skill_factor)
            interceptions = int(random.randint(50, 100) * skill_factor)
            blocks = int(random.randint(40, 80) * skill_factor)
            clearances = int(random.randint(100, 200) * skill_factor)
        elif is_midfielder:
            tackles = int(random.randint(40, 90) * skill_factor)
            interceptions = int(random.randint(30, 70) * skill_factor)
            blocks = int(random.randint(20, 50) * skill_factor)
            clearances = int(random.randint(20, 60) * skill_factor)
        else:  # forward
            tackles = int(random.randint(10, 40) * skill_factor)
            interceptions = int(random.randint(5, 30) * skill_factor)
            blocks = int(random.randint(5, 25) * skill_factor)
            clearances = int(random.randint(5, 20) * skill_factor)
        
        tackles_won = int(tackles * random.uniform(0.6, 0.8))
        pressures = int(random.randint(300, 700) * skill_factor)
        pressure_success_rate = random.uniform(0.25, 0.4)
        aerial_duels_total = int(random.randint(50, 200) * skill_factor)
        aerial_duels_won = int(aerial_duels_total * random.uniform(0.4, 0.7))
        
        # Physical stats
        distance_covered = random.uniform(220, 340)  # in km
        high_intensity_runs = int(random.randint(100, 300) * skill_factor)
        yellow_cards = random.randint(1, 10)
        red_cards = random.randint(0, 2)
        
        # Per 90 metrics
        mins_per90 = minutes_played / 90
        goals_per90 = goals / mins_per90
        assists_per90 = assists / mins_per90
        xg_per90 = xg / mins_per90
        xa_per90 = xa / mins_per90
        npxg_per90 = npxg / mins_per90
        sca_per90 = sca / mins_per90
        gca_per90 = gca / mins_per90
        
        stats = {
            "player_id": player_id,
            "season": season,
            "goals": goals,
            "assists": assists,
            "xg": round(xg, 2),
            "xa": round(xa, 2),
            "npxg": round(npxg, 2),
            "sca": sca,
            "gca": gca,
            "shots": shots,
            "shots_on_target": shots_on_target,
            "progressive_carries": progressive_carries,
            "progressive_passes": progressive_passes,
            "penalty_box_touches": penalty_box_touches,
            "passes_completed": passes_completed,
            "passes_attempted": passes_attempted,
            "pass_completion_pct": round(pass_completion_pct, 2),
            "progressive_passes_received": progressive_passes_received,
            "final_third_passes_completed": final_third_passes_completed,
            "final_third_passes_attempted": final_third_passes_attempted,
            "dribbles_completed": dribbles_completed,
            "dribbles_attempted": dribbles_attempted,
            "ball_recoveries": ball_recoveries,
            "tackles": tackles,
            "tackles_won": tackles_won,
            "interceptions": interceptions,
            "blocks": blocks,
            "clearances": clearances,
            "pressures": pressures,
            "pressure_success_rate": round(pressure_success_rate, 2),
            "aerial_duels_won": aerial_duels_won,
            "aerial_duels_total": aerial_duels_total,
            "minutes_played": minutes_played,
            "games_played": games_played,
            "distance_covered": round(distance_covered, 2),
            "high_intensity_runs": high_intensity_runs,
            "yellow_cards": yellow_cards,
            "red_cards": red_cards,
            "goals_per90": round(goals_per90, 2),
            "assists_per90": round(assists_per90, 2),
            "xg_per90": round(xg_per90, 2),
            "xa_per90": round(xa_per90, 2),
            "npxg_per90": round(npxg_per90, 2),
            "sca_per90": round(sca_per90, 2),
            "gca_per90": round(gca_per90, 2),
        }
        
        stats_list.append(stats)
    
    return stats_list

def populate_database():
    """Populate the database with simulation data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Generate player data
    players = generate_random_player_data()
    
    # Insert players into database
    for player in players:
        cursor.execute('''
        INSERT INTO players (id, name, age, nationality, position, club, league, height, weight, preferred_foot, market_value, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            player["id"], player["name"], player["age"], player["nationality"], player["position"], 
            player["club"], player["league"], player["height"], player["weight"], player["preferred_foot"], 
            player["market_value"], player["last_updated"]
        ))
    
    # Generate and insert player stats
    for player in players:
        stats_list = generate_player_stats(player["id"])
        for stats in stats_list:
            placeholders = ", ".join(["?"] * len(stats))
            columns = ", ".join(stats.keys())
            values = tuple(stats.values())
            
            cursor.execute(f'''
            INSERT INTO player_stats ({columns})
            VALUES ({placeholders})
            ''', values)
    
    conn.commit()
    conn.close()
    print(f"Database populated with {len(players)} players and their statistics.")

def get_all_players():
    """Retrieve all players from the database"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM players", conn)
    conn.close()
    return df

def get_player_stats(player_id=None, season="2023-2024"):
    """Retrieve player statistics from the database"""
    conn = sqlite3.connect(DB_PATH)
    
    if player_id:
        query = f"SELECT * FROM player_stats WHERE player_id = {player_id} AND season = '{season}'"
    else:
        query = f"SELECT * FROM player_stats WHERE season = '{season}'"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_players_with_stats(season="2023-2024"):
    """Retrieve players with their statistics for a given season"""
    conn = sqlite3.connect(DB_PATH)
    
    query = f'''
    SELECT p.*, ps.*
    FROM players p
    JOIN player_stats ps ON p.id = ps.player_id
    WHERE ps.season = '{season}'
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Initialize database if it doesn't exist
if not os.path.exists(DB_PATH):
    create_database()
    populate_database()