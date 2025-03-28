# SoccerScout

SoccerScout is a tool for soccer clubs to identify undervalued and overvalued players by analyzing advanced statistics and comparing predicted values to actual market values.

## Overview

This application analyzes player performance using a comprehensive set of advanced statistics including:
- Expected goals (xG) and assists (xA)
- Shot-creating actions (SCA) and goal-creating actions (GCA)
- Progressive passes and carries
- Defensive metrics like tackles, interceptions, and pressures
- Physical and tactical data

The current version uses simulated player data in a SQLite database, with plans to integrate with real-world APIs and implement machine learning models.

## Features

- **Player Performance Scoring**: Calculate weighted performance scores based on 30+ advanced metrics
- **Market Value Comparison**: Compare performance to market value to identify undervalued/overvalued players
- **Interactive UI**: Filter and search for players, view detailed statistics
- **Data Visualization**: View statistical distributions across metrics

## Tech Stack

- **Backend**: Python, Flask, SQLite, Pandas
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Data**: Simulated soccer statistics database with 20+ players and 3 seasons of data

## Development Goals

1. Implement ML model to predict player market values
2. Use historical data to validate predictions
3. Add more advanced metrics and visualization tools
4. Incorporate real-world data from soccer APIs

## Dev Guidelines

- Push any changes to the `development` branch and open a PR
- Add bugs and features to Issues

## Setup and Running

1. Install dependencies:
   ```
   pip install flask flask_cors pandas
   ```

2. Run the application:
   ```
   cd src
   python app.py
   ```

3. Access the UI in your browser:
   ```
   http://127.0.0.1:5000/
   ```

## API Endpoints

- `/players` - Get all players with scores and stats
- `/player/<id>` - Get detailed stats for a specific player
- `/top-undervalued` - Get top 10 undervalued players
- `/stats-distribution` - Get statistical distributions for key metrics

## Potential Future Data Sources:

Paid:  
- Wyscout https://www.hudl.com/products/wyscout  
- Opta Sports API

Free:  
- FBref
- Understat  
- Whoscored

## Resources:

- [Scraping FBref with Python](https://medium.com/@ricardoandreom/how-to-scrape-and-personalize-data-from-fbref-with-python-a-guide-to-unlocking-football-insights-7e623607afca)
- [Soccer API Scraping Guide](https://ctomashot.medium.com/how-i-scraped-an-api-for-my-soccer-scouting-app-c67df68da6ca)
- [Machine Learning Algorithms Overview](https://builtin.com/data-science/tour-top-10-algorithms-machine-learning-newbies)
