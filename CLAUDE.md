# CLAUDE.md - SoccerScout Project Guidelines

## Commands
- Run app: `cd src && python app.py`
- Access UI: http://127.0.0.1:5000/players
- Install dependencies: `pip install flask flask_cors pandas`
- Lint: `pip install flake8 && flake8 src/`
- Format code: `pip install black && black src/`

## Code Style
- **Python**: Follow PEP 8 guidelines
- **Imports**: Group standard library, third-party, and local imports
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Types**: Use type hints where possible
- **Error handling**: Use try/except blocks for web requests and data processing
- **Comments**: Docstrings for functions, inline comments for complex logic
- **HTML/JS**: 2-space indentation, use consistent quotes (prefer double)

## Git Workflow
- Work on `development` branch
- Create PRs to `main`
- Document new features and bugs in Issues