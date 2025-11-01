# Atlas Query

An interactive natural language query system for US county geographic data. This project combines a PostGIS backend that converts natural language queries into SQL with an interactive React/D3 map frontend for visualizing results.

## Features

- **Natural Language to SQL**: Query US county data using plain English
- **Interactive Map Visualization**: Explore counties on an interactive US map with zoom and pan
- **Real-time Query Results**: See query results overlaid on the map as GeoJSON features
- **Query History**: Track and revisit previous queries
- **Multiple LLM Providers**: Support for Ollama and HuggingFace models

## Project Structure

```
.
├── interactive-us-county-map/    # React frontend application
│   ├── components/               # React components
│   │   ├── USMap.tsx            # Interactive map with zoom/pan
│   │   ├── QueryBar.tsx         # Query input interface
│   │   ├── HistoryPanel.tsx     # Query history display
│   │   └── ...
│   ├── App.tsx                  # Main application component
│   └── package.json             # Frontend dependencies
├── nl2sql_backend.py            # Core NL→SQL conversion engine
├── api.py                       # FastAPI backend server
├── llama.py                     # LLM provider integrations
├── gen_sql_only.py             # SQL generation utilities
└── requirements.txt             # Python dependencies
```

## Prerequisites

- Python 3.8+
- Node.js 16+
- PostgreSQL with PostGIS extension
- USCountyDB database with counties table

## Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (create `.env` file):
```bash
PGDATABASE=USCountyDB
PGHOST=localhost
PGPORT=5432
PGUSER=postgres
PGPASSWORD=your_password

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini
```

3. Start the backend server:
```bash
python api.py
```

The backend API will be available at `http://localhost:8000`

## Frontend Setup

1. Navigate to the frontend directory:
```bash
cd interactive-us-county-map
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Usage

### Querying Counties

Enter natural language queries like:
- "visualize Madison County"
- "Show all counties in California"
- "Counties with area less than 100 square miles"
- "Rank all counties in Arizona by area"

### Map Interaction

- **Zoom**: Scroll wheel to zoom in/out
- **Pan**: Click and drag to move around
- **Click County**: Click any county to see details and visualize it
- **Reset View**: Use the "Reset view" button to return to default zoom

## API Endpoints

- `POST /answer` - Submit a natural language query
  ```json
  {
    "query": "visualize Madison County",
    "provider": "ollama",
    "model": "phi3:mini"
  }
  ```

- `GET /county/{geoid}` - Get county data by GEOID

## Technologies

- **Backend**: Python, FastAPI, PostGIS, Ollama/HuggingFace
- **Frontend**: React, TypeScript, D3.js, Vite
- **Database**: PostgreSQL with PostGIS extension

## License

MIT

