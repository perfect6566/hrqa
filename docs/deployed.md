# Deployed Application

## Deployed URL

**Application URL**: TBD (configure after deployment to Render/Railway)

**Health Endpoint**: `{APP_URL}/health`

**MCP Status Endpoint**: `{APP_URL}/mcp/status`

## Deployment Platform

**Platform**: Render / Railway / Equivalent

**Tier**: Free-tier compatible

## Deployment Configuration

### Required Environment Variables

```bash
# API Configuration
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-v4-flash

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Paths
POLICIES_DIR=policies
MOCK_DATA_DIR=mock_data
VECTOR_STORE_PATH=./data/vector_store

# Server Ports
PORT=8000
MCP_PORT=8001
```

### Render Deployment Steps

1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python -m src.api.main`
4. Add environment variables in Render dashboard
5. Deploy

### Railway Deployment Steps

1. Create new Railway project
2. Connect repository
3. Add environment variables
4. Deploy

## Cold Start Behavior

The free-tier deployment may experience cold starts:

1. **Initial Load**: 30-60 seconds to start services
2. **After Inactivity**: 10-30 seconds to wake up
3. **MCP Server**: Starts with main application as subprocess
4. **RAG Index**: Loaded on startup

### Expected Behavior

- First request after inactivity may timeout
- Subsequent requests should work normally
- Vector store is loaded on startup
- MCP tools are available immediately

### Health Check Before Demo

To warm up the service before demo:

```bash
# Ping health endpoint
curl https://your-app.railway.app/health

# Or MCP status
curl https://your-app.railway.app/mcp/status
```

## Architecture

The deployed application runs as a single service:

```
┌─────────────────────────────────────────────┐
│           Main Application (Port 8000)         │
│  ┌─────────────────────────────────────┐    │
│  │         FastAPI Web Application         │    │
│  └─────────────────────────────────────┘    │
│                    │                           │
│  ┌─────────────────┴───────────────────┐    │
│  │         Agent Orchestrator              │    │
│  │  (with MCP Client)                    │    │
│  └─────────────────┬───────────────────┘    │
│                    │                           │
│  ┌─────────────────┴───────────────────┐    │
│  │    MCP Server (Subprocess)          │    │
│  │    (Port 8001)                    │    │
│  └───────────────────────────────────┘    │
│                    │                           │
│  ┌─────────────────┴───────────────────┐    │
│  │    RAG Pipeline + Mock Data        │    │
│  └───────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

## Local Development

To run locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment
# Copy .env.example to .env and fill in your API key
cp .env.example .env

# Start MCP server (terminal 1)
cd src/mcp
python app.py

# Start main app (terminal 2)
cd src/api
python main.py

# Or run everything together
python -m src.api.main
```

## Troubleshooting

### Application Won't Start
- Check OPENAI_API_KEY is set
- Verify policies directory exists
- Check for Python import errors

### MCP Tools Not Working
- MCP server starts automatically with main app
- Check /mcp/status endpoint
- Verify network connectivity

### RAG Not Finding Documents
- Verify policies directory has .md files
- Check vector store was created
- Try rebuilding index

### Cold Start Issues
- Ping health endpoint before making requests
- Use a keep-alive service if available
- Consider upgrading to paid tier

## Monitoring

- Health endpoint: `GET /health`
- MCP status: `GET /mcp/status`
- Returns MCP connectivity status
- Shows index status and tool count
