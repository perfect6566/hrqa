# HR Policy Assistant

An agentic AI system for HR policy and operations tasks using RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol).

## Features

- **Policy RAG**: Semantic search over company policy documents with citations
- **Agentic Workflows**: Multi-step HR tasks including PTO requests, remote work eligibility, benefits questions
- **MCP Integration**: 8 tools exposed via Model Context Protocol
- **Web Interface**: Chat interface for HR policy questions
- **Deployment Ready**: Free-tier compatible

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd hrqa

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"  # Windows: set OPENAI_API_KEY=your-api-key

# Run the application
python -m src.api.main
```

### Running MCP Server Separately

```bash
# Terminal 1: Start MCP server
cd src/mcp
python app.py

# Terminal 2: Start main application
cd src/api
python main.py
```

### Running Together

```bash
python -m src.api.main
```

The application will start on http://localhost:8000

## Project Structure

```
hrqa/
├── policies/                    # HR policy documents (markdown)
│   ├── pto-policy.md
│   ├── remote-work-policy.md
│   ├── expense-policy.md
│   ├── data-security-policy.md
│   ├── benefits-policy.md
│   ├── onboarding-policy.md
│   ├── leave-policy.md
│   ├── workplace-conduct-policy.md
│   └── equipment-policy.md
├── mock_data/                   # Synthetic employee data
│   ├── employees.json
│   ├── pto_balances.json
│   ├── benefits.json
│   └── hr_tickets.json
├── src/
│   ├── rag/                    # RAG system
│   │   ├── document_loader.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   ├── generator.py
│   │   └── rag_pipeline.py
│   ├── mcp/                    # MCP server
│   │   ├── server.py
│   │   ├── tools.py
│   │   ├── client.py
│   │   └── app.py
│   ├── agent/                  # Agent orchestrator
│   │   ├── planner.py
│   │   ├── executor.py
│   │   └── orchestrator.py
│   └── api/                    # FastAPI web app
│       └── main.py
├── tests/                       # Unit tests
├── evaluation/                  # Evaluation scripts
├── docs/                        # Documentation
├── .github/workflows/          # CI/CD pipeline
└── requirements.txt
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check with MCP status |
| `/chat` | POST | Send HR question |
| `/capabilities` | GET | Get agent capabilities |
| `/demo/pto-request` | GET | Run PTO demo |
| `/demo/remote-work` | GET | Run remote work demo |

### Example Chat Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Can I take 3 days of PTO next week?",
    "employee_id": "EMP001"
  }'
```

## MCP Tools (8 Tools)

1. **lookup_employee_profile** - Get employee information
2. **check_pto_balance** - Check PTO balance
3. **lookup_benefits_status** - Check benefits enrollment
4. **create_mock_hr_ticket** - Create HR ticket (mock)
5. **draft_hr_email** - Draft HR email (mock)
6. **check_policy_compliance** - Check policy compliance
7. **search_policy_documents** - Search policy documents
8. **get_policy_section** - Get specific policy section

## Demo Tasks

### Demo 1: PTO Request Guidance

```
User: Can I take 3 days of PTO next week?

Expected Flow:
1. Look up employee profile
2. Check PTO balance
3. Search PTO policy
4. Provide guidance with citations
```

### Demo 2: Remote Work Eligibility

```
User: Can I work remotely from another state for 6 weeks?

Expected Flow:
1. Look up employee profile
2. Check remote work compliance
3. Search remote work policy
4. Provide eligibility assessment
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_rag.py -v

# Run with coverage
pytest --cov=src tests/
```

## Evaluation

```bash
# Run evaluation
python -m evaluation.run_evaluation

# Results saved to evaluation/results.json
```

## Deployment

### Render (Free Tier)

1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python -m src.api.main`
4. Add environment variable: `OPENAI_API_KEY`

### Railway

1. Create new Railway project
2. Connect repository
3. Configure environment variables
4. Deploy

## Documentation

- [Design & Evaluation](docs/design-and-evaluation.md) - Architecture and evaluation results
- [AI Tooling Usage](docs/ai-tooling.md) - How AI tools were used
- [Deployment Guide](docs/deployed.md) - Deployment instructions

## License

MIT
