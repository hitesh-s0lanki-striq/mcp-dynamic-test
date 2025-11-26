# SEO Agent with MCP Tools

An intelligent SEO agent that uses MCP (Model Context Protocol) tools to help with SEO analysis, keyword research, and search analytics.

## Features

- **SEO Analysis**: Get insights on keyword coverage, search performance, and more
- **Google Search Console Integration**: Access GSC data through MCP tools
- **DataForSEO Integration**: Leverage DataForSEO APIs for comprehensive SEO analysis
- **Interactive Chat UI**: Simple Streamlit interface for easy interaction

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
OPENAI_API_KEY=your_openai_api_key
# Add other required API keys for MCP servers
```

## Usage

### Streamlit Chat UI

Run the interactive chat interface:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser where you can:
- Ask SEO-related questions
- Get insights on keyword coverage
- Analyze search performance
- And much more!

### Python Script

You can also use the agent programmatically:

```python
from langchain_openai import ChatOpenAI
from src.tools.seo_tools import SeoTools
from src.agents.seo_agent import SEOAgent

llm = ChatOpenAI(model="gpt-4o", temperature=0)
seo_tools = SeoTools()
seo_agent = SEOAgent(llm, seo_tools)

user_query = "How can I increase the organic keyword coverage for strique.io?"
final_answer = await seo_agent.run_and_respond(user_query)
print(final_answer)
```

## Configuration

You can set the OpenAI model via environment variable:
```bash
export OPENAI_MODEL="gpt-4o"  # or "gpt-4.1" or any other model
```

## Project Structure

```
├── streamlit_app.py          # Streamlit chat UI
├── app.py                    # Main application entry point
├── src/
│   ├── agents/              # SEO agent components
│   ├── tools/               # MCP tool integrations
│   ├── instructions/        # Agent instructions
│   └── utils/               # Utility functions
└── requirements.txt         # Python dependencies
```


