# QA Operator Installation Guide

This guide covers installation and setup of QA Operator for different environments and use cases.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js and npm (for Playwright MCP server)
- Git (optional, for version control features)

### Installation

#### Option 1: Install from Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/your-org/qa-operator.git
cd qa-operator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Initialize QA Operator in your project
qa-operator init
```

#### Option 2: Direct Execution (No Installation)

```bash
# Clone the repository
git clone https://github.com/your-org/qa-operator.git
cd qa-operator

# Install dependencies
pip install -r requirements.txt

# Run directly
./qa-operator --help
```

#### Option 3: Install from PyPI (When Available)

```bash
# Install from PyPI
pip install qa-operator

# Initialize in your project
qa-operator init
```

## System Requirements

### Python Requirements

- **Python 3.8+**: Required (Python 3.10+ recommended)
- **Virtual Environment**: Highly recommended to avoid conflicts

### Node.js Requirements

- **Node.js 16+**: Required for Playwright MCP server
- **npm**: Required for installing Playwright dependencies

### Optional Requirements

- **Git**: For version control integration
- **uvx**: For Git MCP server (install with `pip install uv`)
- **Ollama**: For local AI model support

## Environment Setup

### 1. Basic Configuration

After installation, initialize QA Operator in your project:

```bash
cd your-project
qa-operator init
```

This creates:
- Required directories (`e2e/`, `artifacts/`, `logs/`, `policies/`)
- Example configuration files
- Basic selector policy

### 2. Environment Variables

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Required for OpenAI models
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Customize other settings
QA_OPERATOR_LOG_LEVEL=INFO
QA_OPERATOR_MODEL_PROVIDER=mixed
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. MCP Server Setup

#### Playwright MCP Server

```bash
# Install Playwright MCP server globally
npm install -g @playwright/mcp

# Or install locally in your project
npm install @playwright/mcp
```

#### Filesystem MCP Server

The filesystem MCP server is typically included with the project. If you need to set it up separately:

```bash
# Create a simple filesystem MCP server
mkdir -p servers
cat > servers/fs-mcp.js << 'EOF'
// Simple filesystem MCP server
const fs = require('fs');
const path = require('path');

// MCP server implementation
// (This would be a full MCP server implementation)
console.log('Filesystem MCP server started');
EOF
```

#### Git MCP Server (Optional)

```bash
# Install uv and uvx
pip install uv

# The git MCP server will be automatically available via uvx
```

### 4. Ollama Setup (Optional)

For local AI model support:

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model (e.g., Qwen)
ollama pull qwen:7b
```

## Verification

### Health Check

Run a comprehensive health check:

```bash
qa-operator health
```

This checks:
- Configuration validity
- System requirements
- Directory permissions
- MCP server configuration
- AI model setup

### Configuration Validation

Validate your configuration:

```bash
qa-operator config validate
```

### Test Run

Try a basic workflow:

```bash
qa-operator run --verbose
```

## Configuration

### Configuration Files

QA Operator uses several configuration files:

1. **`.env`**: Environment variables
2. **`qa-operator.config.json`**: Main configuration
3. **`orchestrator/mcp.config.json`**: MCP server configuration
4. **`policies/selector.md`**: Selector policies

### Configuration Templates

Use pre-configured templates for different environments:

```bash
# List available templates
qa-operator config templates list

# Apply development template
qa-operator config templates apply --template development

# Apply CI template
qa-operator config templates apply --template ci
```

### Environment-Specific Setup

#### Development Environment

```bash
# Apply development template
qa-operator config templates apply --template development

# Set environment variables
export QA_OPERATOR_LOG_LEVEL=DEBUG
export QA_OPERATOR_HEADLESS=false
```

#### CI/CD Environment

```bash
# Set CI environment
export CI=true
export OPENAI_API_KEY="your-ci-api-key"

# Apply CI template
qa-operator config templates apply --template ci
```

#### Production Environment

```bash
# Apply production template
qa-operator config templates apply --template production

# Set required environment variables
export OPENAI_API_KEY="your-production-api-key"
export QA_OPERATOR_LOG_LEVEL=WARN
```

## Troubleshooting

### Common Installation Issues

#### Python Version Issues

```bash
# Check Python version
python --version

# If using older Python, install newer version
# On Ubuntu/Debian:
sudo apt update
sudo apt install python3.10 python3.10-venv

# On macOS with Homebrew:
brew install python@3.10
```

#### Node.js Issues

```bash
# Check Node.js version
node --version

# Install Node.js if missing
# On Ubuntu/Debian:
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# On macOS with Homebrew:
brew install node
```

#### Permission Issues

```bash
# Fix directory permissions
chmod 755 artifacts logs e2e policies

# Fix file permissions
chmod 644 .env qa-operator.config.json
```

### Configuration Issues

#### OpenAI API Key Issues

```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

#### MCP Server Issues

```bash
# Test Playwright MCP server
npx @playwright/mcp --version

# Test filesystem access
ls -la e2e/

# Check MCP configuration
qa-operator config show
```

#### Ollama Issues

```bash
# Check Ollama status
ollama list

# Test Ollama connection
curl http://localhost:11434/api/version
```

### Runtime Issues

#### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

#### MCP Connection Errors

```bash
# Check MCP server processes
ps aux | grep mcp

# Restart MCP servers
# (This depends on your MCP server setup)
```

## Advanced Setup

### Custom MCP Servers

Create custom MCP server configurations:

```json
{
  "mcpServers": {
    "custom-server": {
      "command": "python",
      "args": ["path/to/custom_mcp_server.py"],
      "env": {
        "CUSTOM_ENV_VAR": "value"
      }
    }
  }
}
```

### Docker Setup

Create a Dockerfile for containerized deployment:

```dockerfile
FROM python:3.10-slim

# Install Node.js
RUN apt-get update && apt-get install -y nodejs npm curl

# Install QA Operator
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install -e .

# Install MCP servers
RUN npm install -g @playwright/mcp

# Set up environment
ENV QA_OPERATOR_LOG_LEVEL=INFO
ENV QA_OPERATOR_HEADLESS=true

ENTRYPOINT ["qa-operator"]
```

### CI/CD Integration

#### GitHub Actions

```yaml
name: QA Operator Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          npm install -g @playwright/mcp
      
      - name: Run QA Operator
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          CI: true
        run: |
          qa-operator health
          qa-operator run
```

#### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    environment {
        OPENAI_API_KEY = credentials('openai-api-key')
        CI = 'true'
    }
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'npm install -g @playwright/mcp'
            }
        }
        
        stage('Health Check') {
            steps {
                sh 'qa-operator health'
            }
        }
        
        stage('Run Tests') {
            steps {
                sh 'qa-operator run'
            }
        }
    }
}
```

## Getting Help

### Documentation

- **Configuration Guide**: `config/README.md`
- **API Documentation**: Generated with `qa-operator config docs`
- **Examples**: Check `config/examples/` directory

### Command Line Help

```bash
# General help
qa-operator --help

# Command-specific help
qa-operator run --help
qa-operator config --help

# Configuration documentation
qa-operator config docs
```

### Health Diagnostics

```bash
# Comprehensive health check
qa-operator health

# Configuration validation
qa-operator config validate

# Show current configuration
qa-operator config show
```

### Support

- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join community discussions
- **Documentation**: Check the project wiki

## Next Steps

After successful installation:

1. **Configure your environment**: Set up API keys and preferences
2. **Create test specifications**: Define your testing requirements
3. **Run your first workflow**: Execute `qa-operator run`
4. **Explore features**: Try different commands and options
5. **Integrate with CI/CD**: Set up automated testing pipelines