# NYC Department of Buildings Data Exploration

A comprehensive data exploration and analysis platform for NYC Department of Buildings datasets, featuring graph database storage, automated data profiling, and advanced analytics capabilities.

## 🏗️ Project Overview

This project provides tools and infrastructure for:

- Automated data acquisition from NYC DOB public datasets
- Data quality assessment and profiling
- Graph database storage using Neo4j
- Pattern matching and anomaly detection
- Interactive dashboards and visualizations
- Machine learning-based risk scoring

## 🚀 Quick Start

### Option 1: Docker Setup (Recommended)

The easiest way to get started is using Docker Compose:

```bash
# Clone the repository
git clone <repository-url>
cd nycdob-data-exploration

# Start all services
docker-compose up -d

# Access the applications:
# - Streamlit Dashboard: http://localhost:8501
# - Neo4j Browser: http://localhost:7474
# - Jupyter Lab: http://localhost:8889
```

### Option 2: Manual Setup

#### Prerequisites

- Python 3.12 or higher
- uv package manager
- Docker and Docker Compose (for Neo4j)

#### Installation Steps

1. **Install Python 3.12+**

   ```bash
   # On macOS with Homebrew
   brew install python@3.12

   # On Ubuntu/Debian
   sudo apt update
   sudo apt install python3.12 python3.12-venv

   # On Windows
   # Download from https://python.org
   ```

2. **Install uv package manager**

   ```bash
   pip install uv
   ```

3. **Set up the project**

   ```bash
   # Clone the repository
   git clone <repository-url>
   cd nycdob-data-exploration

   # Create and activate virtual environment
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies
   uv pip install -r requirements.txt

   # Install development dependencies (optional)
   uv pip install -r requirements-dev.txt
   ```

4. **Start Neo4j database**

   ```bash
   # Using Docker Compose (recommended)
   docker-compose up neo4j -d

   # Or install Neo4j locally following their documentation
   ```

5. **Run the application**

   ```bash
   # Start Streamlit dashboard
   streamlit run main.py

   # Or start Jupyter Lab
   jupyter lab
   ```

## 📁 Project Structure

```
nycdob-data-exploration/
├── scripts/
│   ├── data_profiling/     # Data quality and profiling tools
│   ├── etl/               # Extract, Transform, Load pipelines
│   └── analysis/          # Analysis and pattern matching tools
├── data/                  # Data storage directory
├── docs/                  # Documentation
├── .taskmaster/           # Task management configuration
├── requirements.txt       # Python dependencies
├── requirements-dev.txt   # Development dependencies
├── pyproject.toml        # Project configuration
├── Dockerfile            # Docker container configuration
├── docker-compose.yml    # Multi-service orchestration
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Data Sources
NYC_DOB_API_KEY=your_api_key_here  # If required

# Application Settings
STREAMLIT_SERVER_PORT=8501
JUPYTER_PORT=8888
```

### Neo4j Setup

The default configuration uses:

- **Browser Interface**: http://localhost:7474
- **Bolt Protocol**: bolt://localhost:7687
- **Username**: neo4j
- **Password**: password

## 🐳 Docker Services

The `docker-compose.yml` defines three services:

### Application Service (`app`)

- **Port**: 8501
- **Purpose**: Streamlit dashboard
- **Volumes**: Code and data directories mounted for development

### Neo4j Database (`neo4j`)

- **Ports**: 7474 (browser), 7687 (bolt)
- **Version**: Neo4j 5.15 Community Edition
- **Plugins**: APOC procedures enabled
- **Memory**: 2GB heap, 1GB page cache

### Jupyter Service (`jupyter`)

- **Port**: 8889 (mapped to container port 8888)
- **Purpose**: Interactive data analysis
- **Access**: http://localhost:8889 (no token required)

## 📊 Key Dependencies

### Core Libraries

- **pandas** (≥2.0.0): Data manipulation and analysis
- **polars** (≥0.20.0): High-performance DataFrame library
- **neo4j** (≥5.8.0): Neo4j database driver
- **networkx** (≥3.1): Graph analysis and algorithms

### Visualization

- **plotly** (≥5.15.0): Interactive plotting
- **streamlit** (≥1.28.0): Web application framework
- **matplotlib** (≥3.7.0): Static plotting
- **seaborn** (≥0.12.0): Statistical visualization

### Machine Learning

- **scikit-learn** (≥1.3.0): Machine learning algorithms
- **numpy** (≥1.24.0): Numerical computing

### Development Tools

- **pytest** (≥7.4.0): Testing framework
- **black** (≥23.7.0): Code formatting
- **isort** (≥5.12.0): Import sorting
- **mypy** (≥1.5.0): Type checking

## 🔍 Troubleshooting

### Common Issues

1. **Port conflicts**

   ```bash
   # Check what's using the ports
   lsof -i :8501  # Streamlit
   lsof -i :7474  # Neo4j browser
   lsof -i :7687  # Neo4j bolt
   ```

2. **Docker permission issues**

   ```bash
   # Add user to docker group (Linux)
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

3. **Virtual environment issues**

   ```bash
   # Remove and recreate virtual environment
   rm -rf .venv
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

4. **Neo4j connection issues**

   ```bash
   # Check Neo4j logs
   docker-compose logs neo4j

   # Restart Neo4j service
   docker-compose restart neo4j
   ```

### Performance Optimization

- **Memory**: Adjust Neo4j memory settings in `docker-compose.yml` based on available RAM
- **Storage**: Use SSD storage for better I/O performance
- **Network**: Ensure Docker has sufficient resources allocated

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
uv pip install -r requirements-dev.txt

# Run tests
pytest

# Run with coverage
pytest --cov=scripts

# Type checking
mypy scripts/

# Code formatting
black scripts/
isort scripts/
```

## 📚 Next Steps

After completing the environment setup:

1. **Data Acquisition**: Set up data download scripts
2. **Database Schema**: Design and implement Neo4j schema
3. **ETL Pipeline**: Build data transformation pipelines
4. **Analysis Tools**: Implement pattern matching and anomaly detection
5. **Dashboard**: Create interactive visualizations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

[Add your license information here]

## 📞 Support

For questions or issues:

- Create an issue in the repository
- Check the troubleshooting section above
- Review the project documentation in `/docs/`
