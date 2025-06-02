#!/bin/bash

# =============================================================================
# NYC DOB Data Exploration - Start Services
# =============================================================================

set -e

echo "🚀 Starting NYC DOB Data Exploration services..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Please run from project root."
    exit 1
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p docker-volumes/neo4j/{data,logs,import,plugins}
mkdir -p data notebooks

# Set proper permissions for Neo4j directories
echo "🔐 Setting permissions for Neo4j directories..."
sudo chown -R 7474:7474 docker-volumes/neo4j/ 2>/dev/null || {
    echo "⚠️  Warning: Could not set Neo4j directory ownership. You may need to run:"
    echo "   sudo chown -R 7474:7474 docker-volumes/neo4j/"
}

# Check if .env file exists, if not suggest copying from example
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found."
    if [ -f "docker.env.example" ]; then
        echo "📋 Found docker.env.example. Copying to .env..."
        cp docker.env.example .env
        echo "✅ Created .env file from docker.env.example"
        echo "💡 Please review and modify .env file as needed, especially passwords!"
    else
        echo "💡 Please create a .env file or copy from docker.env.example"
    fi
fi

# Start services
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for Neo4j to be ready
echo "⏳ Waiting for Neo4j to be ready..."
timeout=120
counter=0
while [ $counter -lt $timeout ]; do
    if docker-compose exec -T neo4j cypher-shell -u neo4j -p "${NEO4J_PASSWORD:-password}" "RETURN 1" >/dev/null 2>&1; then
        echo "✅ Neo4j is ready!"
        break
    fi
    sleep 2
    counter=$((counter + 2))
    echo -n "."
done

if [ $counter -ge $timeout ]; then
    echo "❌ Timeout waiting for Neo4j to start"
    echo "🔍 Check logs with: docker-compose logs neo4j"
    exit 1
fi

echo ""
echo "🎉 All services are running!"
echo ""
echo "📱 Access your applications:"
echo "   🌐 Streamlit Dashboard: http://localhost:${STREAMLIT_PORT:-8501}"
echo "   🔍 Neo4j Browser:       http://localhost:${NEO4J_HTTP_PORT:-7474}"
echo "   📊 Jupyter Lab:         http://localhost:${JUPYTER_PORT:-8889}"
echo ""
echo "🔑 Neo4j Credentials:"
echo "   Username: ${NEO4J_USER:-neo4j}"
echo "   Password: ${NEO4J_PASSWORD:-password}"
echo ""
echo "🛠️  Useful commands:"
echo "   View logs:     docker-compose logs -f"
echo "   Stop services: ./scripts/stop.sh"
echo "   Reset data:    ./scripts/reset-db.sh"
echo "" 