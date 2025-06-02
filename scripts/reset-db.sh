#!/bin/bash

# =============================================================================
# NYC DOB Data Exploration - Reset Database
# =============================================================================

set -e

echo "🗑️  Resetting NYC DOB Data Exploration database..."

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Please run from project root."
    exit 1
fi

# Confirm with user
echo "⚠️  WARNING: This will permanently delete ALL data in the Neo4j database!"
echo "   - All nodes and relationships will be lost"
echo "   - All imported data will be removed"
echo "   - Neo4j logs will be cleared"
echo ""
read -p "Are you sure you want to continue? (type 'yes' to confirm): " confirmation

if [ "$confirmation" != "yes" ]; then
    echo "❌ Operation cancelled."
    exit 0
fi

echo ""
echo "🛑 Stopping services..."
docker-compose down

echo "🗑️  Removing Neo4j data volumes..."
docker-compose down -v

echo "📁 Removing local data directories..."
if [ -d "docker-volumes/neo4j" ]; then
    echo "   Removing docker-volumes/neo4j/..."
    rm -rf docker-volumes/neo4j/*
fi

echo "🏗️  Recreating directory structure..."
mkdir -p docker-volumes/neo4j/{data,logs,import,plugins}
mkdir -p data notebooks

echo "✅ Database reset complete!"
echo ""
echo "🚀 To start with a fresh database, run: ./scripts/start.sh"
echo "" 