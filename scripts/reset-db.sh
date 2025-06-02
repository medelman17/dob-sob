#!/bin/bash

# =============================================================================
# NYC DOB Data Exploration - Reset Database
# =============================================================================

set -e

echo "🗑️  Resetting NYC DOB Data Exploration database..."
echo "⚠️  WARNING: This will delete all Neo4j data!"
echo ""

# Prompt for confirmation
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Operation cancelled."
    exit 1
fi

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Please run from project root."
    exit 1
fi

# Stop services
echo "🛑 Stopping services..."
docker compose down

# Remove Docker managed volumes (this deletes all Neo4j data)
echo "🗑️  Removing Neo4j data volumes..."
docker volume rm nycdob-data-exploration_neo4j_data 2>/dev/null || echo "   Volume nycdob-data-exploration_neo4j_data not found"
docker volume rm nycdob-data-exploration_neo4j_logs 2>/dev/null || echo "   Volume nycdob-data-exploration_neo4j_logs not found"  
docker volume rm nycdob-data-exploration_neo4j_import 2>/dev/null || echo "   Volume nycdob-data-exploration_neo4j_import not found"
docker volume rm nycdob-data-exploration_neo4j_plugins 2>/dev/null || echo "   Volume nycdob-data-exploration_neo4j_plugins not found"

# Remove orphaned containers
echo "🧹 Cleaning up..."
docker compose down --remove-orphans

echo ""
echo "✅ Database reset complete!"
echo ""
echo "💡 To start fresh:"
echo "   ./scripts/start.sh"
echo ""
echo "📋 Note: Docker managed volumes have been removed."
echo "   Neo4j will initialize with a clean database on next startup." 