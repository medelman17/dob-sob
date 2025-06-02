#!/bin/bash

# =============================================================================
# NYC DOB Data Exploration - Stop Services
# =============================================================================

set -e

echo "🛑 Stopping NYC DOB Data Exploration services..."

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Please run from project root."
    exit 1
fi

# Stop services
echo "🐳 Stopping Docker services..."
docker-compose down

echo "✅ All services stopped."
echo ""
echo "💡 To start services again, run: ./scripts/start.sh"
echo "🗑️  To remove all data and reset, run: ./scripts/reset-db.sh"
echo "" 