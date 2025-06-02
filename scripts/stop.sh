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

# Stop and remove containers
echo "🐳 Stopping Docker services..."
docker compose down

echo "✅ All services stopped!"
echo ""
echo "💡 Useful commands:"
echo "   Start services: ./scripts/start.sh"
echo "   View logs:      docker compose logs"
echo "   Reset data:     ./scripts/reset-db.sh"
echo "" 