#!/bin/bash
#
# QBITEL - Local Development Startup Script
# Usage: ./scripts/start-local.sh [--docker|--native] [--skip-deps]
#
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default settings
MODE="docker"
SKIP_DEPS=false
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            MODE="docker"
            shift
            ;;
        --native)
            MODE="native"
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --help)
            echo "Usage: ./scripts/start-local.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --docker     Run with Docker Compose (default)"
            echo "  --native     Run natively with Python"
            echo "  --skip-deps  Skip starting dependencies (PostgreSQL, Redis)"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  QBITEL - Local Development Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Mode: ${YELLOW}$MODE${NC}"
echo -e "Project Root: $PROJECT_ROOT"
echo ""

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=30
    local attempt=1

    echo -n "Waiting for $service..."
    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ $attempt -ge $max_attempts ]; then
            echo -e " ${RED}FAILED${NC}"
            echo -e "${RED}$service did not start within expected time${NC}"
            return 1
        fi
        echo -n "."
        sleep 1
        ((attempt++))
    done
    echo -e " ${GREEN}OK${NC}"
}

if [ "$MODE" == "docker" ]; then
    echo -e "${YELLOW}Starting with Docker Compose...${NC}"
    echo ""

    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Docker is not running. Please start Docker Desktop.${NC}"
        exit 1
    fi

    # Start services
    docker-compose -f ops/deploy/docker/docker-compose.yml up -d

    echo ""
    echo -e "${GREEN}Services started!${NC}"
    echo ""

    # Wait for services
    wait_for_service localhost 5432 "PostgreSQL"
    wait_for_service localhost 6379 "Redis"
    wait_for_service localhost 8000 "AI Engine"

    echo ""
    echo -e "${GREEN}All services are up!${NC}"
    echo ""
    echo "Access points:"
    echo "  - API:        http://localhost:8000"
    echo "  - Health:     http://localhost:8000/health"
    echo "  - Swagger:    http://localhost:8000/docs"
    echo "  - ReDoc:      http://localhost:8000/redoc"
    echo ""
    echo "View logs:"
    echo "  docker-compose -f ops/deploy/docker/docker-compose.yml logs -f"
    echo ""
    echo "Stop services:"
    echo "  docker-compose -f ops/deploy/docker/docker-compose.yml down"

else
    echo -e "${YELLOW}Starting natively with Python...${NC}"
    echo ""

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 is not installed${NC}"
        exit 1
    fi

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Install dependencies if needed
    if [ ! -f "venv/.deps_installed" ]; then
        echo "Installing dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
        touch venv/.deps_installed
    fi

    # Start dependencies unless skipped
    if [ "$SKIP_DEPS" = false ]; then
        echo ""
        echo -e "${YELLOW}Starting dependencies...${NC}"

        # Start PostgreSQL with Docker
        if ! docker ps | grep -q qbitel-postgres; then
            docker run -d \
                --name qbitel-postgres \
                -e POSTGRES_USER=qbitel \
                -e POSTGRES_PASSWORD=qbitel_dev_password \
                -e POSTGRES_DB=qbitel_dev \
                -p 5432:5432 \
                postgres:15 2>/dev/null || true
        fi

        # Start Redis with Docker
        if ! docker ps | grep -q qbitel-redis; then
            docker run -d \
                --name qbitel-redis \
                -p 6379:6379 \
                redis:7 2>/dev/null || true
        fi

        wait_for_service localhost 5432 "PostgreSQL"
        wait_for_service localhost 6379 "Redis"
    fi

    # Set environment variables
    export QBITEL_ENV=development
    export DATABASE_HOST=localhost
    export DATABASE_PORT=5432
    export DATABASE_NAME=qbitel_dev
    export DATABASE_USER=qbitel
    export DATABASE_PASSWORD=qbitel_dev_password
    export REDIS_HOST=localhost
    export REDIS_PORT=6379
    export JWT_SECRET=development-jwt-secret-key-32ch
    export ENCRYPTION_KEY=development-encryption-key-32c
    export LOG_LEVEL=DEBUG
    export TLS_ENABLED=false

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        export QBITEL_LLM_PROVIDER=ollama
        export QBITEL_LLM_ENDPOINT=http://localhost:11434
        echo -e "${GREEN}Ollama detected - using local LLM${NC}"
    else
        echo -e "${YELLOW}Ollama not running - LLM features may be limited${NC}"
    fi

    # Run migrations
    echo ""
    echo "Applying database migrations..."
    cd ai_engine
    alembic upgrade head 2>/dev/null || echo -e "${YELLOW}Migrations skipped (may already be applied)${NC}"
    cd ..

    echo ""
    echo -e "${GREEN}Starting QBITEL Engine...${NC}"
    echo ""
    echo "Access points:"
    echo "  - API:        http://localhost:8000"
    echo "  - Health:     http://localhost:8000/health"
    echo "  - Swagger:    http://localhost:8000/docs"
    echo ""
    echo "Press Ctrl+C to stop"
    echo ""

    # Start the application
    python -m ai_engine --mode development
fi
