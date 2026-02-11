#!/bin/bash
#
# QBITEL - macOS Deployment Script
#
# Comprehensive deployment script for running QBITEL on macOS
# Supports Apple Silicon (M1/M2/M3) and Intel Macs
#
# Usage:
#   ./scripts/deploy-macos.sh [command] [options]
#
# Commands:
#   install     Install all prerequisites
#   setup       Initial setup (create env, install deps, run migrations)
#   start       Start all services
#   stop        Stop all services
#   restart     Restart all services
#   status      Show service status
#   logs        View logs
#   health      Check service health
#   clean       Clean up (remove containers, volumes)
#   uninstall   Complete uninstall
#
# Options:
#   --quick     Quick start with Docker Compose only
#   --full      Full stack with monitoring (Prometheus, Grafana)
#   --native    Run Python natively (Docker for deps only)
#   --dev       Development mode with hot reload
#   --skip-prereq  Skip prerequisites check
#   --force     Force reinstall/recreate
#   --help      Show this help message
#

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
ENV_FILE="$PROJECT_ROOT/.env"
CONFIG_FILE="$PROJECT_ROOT/config/qbitel.yaml"

# Docker compose file
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/ops/deploy/docker/docker-compose.yml"
DOCKER_COMPOSE_AI="$PROJECT_ROOT/ai_engine/deployment/docker/docker-compose.yml"

# Default settings
MODE="docker"
DEV_MODE=false
FULL_STACK=false
SKIP_PREREQ=false
FORCE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# =============================================================================
# Helper Functions
# =============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                   ║"
    echo "║     ██████╗██████╗  ██████╗ ███╗   ██╗ ██████╗ ███████╗          ║"
    echo "║    ██╔════╝██╔══██╗██╔═══██╗████╗  ██║██╔═══██╗██╔════╝          ║"
    echo "║    ██║     ██████╔╝██║   ██║██╔██╗ ██║██║   ██║███████╗          ║"
    echo "║    ██║     ██╔══██╗██║   ██║██║╚██╗██║██║   ██║╚════██║          ║"
    echo "║    ╚██████╗██║  ██║╚██████╔╝██║ ╚████║╚██████╔╝███████║          ║"
    echo "║     ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚══════╝          ║"
    echo "║                                                                   ║"
    echo "║              AI-Powered Protocol Security Platform                ║"
    echo "║                    macOS Deployment Script                        ║"
    echo "║                                                                   ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${BOLD}${CYAN}==> $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=${4:-30}
    local attempt=1

    echo -n "Waiting for $service"
    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ $attempt -ge $max_attempts ]; then
            echo -e " ${RED}FAILED${NC}"
            return 1
        fi
        echo -n "."
        sleep 1
        ((attempt++))
    done
    echo -e " ${GREEN}OK${NC}"
    return 0
}

# Detect macOS architecture
detect_architecture() {
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        echo "apple_silicon"
    else
        echo "intel"
    fi
}

# Get macOS version
get_macos_version() {
    sw_vers -productVersion
}

# =============================================================================
# Prerequisites Installation
# =============================================================================

install_homebrew() {
    if command_exists brew; then
        log_success "Homebrew already installed"
        return 0
    fi

    log_info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add to PATH for Apple Silicon
    if [ "$(detect_architecture)" = "apple_silicon" ]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi

    log_success "Homebrew installed"
}

install_python() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ $(echo "$PYTHON_VERSION >= 3.9" | bc -l) -eq 1 ]]; then
            log_success "Python $PYTHON_VERSION already installed"
            return 0
        fi
    fi

    log_info "Installing Python 3.11..."
    brew install python@3.11

    # Link python3 to python3.11
    brew link python@3.11 --force 2>/dev/null || true

    log_success "Python 3.11 installed"
}

install_docker() {
    if command_exists docker; then
        if docker info &>/dev/null; then
            log_success "Docker Desktop already installed and running"
            return 0
        else
            log_warning "Docker installed but not running"
            log_info "Please start Docker Desktop and run this script again"
            open -a Docker
            exit 1
        fi
    fi

    log_info "Installing Docker Desktop..."

    ARCH=$(detect_architecture)
    if [ "$ARCH" = "apple_silicon" ]; then
        DOCKER_DMG="Docker-arm64.dmg"
    else
        DOCKER_DMG="Docker.dmg"
    fi

    # Download Docker Desktop
    curl -fsSL "https://desktop.docker.com/mac/main/${ARCH}/Docker.dmg" -o "/tmp/$DOCKER_DMG"

    # Mount and install
    hdiutil attach "/tmp/$DOCKER_DMG" -nobrowse -quiet
    cp -R "/Volumes/Docker/Docker.app" /Applications/
    hdiutil detach "/Volumes/Docker" -quiet
    rm "/tmp/$DOCKER_DMG"

    log_info "Starting Docker Desktop..."
    open -a Docker

    # Wait for Docker to start
    log_info "Waiting for Docker to initialize (this may take a minute)..."
    local attempts=0
    while ! docker info &>/dev/null; do
        sleep 2
        ((attempts++))
        if [ $attempts -ge 60 ]; then
            log_error "Docker failed to start. Please start it manually and re-run this script."
            exit 1
        fi
    done

    log_success "Docker Desktop installed and running"
}

install_postgresql_client() {
    if command_exists psql; then
        log_success "PostgreSQL client already installed"
        return 0
    fi

    log_info "Installing PostgreSQL client..."
    brew install libpq
    brew link libpq --force
    log_success "PostgreSQL client installed"
}

install_redis_cli() {
    if command_exists redis-cli; then
        log_success "Redis CLI already installed"
        return 0
    fi

    log_info "Installing Redis CLI..."
    brew install redis
    log_success "Redis CLI installed"
}

install_ollama() {
    if command_exists ollama; then
        log_success "Ollama already installed"
        return 0
    fi

    log_info "Installing Ollama for local LLM support..."
    brew install ollama
    log_success "Ollama installed"
}

install_additional_tools() {
    # Install jq for JSON parsing
    if ! command_exists jq; then
        log_info "Installing jq..."
        brew install jq
    fi

    # Install netcat for health checks
    if ! command_exists nc; then
        log_info "Installing netcat..."
        brew install netcat
    fi
}

install_prerequisites() {
    log_step "Checking and installing prerequisites..."

    # Detect system
    ARCH=$(detect_architecture)
    MACOS_VERSION=$(get_macos_version)
    log_info "Detected: macOS $MACOS_VERSION on $ARCH"

    install_homebrew
    install_python
    install_docker
    install_postgresql_client
    install_redis_cli
    install_additional_tools

    echo ""
    read -p "Would you like to install Ollama for local LLM support? (y/N): " install_ollama_choice
    if [[ "$install_ollama_choice" =~ ^[Yy]$ ]]; then
        install_ollama
    fi

    log_success "All prerequisites installed!"
}

# =============================================================================
# Environment Setup
# =============================================================================

setup_virtualenv() {
    log_step "Setting up Python virtual environment..."

    if [ -d "$VENV_DIR" ] && [ "$FORCE" = false ]; then
        log_success "Virtual environment already exists"
    else
        if [ -d "$VENV_DIR" ]; then
            rm -rf "$VENV_DIR"
        fi
        python3 -m venv "$VENV_DIR"
        log_success "Virtual environment created"
    fi

    # Activate
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    pip install --upgrade pip wheel setuptools
}

install_dependencies() {
    log_step "Installing Python dependencies..."

    source "$VENV_DIR/bin/activate"

    # Install main requirements
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements.txt"
    fi

    # Install dev requirements if in dev mode
    if [ "$DEV_MODE" = true ] && [ -f "$PROJECT_ROOT/requirements-dev.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements-dev.txt"
    fi

    log_success "Dependencies installed"
}

setup_environment_file() {
    log_step "Setting up environment configuration..."

    if [ -f "$ENV_FILE" ] && [ "$FORCE" = false ]; then
        log_success "Environment file already exists"
        return 0
    fi

    # Copy from example
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
    else
        # Create minimal .env file
        cat > "$ENV_FILE" << 'EOF'
# QBITEL - Environment Configuration
# Generated by deploy-macos.sh

# Environment
QBITEL_ENV=development
DEBUG=true

# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=qbitel_dev
DATABASE_USER=qbitel
DATABASE_PASSWORD=qbitel_dev_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Security (Change in production!)
JWT_SECRET=macos-dev-jwt-secret-key-32chars
ENCRYPTION_KEY=macos-dev-encryption-key-32char

# API
API_HOST=0.0.0.0
API_PORT=8000

# LLM
QBITEL_LLM_PROVIDER=ollama
QBITEL_LLM_ENDPOINT=http://localhost:11434
QBITEL_LLM_MODEL=llama3.2:8b

# Logging
LOG_LEVEL=INFO
TLS_ENABLED=false
EOF
    fi

    log_success "Environment file created at $ENV_FILE"
    log_warning "Review and update $ENV_FILE with your settings"
}

run_migrations() {
    log_step "Running database migrations..."

    source "$VENV_DIR/bin/activate"
    cd "$PROJECT_ROOT/ai_engine"

    # Wait for database
    wait_for_service localhost 5432 "PostgreSQL" 30 || {
        log_error "PostgreSQL is not running. Start it first with: ./scripts/deploy-macos.sh start"
        return 1
    }

    # Run migrations
    alembic upgrade head 2>/dev/null || {
        log_warning "Migrations may have already been applied or no migrations exist"
    }

    cd "$PROJECT_ROOT"
    log_success "Migrations completed"
}

create_admin_user() {
    log_step "Creating admin user..."

    source "$VENV_DIR/bin/activate"

    # Check if admin creation script exists
    if [ -f "$PROJECT_ROOT/ai_engine/scripts/create_admin.py" ]; then
        python "$PROJECT_ROOT/ai_engine/scripts/create_admin.py" \
            --username admin \
            --email admin@qbitel.local \
            --password admin123 2>/dev/null || {
            log_warning "Admin user may already exist"
        }
    else
        log_warning "Admin creation script not found. Create admin user manually."
    fi
}

setup_ollama_models() {
    if ! command_exists ollama; then
        return 0
    fi

    log_step "Setting up Ollama models..."

    # Check if Ollama is running
    if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
        log_info "Starting Ollama..."
        ollama serve &>/dev/null &
        sleep 3
    fi

    # Pull recommended model
    log_info "Pulling llama3.2:8b model (this may take a while)..."
    ollama pull llama3.2:8b || {
        log_warning "Failed to pull model. You can do this later with: ollama pull llama3.2:8b"
    }

    log_success "Ollama models ready"
}

# =============================================================================
# Service Management
# =============================================================================

start_docker_services() {
    log_step "Starting Docker services..."

    cd "$PROJECT_ROOT"

    if [ "$FULL_STACK" = true ]; then
        # Start full stack with monitoring
        if [ -f "$DOCKER_COMPOSE_AI" ]; then
            docker-compose -f "$DOCKER_COMPOSE_AI" up -d
        else
            docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
        fi
    else
        # Start minimal services
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d postgres redis
    fi

    # Wait for services
    wait_for_service localhost 5432 "PostgreSQL" 30
    wait_for_service localhost 6379 "Redis" 30

    log_success "Docker services started"
}

start_native_services() {
    log_step "Starting services in native mode..."

    # Start only dependencies in Docker
    cd "$PROJECT_ROOT"

    # Start PostgreSQL
    if ! docker ps | grep -q qbitel-postgres; then
        log_info "Starting PostgreSQL container..."
        docker run -d \
            --name qbitel-postgres \
            -e POSTGRES_USER=qbitel \
            -e POSTGRES_PASSWORD=qbitel_dev_password \
            -e POSTGRES_DB=qbitel_dev \
            -p 5432:5432 \
            -v qbitel-postgres-data:/var/lib/postgresql/data \
            postgres:15-alpine 2>/dev/null || docker start qbitel-postgres
    fi

    # Start Redis
    if ! docker ps | grep -q qbitel-redis; then
        log_info "Starting Redis container..."
        docker run -d \
            --name qbitel-redis \
            -p 6379:6379 \
            -v qbitel-redis-data:/data \
            redis:7-alpine 2>/dev/null || docker start qbitel-redis
    fi

    # Wait for services
    wait_for_service localhost 5432 "PostgreSQL" 30
    wait_for_service localhost 6379 "Redis" 30

    log_success "Database services started"
}

start_application() {
    log_step "Starting QBITEL application..."

    source "$VENV_DIR/bin/activate"

    # Load environment
    if [ -f "$ENV_FILE" ]; then
        export $(grep -v '^#' "$ENV_FILE" | xargs)
    fi

    cd "$PROJECT_ROOT"

    if [ "$DEV_MODE" = true ]; then
        log_info "Starting in development mode with hot reload..."
        python -m ai_engine --mode development --reload &
    else
        log_info "Starting in production mode..."
        python -m ai_engine --mode production &
    fi

    APP_PID=$!
    echo $APP_PID > "$PROJECT_ROOT/.qbitel.pid"

    # Wait for application
    sleep 3
    wait_for_service localhost 8000 "QBITEL" 30 || {
        log_error "Application failed to start"
        return 1
    }

    log_success "QBITEL started (PID: $APP_PID)"
}

stop_services() {
    log_step "Stopping services..."

    # Stop application
    if [ -f "$PROJECT_ROOT/.qbitel.pid" ]; then
        PID=$(cat "$PROJECT_ROOT/.qbitel.pid")
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID 2>/dev/null || true
            log_success "Application stopped"
        fi
        rm -f "$PROJECT_ROOT/.qbitel.pid"
    fi

    # Stop Docker services
    if [ "$MODE" = "docker" ]; then
        cd "$PROJECT_ROOT"
        docker-compose -f "$DOCKER_COMPOSE_FILE" down 2>/dev/null || true
    else
        docker stop qbitel-postgres qbitel-redis 2>/dev/null || true
    fi

    # Stop Ollama
    pkill -f "ollama serve" 2>/dev/null || true

    log_success "All services stopped"
}

show_status() {
    log_step "Service Status"

    echo ""
    echo -e "${BOLD}Docker Containers:${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "qbitel|postgres|redis" || echo "  No containers running"

    echo ""
    echo -e "${BOLD}Application:${NC}"
    if [ -f "$PROJECT_ROOT/.qbitel.pid" ]; then
        PID=$(cat "$PROJECT_ROOT/.qbitel.pid")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "  QBITEL: ${GREEN}Running${NC} (PID: $PID)"
        else
            echo -e "  QBITEL: ${RED}Not Running${NC}"
        fi
    else
        echo -e "  QBITEL: ${RED}Not Running${NC}"
    fi

    echo ""
    echo -e "${BOLD}Ollama:${NC}"
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo -e "  Ollama: ${GREEN}Running${NC}"
    else
        echo -e "  Ollama: ${YELLOW}Not Running${NC}"
    fi

    echo ""
    echo -e "${BOLD}Endpoints:${NC}"
    echo "  API:        http://localhost:8000"
    echo "  Swagger:    http://localhost:8000/docs"
    echo "  ReDoc:      http://localhost:8000/redoc"
    echo "  Health:     http://localhost:8000/health"
    echo "  Metrics:    http://localhost:8000/metrics"

    if [ "$FULL_STACK" = true ]; then
        echo "  Grafana:    http://localhost:3000"
        echo "  Prometheus: http://localhost:9090"
    fi
}

check_health() {
    log_step "Health Check"

    echo ""

    # PostgreSQL
    echo -n "PostgreSQL: "
    if pg_isready -h localhost -p 5432 &>/dev/null; then
        echo -e "${GREEN}Healthy${NC}"
    else
        echo -e "${RED}Unhealthy${NC}"
    fi

    # Redis
    echo -n "Redis: "
    if redis-cli -h localhost -p 6379 ping &>/dev/null; then
        echo -e "${GREEN}Healthy${NC}"
    else
        echo -e "${RED}Unhealthy${NC}"
    fi

    # API
    echo -n "API: "
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "${GREEN}Healthy${NC}"

        # Show detailed health
        echo ""
        echo "API Health Response:"
        curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health
    else
        echo -e "${RED}Unhealthy (HTTP $HTTP_CODE)${NC}"
    fi

    # Ollama
    echo ""
    echo -n "Ollama: "
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo -e "${GREEN}Healthy${NC}"
    else
        echo -e "${YELLOW}Not Running${NC}"
    fi
}

show_logs() {
    log_step "Viewing logs..."

    if [ "$MODE" = "docker" ]; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f --tail=100
    else
        echo "Application logs:"
        if [ -f "/var/log/qbitel/app.log" ]; then
            tail -f /var/log/qbitel/app.log
        else
            log_warning "Log file not found. Check application output."
        fi
    fi
}

clean_up() {
    log_step "Cleaning up..."

    read -p "This will remove all containers and volumes. Continue? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        log_info "Cancelled"
        return 0
    fi

    # Stop services
    stop_services

    # Remove containers
    docker rm -f qbitel-postgres qbitel-redis 2>/dev/null || true

    # Remove volumes
    docker volume rm qbitel-postgres-data qbitel-redis-data 2>/dev/null || true

    # Remove docker-compose resources
    cd "$PROJECT_ROOT"
    docker-compose -f "$DOCKER_COMPOSE_FILE" down -v 2>/dev/null || true

    # Remove virtual environment
    read -p "Remove Python virtual environment? (y/N): " remove_venv
    if [[ "$remove_venv" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    fi

    # Remove .env
    read -p "Remove .env file? (y/N): " remove_env
    if [[ "$remove_env" =~ ^[Yy]$ ]]; then
        rm -f "$ENV_FILE"
    fi

    log_success "Cleanup completed"
}

# =============================================================================
# Main Commands
# =============================================================================

do_install() {
    print_banner
    install_prerequisites

    echo ""
    log_success "Installation complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./scripts/deploy-macos.sh setup"
    echo "  2. Run: ./scripts/deploy-macos.sh start"
}

do_setup() {
    print_banner

    if [ "$SKIP_PREREQ" = false ]; then
        install_prerequisites
    fi

    setup_virtualenv
    install_dependencies
    setup_environment_file

    # Start services temporarily for migrations
    start_native_services

    run_migrations
    create_admin_user

    if command_exists ollama; then
        setup_ollama_models
    fi

    echo ""
    log_success "Setup complete!"
    echo ""
    echo "Your environment is ready. Start the application with:"
    echo "  ./scripts/deploy-macos.sh start"
}

do_start() {
    print_banner

    if [ "$MODE" = "docker" ]; then
        start_docker_services

        if [ "$FULL_STACK" = false ]; then
            # Also start the application
            start_application
        fi
    else
        start_native_services
        start_application
    fi

    # Start Ollama if available
    if command_exists ollama; then
        if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
            log_info "Starting Ollama..."
            ollama serve &>/dev/null &
        fi
    fi

    echo ""
    show_status

    echo ""
    log_success "QBITEL is running!"
    echo ""
    echo "Quick test:"
    echo "  curl http://localhost:8000/health"
    echo ""
    echo "Access Swagger UI:"
    echo "  open http://localhost:8000/docs"
}

do_restart() {
    stop_services
    sleep 2
    do_start
}

show_help() {
    echo "QBITEL - macOS Deployment Script"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  install     Install all prerequisites (Homebrew, Python, Docker, etc.)"
    echo "  setup       Initial setup (virtualenv, dependencies, migrations)"
    echo "  start       Start all services"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show service status"
    echo "  logs        View logs"
    echo "  health      Check service health"
    echo "  clean       Clean up containers and volumes"
    echo ""
    echo "Options:"
    echo "  --quick       Quick start with Docker Compose only"
    echo "  --full        Full stack with monitoring (Prometheus, Grafana)"
    echo "  --native      Run Python natively (Docker for deps only)"
    echo "  --dev         Development mode with hot reload"
    echo "  --skip-prereq Skip prerequisites check"
    echo "  --force       Force reinstall/recreate"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 install                    # Install prerequisites"
    echo "  $0 setup                      # Setup environment"
    echo "  $0 start                      # Start with Docker"
    echo "  $0 start --native --dev       # Start native with dev mode"
    echo "  $0 start --full               # Start with monitoring stack"
    echo "  $0 status                     # Check status"
    echo "  $0 health                     # Health check"
    echo ""
}

# =============================================================================
# Main Entry Point
# =============================================================================

main() {
    # Parse options
    COMMAND=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            install|setup|start|stop|restart|status|logs|health|clean|uninstall)
                COMMAND=$1
                shift
                ;;
            --quick)
                MODE="docker"
                shift
                ;;
            --full)
                MODE="docker"
                FULL_STACK=true
                shift
                ;;
            --native)
                MODE="native"
                shift
                ;;
            --dev)
                DEV_MODE=true
                shift
                ;;
            --skip-prereq)
                SKIP_PREREQ=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Default command
    if [ -z "$COMMAND" ]; then
        COMMAND="start"
    fi

    # Change to project root
    cd "$PROJECT_ROOT"

    # Execute command
    case $COMMAND in
        install)
            do_install
            ;;
        setup)
            do_setup
            ;;
        start)
            do_start
            ;;
        stop)
            stop_services
            ;;
        restart)
            do_restart
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        health)
            check_health
            ;;
        clean|uninstall)
            clean_up
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
