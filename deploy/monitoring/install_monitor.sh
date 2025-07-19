#!/usr/bin/env bash

# HA Intent Predictor - Real-time Installation Monitor
# Run this in a separate terminal to monitor installation progress

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

CTID=""
MONITOR_LOG="/tmp/ha-predictor-install-monitor.log"
PROGRESS_FILE="/tmp/ha-predictor-progress"

function usage() {
    echo "Usage: $0 <container_id>"
    echo "Monitor the installation progress of HA Intent Predictor"
    echo ""
    echo "Example:"
    echo "  $0 <container_id>  # Monitor specific container"
    exit 1
}

function header() {
    clear
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                     HA INTENT PREDICTOR - INSTALL MONITOR                   ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${CYAN}Container ID: ${CTID}${NC}"
    echo -e "${CYAN}Monitor Log: ${MONITOR_LOG}${NC}"
    echo -e "${CYAN}Started: $(date)${NC}"
    echo ""
}

function log_info() {
    local msg="$1"
    echo -e "${BLUE}[$(date '+%H:%M:%S')] INFO:${NC} $msg" | tee -a "$MONITOR_LOG"
}

function log_success() {
    local msg="$1"
    echo -e "${GREEN}[$(date '+%H:%M:%S')] SUCCESS:${NC} $msg" | tee -a "$MONITOR_LOG"
}

function log_warning() {
    local msg="$1"
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $msg" | tee -a "$MONITOR_LOG"
}

function log_error() {
    local msg="$1"
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $msg" | tee -a "$MONITOR_LOG"
}

function check_container_status() {
    if ! pct status "$CTID" &>/dev/null; then
        log_error "Container $CTID does not exist"
        return 1
    fi
    
    local status=$(pct status "$CTID" | awk '{print $2}')
    case "$status" in
        "running")
            log_success "Container is running"
            return 0
            ;;
        "stopped")
            log_warning "Container is stopped"
            return 1
            ;;
        *)
            log_warning "Container status: $status"
            return 1
            ;;
    esac
}

function monitor_system_resources() {
    echo -e "\n${PURPLE}=== SYSTEM RESOURCES ===${NC}"
    
    # Host resources
    local host_cpu=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    local host_memory=$(free -h | awk '/^Mem:/ {printf "%.1f%", ($3/$2)*100}')
    local host_disk=$(df -h | awk '$NF=="/" {printf "%s", $5}')
    
    echo -e "${CYAN}Host Resources:${NC}"
    echo -e "  CPU Usage: ${host_cpu}%"
    echo -e "  Memory Usage: ${host_memory}"
    echo -e "  Disk Usage: ${host_disk}"
    
    # Container resources (if running)
    if check_container_status >/dev/null 2>&1; then
        echo -e "\n${CYAN}Container Resources:${NC}"
        
        # Get container PID
        local ct_pid=$(pct exec "$CTID" -- sh -c 'echo $$' 2>/dev/null || echo "N/A")
        if [ "$ct_pid" != "N/A" ]; then
            # Container CPU and memory from host perspective
            local ct_memory=$(pct exec "$CTID" -- free -h 2>/dev/null | awk '/^Mem:/ {printf "%.1f%", ($3/$2)*100}' || echo "N/A")
            local ct_disk=$(pct exec "$CTID" -- df -h / 2>/dev/null | awk 'NR==2 {print $5}' || echo "N/A")
            
            echo -e "  Memory Usage: ${ct_memory}"
            echo -e "  Disk Usage: ${ct_disk}"
        else
            echo -e "  ${YELLOW}Container not accessible${NC}"
        fi
    fi
}

function monitor_docker_services() {
    echo -e "\n${PURPLE}=== DOCKER SERVICES ===${NC}"
    
    if ! check_container_status >/dev/null 2>&1; then
        echo -e "${YELLOW}Container not running - Docker services not available${NC}"
        return
    fi
    
    # Check if Docker is installed and running
    if pct exec "$CTID" -- which docker &>/dev/null; then
        if pct exec "$CTID" -- systemctl is-active docker &>/dev/null; then
            log_success "Docker daemon is running"
            
            # Check Docker Compose services
            if pct exec "$CTID" -- test -f /opt/ha-intent-predictor/docker-compose.yml; then
                echo -e "\n${CYAN}Docker Compose Services:${NC}"
                pct exec "$CTID" -- bash -c "cd /opt/ha-intent-predictor && docker compose ps" 2>/dev/null || echo -e "${YELLOW}  Services not started yet${NC}"
            else
                echo -e "${YELLOW}  Docker Compose file not found${NC}"
            fi
        else
            log_warning "Docker daemon not running"
        fi
    else
        echo -e "${YELLOW}Docker not installed yet${NC}"
    fi
}

function monitor_python_services() {
    echo -e "\n${PURPLE}=== PYTHON SERVICES ===${NC}"
    
    if ! check_container_status >/dev/null 2>&1; then
        echo -e "${YELLOW}Container not running - Python services not available${NC}"
        return
    fi
    
    local services=(
        "ha-intent-predictor.service"
        "ha-predictor-ingestion.service"
        "ha-predictor-training.service"
        "ha-predictor-api.service"
    )
    
    for service in "${services[@]}"; do
        if pct exec "$CTID" -- systemctl list-unit-files "$service" &>/dev/null; then
            local status=$(pct exec "$CTID" -- systemctl is-active "$service" 2>/dev/null || echo "inactive")
            local enabled=$(pct exec "$CTID" -- systemctl is-enabled "$service" 2>/dev/null || echo "disabled")
            
            case "$status" in
                "active")
                    echo -e "  ${GREEN}●${NC} $service - ${GREEN}$status${NC} (${enabled})"
                    ;;
                "failed")
                    echo -e "  ${RED}●${NC} $service - ${RED}$status${NC} (${enabled})"
                    ;;
                *)
                    echo -e "  ${YELLOW}●${NC} $service - ${YELLOW}$status${NC} (${enabled})"
                    ;;
            esac
        else
            echo -e "  ${YELLOW}○${NC} $service - ${YELLOW}not installed${NC}"
        fi
    done
}

function monitor_installation_progress() {
    echo -e "\n${PURPLE}=== INSTALLATION PROGRESS ===${NC}"
    
    # Check for progress indicators
    local steps=(
        "Container Creation:pct status $CTID"
        "System Updates:pct exec $CTID -- dpkg -l | grep -q updated"
        "Docker Installation:pct exec $CTID -- which docker"
        "Python Environment:pct exec $CTID -- test -d /opt/ha-intent-predictor/venv"
        "Database Setup:pct exec $CTID -- docker ps | grep -q postgres"
        "Services Configuration:pct exec $CTID -- test -f /etc/systemd/system/ha-intent-predictor.service"
        "Application Startup:pct exec $CTID -- systemctl is-active ha-intent-predictor.service"
    )
    
    local completed=0
    local total=${#steps[@]}
    
    for step in "${steps[@]}"; do
        local name="${step%%:*}"
        local command="${step##*:}"
        
        if eval "$command" &>/dev/null; then
            echo -e "  ${GREEN}✓${NC} $name"
            ((completed++))
        else
            echo -e "  ${YELLOW}○${NC} $name"
        fi
    done
    
    local progress=$(( (completed * 100) / total ))
    echo -e "\n${CYAN}Overall Progress: ${progress}% (${completed}/${total} steps)${NC}"
    
    # Progress bar
    local bar_length=40
    local filled=$(( (completed * bar_length) / total ))
    local empty=$(( bar_length - filled ))
    
    printf "  ["
    printf "%*s" $filled | tr ' ' '█'
    printf "%*s" $empty | tr ' ' '░'
    printf "] %d%%\n" $progress
}

function monitor_logs() {
    echo -e "\n${PURPLE}=== RECENT LOGS ===${NC}"
    
    if ! check_container_status >/dev/null 2>&1; then
        echo -e "${YELLOW}Container not running - Logs not available${NC}"
        return
    fi
    
    # Check installation logs
    if pct exec "$CTID" -- test -f /var/log/ha-intent-predictor/app.log; then
        echo -e "\n${CYAN}Application Logs (last 5 lines):${NC}"
        pct exec "$CTID" -- tail -5 /var/log/ha-intent-predictor/app.log 2>/dev/null || echo -e "${YELLOW}  No application logs yet${NC}"
    fi
    
    # Check systemd journal for our services
    echo -e "\n${CYAN}System Logs (last 3 lines):${NC}"
    pct exec "$CTID" -- journalctl -u ha-intent-predictor.service --no-pager -n 3 2>/dev/null || echo -e "${YELLOW}  No service logs yet${NC}"
}

function monitor_network_connectivity() {
    echo -e "\n${PURPLE}=== NETWORK CONNECTIVITY ===${NC}"
    
    if ! check_container_status >/dev/null 2>&1; then
        echo -e "${YELLOW}Container not running - Network not testable${NC}"
        return
    fi
    
    # Test basic connectivity
    if pct exec "$CTID" -- ping -c 1 8.8.8.8 &>/dev/null; then
        log_success "Internet connectivity: OK"
    else
        log_error "Internet connectivity: FAILED"
    fi
    
    # Test internal services
    local services=(
        "5432:PostgreSQL"
        "6379:Redis"
        "9092:Kafka"
        "8000:API"
        "3000:Grafana"
    )
    
    echo -e "\n${CYAN}Internal Services:${NC}"
    for service in "${services[@]}"; do
        local port="${service%%:*}"
        local name="${service##*:}"
        
        if pct exec "$CTID" -- timeout 3 bash -c "</dev/tcp/localhost/$port" &>/dev/null; then
            echo -e "  ${GREEN}✓${NC} $name (port $port)"
        else
            echo -e "  ${YELLOW}○${NC} $name (port $port) - not ready"
        fi
    done
}

function monitor_database_health() {
    echo -e "\n${PURPLE}=== DATABASE HEALTH ===${NC}"
    
    if ! check_container_status >/dev/null 2>&1; then
        echo -e "${YELLOW}Container not running - Database not testable${NC}"
        return
    fi
    
    # Check if PostgreSQL container is running
    if pct exec "$CTID" -- docker ps | grep -q ha-predictor-postgres; then
        log_success "PostgreSQL container running"
        
        # Test database connection
        if pct exec "$CTID" -- docker exec ha-predictor-postgres pg_isready -U ha_predictor &>/dev/null; then
            log_success "Database accepting connections"
            
            # Check if tables exist
            local table_count=$(pct exec "$CTID" -- docker exec ha-predictor-postgres psql -U ha_predictor -d ha_predictor -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema IN ('raw_data', 'features', 'models', 'analytics');" 2>/dev/null | tr -d ' ' || echo "0")
            
            if [ "$table_count" -gt 0 ]; then
                log_success "Database schema initialized ($table_count tables)"
            else
                log_warning "Database schema not initialized yet"
            fi
        else
            log_warning "Database not accepting connections yet"
        fi
    else
        echo -e "${YELLOW}PostgreSQL container not running yet${NC}"
    fi
}

function show_helpful_commands() {
    echo -e "\n${PURPLE}=== HELPFUL COMMANDS ===${NC}"
    echo -e "${CYAN}Monitor installation in real-time:${NC}"
    echo -e "  pct enter $CTID"
    echo -e "  journalctl -f"
    echo -e ""
    echo -e "${CYAN}Check specific service logs:${NC}"
    echo -e "  pct exec $CTID -- journalctl -u ha-intent-predictor.service -f"
    echo -e "  pct exec $CTID -- docker compose logs -f"
    echo -e ""
    echo -e "${CYAN}Manual troubleshooting:${NC}"
    echo -e "  pct exec $CTID -- systemctl status ha-intent-predictor.service"
    echo -e "  pct exec $CTID -- docker compose ps"
    echo -e "  pct exec $CTID -- curl http://localhost:8000/health"
}

function main_monitor_loop() {
    if [ -z "$CTID" ]; then
        usage
    fi
    
    # Create monitor log
    echo "Installation monitor started at $(date)" > "$MONITOR_LOG"
    
    while true; do
        header
        
        monitor_system_resources
        monitor_installation_progress
        monitor_docker_services
        monitor_python_services
        monitor_network_connectivity
        monitor_database_health
        monitor_logs
        show_helpful_commands
        
        echo -e "\n${BLUE}Press Ctrl+C to exit monitor${NC}"
        echo -e "${BLUE}Refreshing in 10 seconds...${NC}"
        
        sleep 10
    done
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${GREEN}Monitor stopped.${NC}"; exit 0' INT

# Parse arguments
if [ $# -ne 1 ]; then
    usage
fi

CTID="$1"

# Validate container ID
if ! [[ "$CTID" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: Container ID must be numeric${NC}"
    exit 1
fi

# Start monitoring
main_monitor_loop