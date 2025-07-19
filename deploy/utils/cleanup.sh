#!/usr/bin/env bash

# HA Intent Predictor - Cleanup and Rollback Utility
# Safely removes a failed or unwanted installation

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

CTID=""
FORCE_CLEANUP=false
BACKUP_BEFORE_CLEANUP=true

function usage() {
    echo "Usage: $0 <container_id> [options]"
    echo ""
    echo "Options:"
    echo "  --force              Force cleanup without confirmation"
    echo "  --no-backup          Skip backup before cleanup"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 <container_id>                    # Interactive cleanup"
    echo "  $0 <container_id> --force           # Force cleanup without confirmation"
    echo "  $0 <container_id> --no-backup       # Cleanup without creating backup"
    exit 1
}

function msg_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

function msg_ok() {
    echo -e "${GREEN}[✓]${NC} $1"
}

function msg_error() {
    echo -e "${RED}[✗]${NC} $1"
}

function msg_warn() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

function header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                     HA INTENT PREDICTOR - CLEANUP UTILITY                   ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo
}

function check_container_exists() {
    if ! pct status "$CTID" &>/dev/null; then
        msg_error "Container $CTID does not exist"
        return 1
    fi
    return 0
}

function show_container_info() {
    msg_info "Container Information:"
    
    local status=$(pct status "$CTID" | awk '{print $2}')
    local config=$(pct config "$CTID" 2>/dev/null || echo "Config unavailable")
    
    echo -e "  Status: ${CYAN}$status${NC}"
    
    if [[ "$config" != "Config unavailable" ]]; then
        local hostname=$(echo "$config" | grep "^hostname:" | cut -d' ' -f2 || echo "unknown")
        local memory=$(echo "$config" | grep "^memory:" | cut -d' ' -f2 || echo "unknown")
        local cores=$(echo "$config" | grep "^cores:" | cut -d' ' -f2 || echo "unknown")
        
        echo -e "  Hostname: ${CYAN}$hostname${NC}"
        echo -e "  Memory: ${CYAN}${memory}MB${NC}"
        echo -e "  Cores: ${CYAN}$cores${NC}"
    fi
    
    # Check if it looks like our installation
    if [ "$status" = "running" ]; then
        if pct exec "$CTID" -- test -d /opt/ha-intent-predictor 2>/dev/null; then
            echo -e "  HA Intent Predictor: ${GREEN}Detected${NC}"
            
            # Show Docker containers if any
            local docker_containers=$(pct exec "$CTID" -- docker ps --format "table {{.Names}}\t{{.Status}}" 2>/dev/null | grep ha-predictor || echo "")
            if [ -n "$docker_containers" ]; then
                echo -e "  Docker Services:"
                echo "$docker_containers" | while read line; do
                    echo -e "    ${CYAN}$line${NC}"
                done
            fi
        else
            echo -e "  HA Intent Predictor: ${YELLOW}Not detected${NC}"
        fi
    fi
    echo
}

function create_backup() {
    if [ "$BACKUP_BEFORE_CLEANUP" = false ]; then
        msg_info "Skipping backup (--no-backup specified)"
        return 0
    fi
    
    msg_info "Creating backup before cleanup..."
    
    local backup_dir="/tmp/ha-predictor-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Export container configuration
    msg_info "Backing up container configuration..."
    pct config "$CTID" > "$backup_dir/container-config.conf" 2>/dev/null || true
    
    # Backup application data if container is running
    if pct status "$CTID" | grep -q "running"; then
        if pct exec "$CTID" -- test -d /opt/ha-intent-predictor 2>/dev/null; then
            msg_info "Backing up application data..."
            
            # Backup configuration files
            pct exec "$CTID" -- bash -c "
                if [ -d /opt/ha-intent-predictor/config ]; then
                    tar -czf /tmp/app-config-backup.tar.gz -C /opt/ha-intent-predictor config/ 2>/dev/null || true
                fi
            " 2>/dev/null || true
            
            # Copy backup file from container
            if pct exec "$CTID" -- test -f /tmp/app-config-backup.tar.gz 2>/dev/null; then
                pct pull "$CTID" /tmp/app-config-backup.tar.gz "$backup_dir/app-config-backup.tar.gz" 2>/dev/null || true
                pct exec "$CTID" -- rm -f /tmp/app-config-backup.tar.gz 2>/dev/null || true
            fi
            
            # Backup database if possible
            if pct exec "$CTID" -- docker ps | grep -q ha-predictor-postgres 2>/dev/null; then
                msg_info "Backing up database..."
                pct exec "$CTID" -- docker exec ha-predictor-postgres pg_dump -U ha_predictor ha_predictor > "$backup_dir/database-backup.sql" 2>/dev/null || true
            fi
        fi
    fi
    
    # Create backup summary
    cat > "$backup_dir/backup-info.txt" << EOF
HA Intent Predictor Backup
Created: $(date)
Container ID: $CTID
Backup Directory: $backup_dir

Files included:
- container-config.conf: Container configuration
- app-config-backup.tar.gz: Application configuration files
- database-backup.sql: Database dump (if available)

To restore:
1. Create a new container using the configuration
2. Restore application files to /opt/ha-intent-predictor/config/
3. Import database dump if needed
EOF
    
    msg_ok "Backup created in: $backup_dir"
    echo
}

function stop_services() {
    msg_info "Stopping services..."
    
    if ! pct status "$CTID" | grep -q "running"; then
        msg_warn "Container is not running"
        return 0
    fi
    
    # Stop systemd services
    local services=(
        "ha-intent-predictor.service"
        "ha-predictor-ingestion.service"
        "ha-predictor-training.service"
        "ha-predictor-api.service"
        "nginx"
    )
    
    for service in "${services[@]}"; do
        if pct exec "$CTID" -- systemctl is-active "$service" &>/dev/null; then
            msg_info "Stopping $service..."
            pct exec "$CTID" -- systemctl stop "$service" 2>/dev/null || true
        fi
    done
    
    # Stop Docker containers
    if pct exec "$CTID" -- which docker &>/dev/null; then
        msg_info "Stopping Docker containers..."
        pct exec "$CTID" -- bash -c "
            cd /opt/ha-intent-predictor 2>/dev/null && docker compose down 2>/dev/null || true
            docker stop \$(docker ps -aq) 2>/dev/null || true
        " 2>/dev/null || true
    fi
    
    msg_ok "Services stopped"
}

function cleanup_application_data() {
    msg_info "Cleaning up application data..."
    
    if ! pct status "$CTID" | grep -q "running"; then
        msg_warn "Container is not running - skipping application cleanup"
        return 0
    fi
    
    # Remove application directory
    if pct exec "$CTID" -- test -d /opt/ha-intent-predictor 2>/dev/null; then
        msg_info "Removing application directory..."
        pct exec "$CTID" -- rm -rf /opt/ha-intent-predictor 2>/dev/null || true
    fi
    
    # Remove systemd services
    local service_files=(
        "/etc/systemd/system/ha-intent-predictor.service"
        "/etc/systemd/system/ha-predictor-ingestion.service"
        "/etc/systemd/system/ha-predictor-training.service"
        "/etc/systemd/system/ha-predictor-api.service"
    )
    
    for service_file in "${service_files[@]}"; do
        if pct exec "$CTID" -- test -f "$service_file" 2>/dev/null; then
            msg_info "Removing service file: $(basename $service_file)"
            pct exec "$CTID" -- rm -f "$service_file" 2>/dev/null || true
        fi
    done
    
    # Reload systemd
    pct exec "$CTID" -- systemctl daemon-reload 2>/dev/null || true
    
    # Remove nginx site
    if pct exec "$CTID" -- test -f /etc/nginx/sites-enabled/ha-intent-predictor 2>/dev/null; then
        msg_info "Removing nginx configuration..."
        pct exec "$CTID" -- rm -f /etc/nginx/sites-enabled/ha-intent-predictor 2>/dev/null || true
        pct exec "$CTID" -- rm -f /etc/nginx/sites-available/ha-intent-predictor 2>/dev/null || true
        pct exec "$CTID" -- systemctl reload nginx 2>/dev/null || true
    fi
    
    # Remove log directories
    if pct exec "$CTID" -- test -d /var/log/ha-intent-predictor 2>/dev/null; then
        msg_info "Removing log directories..."
        pct exec "$CTID" -- rm -rf /var/log/ha-intent-predictor 2>/dev/null || true
    fi
    
    # Clean Docker data
    if pct exec "$CTID" -- which docker &>/dev/null; then
        msg_info "Cleaning Docker data..."
        pct exec "$CTID" -- bash -c "
            docker system prune -af 2>/dev/null || true
            docker volume prune -f 2>/dev/null || true
        " 2>/dev/null || true
    fi
    
    msg_ok "Application data cleaned up"
}

function remove_container() {
    msg_info "Removing container..."
    
    # Stop container if running
    if pct status "$CTID" | grep -q "running"; then
        msg_info "Stopping container..."
        pct stop "$CTID" 2>/dev/null || true
        
        # Wait for container to stop
        local attempts=0
        while [ $attempts -lt 30 ]; do
            if ! pct status "$CTID" | grep -q "running"; then
                break
            fi
            sleep 2
            ((attempts++))
        done
    fi
    
    # Destroy container
    msg_info "Destroying container..."
    if pct destroy "$CTID" 2>/dev/null; then
        msg_ok "Container $CTID removed successfully"
    else
        msg_error "Failed to remove container $CTID"
        return 1
    fi
}

function partial_cleanup() {
    msg_info "Performing partial cleanup (keeping container)..."
    
    stop_services
    cleanup_application_data
    
    msg_ok "Partial cleanup completed"
    msg_info "Container $CTID is still available but cleaned up"
    msg_info "You can now run the installer again or manually remove the container"
}

function full_cleanup() {
    msg_info "Performing full cleanup (removing container)..."
    
    create_backup
    stop_services
    remove_container
    
    msg_ok "Full cleanup completed"
    msg_info "Container $CTID has been completely removed"
}

function interactive_cleanup() {
    show_container_info
    
    echo -e "${YELLOW}Choose cleanup option:${NC}"
    echo "1) Partial cleanup (remove app data, keep container)"
    echo "2) Full cleanup (remove entire container)"
    echo "3) Cancel"
    echo
    
    read -p "$(echo -e ${CYAN}Enter your choice [1-3]: ${NC})" choice
    
    case "$choice" in
        1)
            partial_cleanup
            ;;
        2)
            full_cleanup
            ;;
        3)
            msg_info "Cleanup cancelled"
            exit 0
            ;;
        *)
            msg_error "Invalid choice"
            exit 1
            ;;
    esac
}

function main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                FORCE_CLEANUP=true
                shift
                ;;
            --no-backup)
                BACKUP_BEFORE_CLEANUP=false
                shift
                ;;
            --help)
                usage
                ;;
            *)
                if [ -z "$CTID" ]; then
                    CTID="$1"
                else
                    msg_error "Unknown option: $1"
                    usage
                fi
                shift
                ;;
        esac
    done
    
    # Validate arguments
    if [ -z "$CTID" ]; then
        msg_error "Container ID is required"
        usage
    fi
    
    if ! [[ "$CTID" =~ ^[0-9]+$ ]]; then
        msg_error "Container ID must be numeric"
        exit 1
    fi
    
    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        msg_error "This script must be run as root"
        exit 1
    fi
    
    header
    
    # Check if container exists
    if ! check_container_exists; then
        exit 1
    fi
    
    # Perform cleanup
    if [ "$FORCE_CLEANUP" = true ]; then
        msg_warn "Force cleanup mode - removing entire container without confirmation"
        full_cleanup
    else
        interactive_cleanup
    fi
    
    echo
    msg_ok "Cleanup operation completed successfully!"
}

# Run main function
main "$@"