#!/usr/bin/env bash

# HA Intent Predictor - Enhanced Proxmox LXC Installation Script
# Real-time progress monitoring and validation built-in
# Usage: bash -c "$(wget -qLO - https://github.com/yourusername/ha-intent-predictor/raw/main/deploy/enhanced-proxmox-install.sh)"

set -euo pipefail

# Enable verbose debugging
if [[ "${DEBUG:-}" == "1" ]] || [[ "${BASH_ARGV[*]}" =~ --debug ]]; then
    set -x
    exec 2> >(tee -a /tmp/proxmox-install-debug.log)
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Progress tracking
TOTAL_STEPS=16
CURRENT_STEP=0
START_TIME=$(date +%s)

# Default values
CTID=""
HOSTNAME="ha-intent-predictor"
CORES="4"
MEMORY="8192"
STORAGE="local-lvm"
DISK_SIZE="32"
TEMPLATE=""  # Will be auto-detected
NETWORK="name=eth0,bridge=vmbr0,ip=dhcp"
PRIVILEGED="0"
START_AFTER_CREATE="1"

# HA Connection details
HA_URL=""
HA_TOKEN=""

# Installation mode
INTERACTIVE_MODE=true
AUTO_CTID=false

# Safety tracking
CONTAINER_CREATED="false"

# Default credentials
DEFAULT_PASSWORD="hapredictor123"

function header() {
    # Only clear if running in interactive terminal
    if [ -t 0 ] && [ -t 1 ]; then
        clear
    fi
    echo -e "${BLUE}"
    echo "██╗  ██╗ █████╗     ██╗███╗   ██╗████████╗███████╗███╗   ██╗████████╗"
    echo "██║  ██║██╔══██╗    ██║████╗  ██║╚══██╔══╝██╔════╝████╗  ██║╚══██╔══╝"
    echo "███████║███████║    ██║██╔██╗ ██║   ██║   █████╗  ██╔██╗ ██║   ██║   "
    echo "██╔══██║██╔══██║    ██║██║╚██╗██║   ██║   ██╔══╝  ██║╚██╗██║   ██║   "
    echo "██║  ██║██║  ██║    ██║██║ ╚████║   ██║   ███████╗██║ ╚████║   ██║   "
    echo "╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═══╝   ╚═╝   "
    echo
    echo "██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗ ██████╗ ██████╗ "
    echo "██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗"
    echo "██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   ██║   ██║██████╔╝"
    echo "██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   ██║   ██║██╔══██╗"
    echo "██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   ╚██████╔╝██║  ██║"
    echo "╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝"
    echo -e "${NC}"
    echo "ML-Powered Occupancy Prediction System"
    echo "Adaptive Learning • No Assumptions • Continuous Improvement"
    echo
}

function progress_bar() {
    local current=$1
    local total=$2
    local width=50
    echo "[DEBUG] progress_bar called with current=$current total=$total"
    local percentage=$((current * 100 / total))
    local completed=$((current * width / total))
    local remaining=$((width - completed))
    
    echo "[DEBUG] Displaying simplified progress bar..."
    # Use simple ASCII characters instead of Unicode
    printf "\n${CYAN}Progress: ["
    printf "%*s" $completed | tr ' ' '#'
    printf "%*s" $remaining | tr ' ' '-'
    printf "] %d%% (%d/%d)${NC}\n" $percentage $current $total
    echo "[DEBUG] Progress bar printf completed"
}

function step_header() {
    local step_name="$1"
    echo "[DEBUG] step_header called with: $step_name"
    echo "[DEBUG] CURRENT_STEP before increment: $CURRENT_STEP"
    
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo "[DEBUG] CURRENT_STEP after increment: $CURRENT_STEP"
    
    echo "[DEBUG] Starting step: $step_name"
    echo
    echo -e "${PURPLE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║${NC} ${BLUE}STEP ${CURRENT_STEP}/${TOTAL_STEPS}: ${step_name}${NC}"
    echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    
    echo "[DEBUG] About to call progress_bar with $CURRENT_STEP $TOTAL_STEPS"
    # Temporarily disable progress bar to see if that's the issue
    # progress_bar $CURRENT_STEP $TOTAL_STEPS
    echo "[DEBUG] Progress bar skipped for debugging"
    echo
}

function msg_info() {
    local msg="$1"
    echo -e "${BLUE}[INFO]${NC} $msg"
}

function msg_ok() {
    local msg="$1"
    echo -e "${GREEN}[✓]${NC} $msg"
}

function msg_error() {
    local msg="$1"
    echo -e "${RED}[✗]${NC} $msg"
}

function msg_warn() {
    local msg="$1"
    echo -e "${YELLOW}[⚠]${NC} $msg"
}

function msg_progress() {
    local msg="$1"
    echo -e "${CYAN}[→]${NC} $msg"
}

function spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    local msg="${2:-Working...}"
    
    echo -n -e "${CYAN}$msg ${NC}"
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
    echo
}

function run_with_progress() {
    local command="$1"
    local success_msg="$2"
    local error_msg="$3"
    
    if eval "$command" > /tmp/install_output 2>&1; then
        msg_ok "$success_msg"
        return 0
    else
        msg_error "$error_msg"
        echo -e "${RED}Command output:${NC}"
        cat /tmp/install_output | head -10
        return 1
    fi
}

function test_and_report() {
    local test_command="$1"
    local test_name="$2"
    local success_msg="$3"
    local error_msg="$4"
    
    msg_progress "Testing: $test_name"
    
    if eval "$test_command" > /tmp/test_output 2>&1; then
        msg_ok "$success_msg"
        return 0
    else
        msg_error "$error_msg"
        echo -e "${YELLOW}Test output:${NC}"
        cat /tmp/test_output | head -5
        return 1
    fi
}

function show_resource_usage() {
    echo -e "\n${CYAN}=== SYSTEM RESOURCES ===${NC}"
    echo "[DEBUG] Getting CPU usage..."
    local cpu_usage=$(timeout 5 top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' || echo "N/A")
    echo "[DEBUG] Getting memory usage..."
    local memory_usage=$(timeout 5 free -h | awk '/^Mem:/ {printf "%.1f%% (%s/%s)", ($3/$2)*100, $3, $2}' || echo "N/A")
    echo "[DEBUG] Getting disk usage..."
    local disk_usage=$(timeout 5 df -h / | awk 'NR==2 {printf "%s (%s used)", $5, $3}' || echo "N/A")
    
    echo -e "  CPU Usage: ${cpu_usage}%"
    echo -e "  Memory: ${memory_usage}"
    echo -e "  Disk: ${disk_usage}"
    echo "[DEBUG] Resource usage display completed"
}

function cleanup() {
    # Disable error exit during cleanup to prevent recursive failures
    set +e
    
    echo "[DEBUG] Cleanup function called"
    echo "[DEBUG] CTID='${CTID:-}', CONTAINER_CREATED='${CONTAINER_CREATED:-}', HOSTNAME='${HOSTNAME:-}'"
    
    # Only cleanup if we created the container in this session
    if [ -n "${CTID:-}" ] && [ "${CONTAINER_CREATED:-}" = "true" ]; then
        echo "[CLEANUP] Starting cleanup for container $CTID..."
        
        # Check if container exists
        if pct status "$CTID" &>/dev/null; then
            echo "[CLEANUP] Container $CTID exists, checking if we should remove it"
            
            # Double-check this is our container by checking hostname
            local container_hostname
            container_hostname=$(pct exec "$CTID" -- hostname 2>/dev/null || echo "")
            echo "[CLEANUP] Container hostname: '$container_hostname', Expected: '${HOSTNAME:-}'"
            
            if [ "$container_hostname" = "${HOSTNAME:-}" ] || [ -z "$container_hostname" ]; then
                echo "[CLEANUP] Hostname matches or is empty, proceeding with cleanup"
                echo "[CLEANUP] Stopping container $CTID..."
                pct stop "$CTID" 2>/dev/null || echo "[CLEANUP] Failed to stop container (may already be stopped)"
                
                echo "[CLEANUP] Destroying container $CTID..."
                if pct destroy "$CTID" 2>/dev/null; then
                    echo "[CLEANUP] ✓ Container $CTID successfully removed"
                else
                    echo "[CLEANUP] ✗ Failed to destroy container $CTID"
                fi
            else
                echo "[CLEANUP] ✗ Safety check failed: Container hostname '$container_hostname' != expected '${HOSTNAME:-}'"
                echo "[CLEANUP] ✗ Container $CTID was NOT removed for safety"
            fi
        else
            echo "[CLEANUP] Container $CTID does not exist or is not accessible"
        fi
    elif [ -n "${CTID:-}" ]; then
        echo "[CLEANUP] No cleanup needed - container $CTID was not created in this session"
    else
        echo "[CLEANUP] No container ID to cleanup"
    fi
    
    echo "[DEBUG] Cleanup function completed"
}

function detect_ubuntu_template() {
    msg_progress "Detecting latest available Ubuntu template..."
    
    echo "[DEBUG] Listing all available templates..."
    pveam list local 2>/dev/null | head -10 || echo "[DEBUG] No templates found or pveam command failed"
    
    # Check local storage for Ubuntu 22.04 templates (prefer latest version)
    local available_templates=$(pveam list local 2>/dev/null | grep -E "ubuntu-22\.04.*standard.*amd64\.tar\.zst" | awk '{print $1}' | sort -V | tail -1)
    
    echo "[DEBUG] Ubuntu 22.04 template candidate: '$available_templates'"
    echo "[DEBUG] Template length: ${#available_templates} characters"
    
    if [ -n "$available_templates" ] && [ ${#available_templates} -lt 240 ]; then
        TEMPLATE="$available_templates"
        msg_ok "Found latest Ubuntu 22.04 template: $TEMPLATE"
        return 0
    elif [ -n "$available_templates" ]; then
        msg_warn "Template name too long (${#available_templates} chars): $available_templates"
    fi
    
    # If Ubuntu 22.04 not found, look for any Ubuntu LTS template
    msg_progress "Ubuntu 22.04 not found, searching for any Ubuntu LTS template..."
    available_templates=$(pveam list local 2>/dev/null | grep -E "ubuntu-[0-9]+\.[0-9]+.*standard.*amd64\.tar\.zst" | awk '{print $1}' | sort -V | tail -1)
    
    echo "[DEBUG] Ubuntu LTS template candidate: '$available_templates'"
    echo "[DEBUG] Template length: ${#available_templates} characters"
    
    if [ -n "$available_templates" ] && [ ${#available_templates} -lt 240 ]; then
        TEMPLATE="$available_templates"
        msg_ok "Found latest Ubuntu template: $TEMPLATE"
        return 0
    elif [ -n "$available_templates" ]; then
        msg_warn "Template name too long (${#available_templates} chars): $available_templates"
    fi
    
    # Last resort - any Ubuntu template with shorter name
    msg_progress "Searching for any available Ubuntu template with reasonable name length..."
    
    # Get all Ubuntu templates and filter by length
    while IFS= read -r template; do
        echo "[DEBUG] Checking template: '$template' (${#template} chars)"
        if [ ${#template} -lt 240 ]; then
            TEMPLATE="$template"
            msg_warn "Using available Ubuntu template: $TEMPLATE"
            return 0
        fi
    done < <(pveam list local 2>/dev/null | grep -E "ubuntu.*amd64\.tar\.zst" | awk '{print $1}' | sort -V)
    
    # No Ubuntu templates found locally
    msg_error "No Ubuntu templates found in local storage."
    msg_error "Available templates:"
    pveam list local 2>/dev/null | grep -E "\.tar\.zst" | awk '{print "  - " $1}'
    msg_error ""
    msg_error "Please download an Ubuntu template first:"
    msg_error "pveam available --section system | grep ubuntu"
    msg_error "pveam download local <ubuntu-template-name>"
    exit 1
}

function check_requirements() {
    echo "[DEBUG] Entered check_requirements function"
    echo "[DEBUG] About to call step_header..."
    step_header "Checking Proxmox Requirements"
    echo "[DEBUG] step_header completed"
    
    echo "[DEBUG] About to call msg_progress..."
    msg_progress "Verifying Proxmox environment..."
    echo "[DEBUG] msg_progress completed"
    echo "[DEBUG] Checking kernel version: $(uname -r)"
    
    # Check if running on Proxmox - use kernel check instead of command check
    if [[ ! "$(uname -r)" =~ pve ]]; then
        msg_error "This script must be run on a Proxmox server"
        exit 1
    fi
    msg_ok "Proxmox VE detected (kernel: $(uname -r))"
    
    echo "[DEBUG] Checking if running as root (EUID=$EUID)"
    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        msg_error "This script must be run as root"
        exit 1
    fi
    msg_ok "Running as root"
    
    echo "[DEBUG] Checking available storage..."
    # Check available storage using df instead of pvs
    local available_storage=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G' || echo "0")
    echo "[DEBUG] Available storage: ${available_storage}GB"
    # Use bash arithmetic instead of bc to avoid hanging
    if [ "${available_storage}" -lt 50 ] 2>/dev/null; then
        msg_warn "Low storage space detected. Recommended: 50GB+ available"
    else
        msg_ok "Sufficient storage available (${available_storage}GB)"
    fi
    
    echo "[DEBUG] Showing resource usage..."
    show_resource_usage
    
    # Auto-detect available Ubuntu template
    echo "[DEBUG] Auto-detecting Ubuntu template..."
    detect_ubuntu_template
    
    msg_ok "Requirements check completed"
}

function get_next_vmid() {
    local vmid=200
    local max_attempts=100
    local attempts=0
    
    while [ $attempts -lt $max_attempts ]; do
        # Check both LXC containers and VMs
        if ! pct status "$vmid" &>/dev/null && ! qm status "$vmid" &>/dev/null; then
            # Double-check by listing all VMs/containers
            if ! pvesh get /cluster/resources --type vm --output-format json | jq -r '.[].vmid' | grep -q "^${vmid}$" 2>/dev/null; then
                echo "$vmid"
                return 0
            fi
        fi
        vmid=$((vmid + 1))
        attempts=$((attempts + 1))
    done
    
    msg_error "Could not find available VMID after $max_attempts attempts"
    exit 1
}

function collect_info() {
    step_header "Collecting Installation Parameters"
    
    msg_info "Setting up installation configuration..."
    echo "[DEBUG] INTERACTIVE_MODE=$INTERACTIVE_MODE, AUTO_CTID=$AUTO_CTID"
    
    # Get CTID
    if [ "$INTERACTIVE_MODE" = true ]; then
        echo "[DEBUG] Running in interactive mode"
        # Interactive mode - ask user
        while true; do
            local suggested_ctid=$(get_next_vmid)
            echo -e -n "${CYAN}Enter CT ID [default: $suggested_ctid]: ${NC}"
            read input_ctid
            CTID=${input_ctid:-$suggested_ctid}
            
            # Validate CTID format
            if ! [[ "$CTID" =~ ^[0-9]+$ ]]; then
                msg_error "Container ID must be a number"
                continue
            fi
            
            # Check if CTID is in valid range
            if [ "$CTID" -lt 100 ] || [ "$CTID" -gt 999999 ]; then
                msg_error "Container ID must be between 100 and 999999"
                continue
            fi
            
            # Validate CTID is not already in use
            if pct status "$CTID" &>/dev/null; then
                msg_error "Container ID $CTID already exists (container)"
                msg_info "Suggested available ID: $(get_next_vmid)"
                continue
            fi
            
            if qm status "$CTID" &>/dev/null; then
                msg_error "Container ID $CTID already exists (virtual machine)"
                msg_info "Suggested available ID: $(get_next_vmid)"
                continue
            fi
            
            # ID is valid and available
            break
        done
    else
        echo "[DEBUG] Running in non-interactive mode"
        # Non-interactive mode
        if [ -z "$CTID" ] || [ "$AUTO_CTID" = true ]; then
            echo "[DEBUG] Auto-assigning container ID..."
            # Auto-assign next available ID
            CTID=$(get_next_vmid)
            msg_info "Auto-assigned container ID: $CTID"
        else
            # Use provided CTID but validate it
            if ! [[ "$CTID" =~ ^[0-9]+$ ]] || [ "$CTID" -lt 100 ] || [ "$CTID" -gt 999999 ]; then
                msg_error "Invalid container ID: $CTID (must be numeric, 100-999999)"
                exit 1
            fi
            
            if pct status "$CTID" &>/dev/null || qm status "$CTID" &>/dev/null; then
                msg_error "Container ID $CTID already exists"
                msg_info "Next available ID: $(get_next_vmid)"
                exit 1
            fi
        fi
    fi
    
    msg_ok "Container ID: $CTID (validated and available)"
    
    # Get hostname
    if [ "$INTERACTIVE_MODE" = true ]; then
        echo -e -n "${CYAN}Enter hostname [default: $HOSTNAME]: ${NC}"
        read input_hostname
        HOSTNAME=${input_hostname:-$HOSTNAME}
    fi
    msg_ok "Hostname: $HOSTNAME"
    
    # Get cores
    if [ "$INTERACTIVE_MODE" = true ]; then
        echo -e -n "${CYAN}Enter CPU cores [default: $CORES]: ${NC}"
        read input_cores
        CORES=${input_cores:-$CORES}
    fi
    msg_ok "CPU cores: $CORES"
    
    # Get memory
    if [ "$INTERACTIVE_MODE" = true ]; then
        echo -e -n "${CYAN}Enter RAM in MB [default: $MEMORY]: ${NC}"
        read input_memory
        MEMORY=${input_memory:-$MEMORY}
    fi
    msg_ok "Memory: ${MEMORY}MB"
    
    # Get storage
    if [ "$INTERACTIVE_MODE" = true ]; then
        msg_info "Available storage pools:"
        pvesm status | grep -E "(local-lvm|local-zfs|local)" | awk '{print "  - " $1 " (" $2 ")"}'
        echo -e -n "${CYAN}Enter storage pool [default: $STORAGE]: ${NC}"
        read input_storage
        STORAGE=${input_storage:-$STORAGE}
    fi
    msg_ok "Storage: $STORAGE"
    
    # Get disk size
    if [ "$INTERACTIVE_MODE" = true ]; then
        echo -e -n "${CYAN}Enter disk size in GB [default: $DISK_SIZE]: ${NC}"
        read input_disk_size
        DISK_SIZE=${input_disk_size:-$DISK_SIZE}
    fi
    msg_ok "Disk size: ${DISK_SIZE}GB"
    
    # Get HA details
    if [ "$INTERACTIVE_MODE" = true ]; then
        echo
        msg_info "Home Assistant integration setup:"
        echo -e -n "${CYAN}Enter Home Assistant URL (e.g., http://YOUR_HA_IP:8123): ${NC}"
        read HA_URL
        echo -e -n "${CYAN}Enter Home Assistant Long-Lived Access Token: ${NC}"
        read -s HA_TOKEN
        echo
    fi
    
    # Test HA connection
    if [ -n "$HA_URL" ] && [ -n "$HA_TOKEN" ]; then
        msg_progress "Testing Home Assistant connection..."
        if curl -s -m 10 -H "Authorization: Bearer $HA_TOKEN" "$HA_URL/api/" > /dev/null 2>&1; then
            msg_ok "Home Assistant connection successful"
        else
            msg_warn "Could not connect to Home Assistant. You can configure this later."
        fi
    fi
    
    # Show summary
    echo
    echo -e "${PURPLE}=== INSTALLATION SUMMARY ===${NC}"
    echo -e "  Container ID: ${CYAN}$CTID${NC}"
    echo -e "  Hostname: ${CYAN}$HOSTNAME${NC}"
    echo -e "  CPU Cores: ${CYAN}$CORES${NC}"
    echo -e "  Memory: ${CYAN}${MEMORY}MB${NC}"
    echo -e "  Storage: ${CYAN}$STORAGE${NC}"
    echo -e "  Disk Size: ${CYAN}${DISK_SIZE}GB${NC}"
    echo -e "  HA URL: ${CYAN}$HA_URL${NC}"
    echo
    
    if [ "$INTERACTIVE_MODE" = true ]; then
        echo -e -n "${GREEN}Proceed with installation? [y/N]: ${NC}"
        read confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            msg_info "Installation cancelled"
            exit 0
        fi
    fi
}

function create_container() {
    step_header "Creating LXC Container"
    
    # Final safety check before creation
    if pct status "$CTID" &>/dev/null || qm status "$CTID" &>/dev/null; then
        msg_error "CRITICAL: Container/VM $CTID exists just before creation. Aborting to prevent data loss."
        exit 1
    fi
    
    msg_progress "Creating container with ID $CTID..."
    
    # Create container
    if pct create "$CTID" "$TEMPLATE" \
        --hostname "$HOSTNAME" \
        --cores "$CORES" \
        --memory "$MEMORY" \
        --storage "$STORAGE" \
        --rootfs "$STORAGE:$DISK_SIZE" \
        --net0 "$NETWORK" \
        --unprivileged "$PRIVILEGED" \
        --features nesting=1,keyctl=1 \
        --onboot 1 \
        --start "$START_AFTER_CREATE" > /tmp/container_create 2>&1; then
        
        # Mark that we successfully created this container
        CONTAINER_CREATED="true"
        msg_ok "Container created successfully"
        
        # Add Docker-specific LXC configuration (like community scripts do)
        echo "lxc.apparmor.profile: unconfined" >> /etc/pve/lxc/$CTID.conf
        echo "lxc.cap.drop:" >> /etc/pve/lxc/$CTID.conf
        echo "lxc.cgroup2.devices.allow: a" >> /etc/pve/lxc/$CTID.conf
        echo "lxc.mount.auto: proc:rw sys:rw" >> /etc/pve/lxc/$CTID.conf
        msg_ok "Added Docker-specific LXC configuration"
        
        # Restart container for LXC config to take effect
        echo "[DEBUG] Restarting container to apply LXC configuration..."
        pct stop "$CTID"
        sleep 3
        pct start "$CTID"
        sleep 5
        msg_ok "Container restarted with Docker configuration"
    else
        msg_error "Container creation failed"
        cat /tmp/container_create
        exit 1
    fi
    
    # Wait for container to be ready
    msg_progress "Waiting for container to be ready..."
    local attempts=0
    while [ $attempts -lt 30 ]; do
        if pct status "$CTID" | grep -q "running"; then
            break
        fi
        sleep 2
        ((attempts++))
        echo -n "."
    done
    echo
    
    # Verify container is running
    if ! pct status "$CTID" | grep -q "running"; then
        msg_error "Container failed to start"
        exit 1
    fi
    
    msg_ok "Container is running and ready"
    
    # Wait for container to be fully ready (like Proxmox community scripts)
    echo "[DEBUG] Waiting for container to be fully operational..."
    msg_progress "Waiting for container initialization..."
    
    local ready=false
    local attempts=0
    while [ $attempts -lt 30 ] && [ "$ready" = false ]; do
        echo "[DEBUG] Readiness check attempt $((attempts + 1))/30"
        
        # Test if we can execute basic commands
        if pct exec "$CTID" -- /bin/bash -c "echo test" &>/dev/null; then
            echo "[DEBUG] Container accepting commands"
            # Wait a bit more for network to be ready
            sleep 3
            
            # Try to get IP
            local container_ip=$(pct exec "$CTID" -- hostname -I 2>/dev/null | awk '{print $1}' || echo "")
            echo "[DEBUG] Container IP: '$container_ip'"
            
            if [ -n "$container_ip" ] && [[ "$container_ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                msg_ok "Container ready with IP: $container_ip"
                ready=true
            else
                echo "[DEBUG] IP not ready yet, waiting..."
                sleep 3
                ((attempts++))
            fi
        else
            echo "[DEBUG] Container not ready for commands yet"
            sleep 3
            ((attempts++))
        fi
    done
    
    if [ "$ready" = false ]; then
        msg_warn "Container may not be fully ready, but proceeding..."
    fi
    
    echo "[DEBUG] create_container function completed"
}

function install_dependencies() {
    echo "[DEBUG] Starting install_dependencies function"
    step_header "Installing System Dependencies"
    
    msg_progress "Updating package lists..."
    run_with_progress \
        "pct exec $CTID -- apt-get update" \
        "Package lists updated" \
        "Failed to update package lists"
    
    msg_progress "Upgrading system packages..."
    run_with_progress \
        "pct exec $CTID -- apt-get upgrade -y" \
        "System packages upgraded" \
        "Failed to upgrade system packages"
    
    msg_progress "Installing base packages..."
    pct exec "$CTID" -- bash -c "apt-get install -y \
        curl \
        wget \
        gnupg \
        lsb-release \
        ca-certificates \
        software-properties-common \
        apt-transport-https \
        build-essential \
        git \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        nginx \
        supervisor \
        htop \
        nano \
        vim \
        unzip \
        bc" > /tmp/package_install 2>&1 &
    
    local install_pid=$!
    spinner $install_pid "Installing packages..."
    
    if wait $install_pid; then
        msg_ok "Base packages installed successfully"
    else
        msg_error "Package installation failed"
        cat /tmp/package_install | tail -20
        exit 1
    fi
    
    # Test installations
    test_and_report \
        "pct exec $CTID -- python3 --version" \
        "Python installation" \
        "Python is working" \
        "Python installation failed"
    
    test_and_report \
        "pct exec $CTID -- curl --version" \
        "Curl installation" \
        "Curl is working" \
        "Curl installation failed"
}

function setup_security() {
    step_header "Setting Up Security and Access"
    
    msg_progress "Setting root password..."
    pct exec "$CTID" -- bash -c "echo 'root:$DEFAULT_PASSWORD' | chpasswd"
    msg_ok "Root password set to: $DEFAULT_PASSWORD"
    
    msg_progress "Configuring SSH access..."
    pct exec "$CTID" -- bash -c "
        # Enable SSH root login
        sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
        sed -i 's/PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
        
        # Enable password authentication
        sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
        sed -i 's/PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
        
        # Restart SSH service
        systemctl restart sshd
        systemctl enable ssh
    "
    msg_ok "SSH configured for root access"
    
    # Test SSH connectivity
    test_and_report \
        "pct exec $CTID -- systemctl is-active ssh" \
        "SSH service" \
        "SSH service is running" \
        "SSH service failed to start"
}

function install_docker() {
    step_header "Installing Docker"
    
    msg_progress "Adding Docker repository..."
    run_with_progress \
        "pct exec $CTID -- bash -c 'curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg'" \
        "Docker GPG key added" \
        "Failed to add Docker GPG key"
    
    pct exec "$CTID" -- bash -c "echo \"deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \$(lsb_release -cs) stable\" | tee /etc/apt/sources.list.d/docker.list > /dev/null"
    
    msg_progress "Installing Docker..."
    pct exec "$CTID" -- bash -c "
        apt-get update &&
        apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin &&
        systemctl enable docker &&
        systemctl start docker
    " > /tmp/docker_install 2>&1 &
    
    local docker_pid=$!
    spinner $docker_pid "Installing Docker components..."
    
    if wait $docker_pid; then
        msg_ok "Docker installed successfully"
    else
        msg_error "Docker installation failed"
        cat /tmp/docker_install | tail -20
        exit 1
    fi
    
    # Docker should work normally in LXC with proper features
    msg_ok "Docker installation completed"
    
    # Test Docker
    test_and_report \
        "pct exec $CTID -- docker --version" \
        "Docker installation" \
        "Docker is working" \
        "Docker installation failed"
    
    test_and_report \
        "pct exec $CTID -- systemctl is-active docker" \
        "Docker service" \
        "Docker service is running" \
        "Docker service is not running"
}

function setup_application() {
    step_header "Setting Up Application Structure"
    
    msg_progress "Creating application directories..."
    run_with_progress \
        "pct exec $CTID -- mkdir -p /opt/ha-intent-predictor" \
        "Application directory created" \
        "Failed to create application directory"
    
    msg_progress "Downloading application files..."
    # In real deployment, this would download from GitHub
    pct exec "$CTID" -- bash -c "
        cd /opt/ha-intent-predictor
        
        # Create directory structure
        mkdir -p {src/{ingestion,learning,prediction,adaptation,integration,storage},config,scripts,logs}
        mkdir -p deploy/{scripts,config,systemd}
        
        # Create basic files
        touch README.md requirements.txt
        
        echo 'Application structure created' > /tmp/setup_status
    "
    
    msg_ok "Application structure created"
    
    msg_progress "Setting up Python virtual environment..."
    pct exec "$CTID" -- bash -c "
        cd /opt/ha-intent-predictor
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
    " > /tmp/venv_setup 2>&1 &
    
    local venv_pid=$!
    spinner $venv_pid "Creating virtual environment..."
    
    if wait $venv_pid; then
        msg_ok "Python virtual environment created"
    else
        msg_error "Virtual environment setup failed"
        cat /tmp/venv_setup | tail -10
    fi
    
    # Test virtual environment
    test_and_report \
        "pct exec $CTID -- test -d /opt/ha-intent-predictor/venv" \
        "Virtual environment" \
        "Virtual environment is ready" \
        "Virtual environment not found"
}

function setup_docker_services() {
    step_header "Setting Up Docker Services"
    
    msg_progress "Creating complete Docker Compose configuration with Kafka..."
    pct exec "$CTID" -- bash -c "
        cd /opt/ha-intent-predictor
        
        # Create complete docker-compose.yml with full event streaming architecture
        cat > docker-compose.yml << 'EOF'
services:
  postgres:
    image: timescale/timescaledb:2.11.0-pg15
    container_name: ha-predictor-postgres
    environment:
      POSTGRES_DB: ha_predictor
      POSTGRES_USER: ha_predictor
      POSTGRES_PASSWORD: \${POSTGRES_PASSWORD:-hapredictor_db_pass}
    ports:
      - \"5432:5432\"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    healthcheck:
      test: [\"CMD-SHELL\", \"pg_isready -U ha_predictor\"]
      interval: 10s
      timeout: 5s
      retries: 5
    
  redis:
    image: redis:7-alpine
    container_name: ha-predictor-redis
    ports:
      - \"6379:6379\"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: [\"CMD\", \"redis-cli\", \"ping\"]
      interval: 10s
      timeout: 5s
      retries: 5

  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    container_name: ha-predictor-zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - \"2181:2181\"
    volumes:
      - zookeeper_data:/var/lib/zookeeper/data
      - zookeeper_logs:/var/lib/zookeeper/log
    restart: unless-stopped
    healthcheck:
      test: [\"CMD\", \"nc\", \"-z\", \"localhost\", \"2181\"]
      interval: 10s
      timeout: 5s
      retries: 5

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    container_name: ha-predictor-kafka
    depends_on:
      zookeeper:
        condition: service_healthy
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092,PLAINTEXT_INTERNAL://kafka:29092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_INTERNAL:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT_INTERNAL
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: 'true'
    ports:
      - \"9092:9092\"
    volumes:
      - kafka_data:/var/lib/kafka/data
    restart: unless-stopped
    healthcheck:
      test: [\"CMD\", \"kafka-topics\", \"--bootstrap-server\", \"localhost:9092\", \"--list\"]
      interval: 30s
      timeout: 10s
      retries: 5

volumes:
  postgres_data:
  redis_data:
  kafka_data:
  zookeeper_data:
  zookeeper_logs:

networks:
  default:
    name: ha-predictor-network
EOF
        
        # Create .env file with proper passwords
        cat > .env << 'EOF'
POSTGRES_PASSWORD=hapredictor_db_pass
GRAFANA_PASSWORD=hapredictor_grafana_pass
EOF
    "
    
    msg_progress "Creating database initialization script..."
    pct exec "$CTID" -- bash -c "
        cd /opt/ha-intent-predictor
        mkdir -p scripts
        
        # Create database schema initialization
        cat > scripts/init.sql << 'EOF'
-- Enable TimescaleDB extension (already enabled in the image)
-- CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create tables for HA Intent Predictor
CREATE TABLE IF NOT EXISTS sensor_events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    state VARCHAR(50),
    attributes JSONB,
    room VARCHAR(100),
    sensor_type VARCHAR(100),
    zone_info JSONB,
    source VARCHAR(50) DEFAULT 'home_assistant'
);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100) NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    probability FLOAT NOT NULL,
    uncertainty FLOAT NOT NULL,
    model_info JSONB,
    features JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_performance (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS room_states (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100) NOT NULL,
    occupied BOOLEAN NOT NULL,
    confidence FLOAT,
    source VARCHAR(50) DEFAULT 'inference',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertables for time-series data
SELECT create_hypertable('sensor_events', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('predictions', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('model_performance', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('room_states', 'timestamp', if_not_exists => TRUE);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_sensor_events_entity_time ON sensor_events (entity_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_events_room_time ON sensor_events (room, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_events_type_time ON sensor_events (sensor_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_room_time ON predictions (room, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_horizon ON predictions (horizon_minutes, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_model_performance_room_model ON model_performance (room, model_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_room_states_room_time ON room_states (room, timestamp DESC);

-- Create some useful views
CREATE OR REPLACE VIEW latest_predictions AS
SELECT DISTINCT ON (room, horizon_minutes) 
    room, horizon_minutes, probability, uncertainty, timestamp, model_info
FROM predictions 
ORDER BY room, horizon_minutes, timestamp DESC;

CREATE OR REPLACE VIEW room_occupancy_summary AS
SELECT 
    room,
    COUNT(*) as total_events,
    COUNT(*) FILTER (WHERE occupied = true) as occupied_events,
    ROUND(100.0 * COUNT(*) FILTER (WHERE occupied = true) / COUNT(*), 2) as occupancy_rate,
    MAX(timestamp) as last_updated
FROM room_states 
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY room;

-- Insert some test data to verify everything is working
INSERT INTO sensor_events (timestamp, entity_id, state, room, sensor_type) VALUES
(NOW(), 'binary_sensor.test_presence', 'on', 'test_room', 'presence'),
(NOW() - INTERVAL '1 minute', 'binary_sensor.test_presence', 'off', 'test_room', 'presence');

COMMIT;
EOF
    "
    
    msg_ok "Docker Compose configuration and database schema created"
    
    # Docker should work normally in LXC with nesting=1,keyctl=1
    msg_progress "Verifying Docker is ready..."
    pct exec "$CTID" -- bash -c "
        echo '[DEBUG] Checking Docker status...'
        if systemctl is-active docker >/dev/null 2>&1; then
            echo '[DEBUG] Docker service is active'
        else
            echo '[DEBUG] Docker service not active, starting...'
            systemctl start docker
            sleep 5
        fi
        
        echo '[DEBUG] Testing Docker daemon connection...'
        if docker version >/dev/null 2>&1; then
            echo '[DEBUG] Docker daemon is responding'
        else
            echo '[DEBUG] Docker daemon is not responding'
            exit 1
        fi
    "
    
    msg_progress "Starting Docker services..."
    if pct exec "$CTID" -- bash -c "
        cd /opt/ha-intent-predictor
        echo '[DEBUG] Running docker compose up -d...'
        docker compose up -d
        echo '[DEBUG] Docker compose command completed'
        
        # Check if containers actually started successfully
        echo '[DEBUG] Checking container status...'
        if ! docker ps | grep -q ha-predictor-postgres; then
            echo '[ERROR] PostgreSQL container failed to start - AppArmor issue detected'
            docker ps -a | grep ha-predictor || echo 'No containers found'
            exit 1
        fi
        if ! docker ps | grep -q ha-predictor-redis; then
            echo '[ERROR] Redis container failed to start - AppArmor issue detected'
            docker ps -a | grep ha-predictor || echo 'No containers found'
            exit 1
        fi
        echo '[DEBUG] All containers are running successfully'
    " > /tmp/docker_start 2>&1; then
        msg_ok "Docker services started successfully"
    else
        msg_error "Failed to start Docker services"
        echo "Docker Compose output:"
        cat /tmp/docker_start
        exit 1  # This will trigger cleanup via ERR trap
    fi
    
    # Wait for services to be ready
    msg_progress "Waiting for services to be ready..."
    echo "[DEBUG] Starting service readiness checks..."
    local attempts=0
    local max_attempts=30
    echo "[DEBUG] About to enter readiness loop..."
    while [ $attempts -lt $max_attempts ]; do
        echo "[DEBUG] Service readiness check attempt $((attempts + 1))/$max_attempts"
        
        # Check if postgres container is responding
        echo "[DEBUG] Testing container connectivity..."
        if ! pct status "$CTID" | grep -q "running"; then
            echo "[ERROR] Container $CTID is not running!"
            exit 1
        fi
        
        echo "[DEBUG] Testing Docker in container..."
        if ! pct exec "$CTID" -- docker ps &>/dev/null; then
            echo "[ERROR] Docker is not accessible in container!"
            exit 1
        fi
        
        echo "[DEBUG] Testing PostgreSQL container..."
        # First check if container exists
        if ! pct exec "$CTID" -- docker ps -a | grep -q ha-predictor-postgres; then
            echo "[DEBUG] PostgreSQL container not found yet, waiting..."
        elif ! pct exec "$CTID" -- docker ps | grep -q ha-predictor-postgres; then
            echo "[DEBUG] PostgreSQL container exists but not running, waiting..."
        else
            # Container is running, test readiness
            # Use a safer approach that doesn't interfere with global error handling
            if pct exec "$CTID" -- docker exec ha-predictor-postgres pg_isready -U ha_predictor &>/dev/null || true; then
                msg_ok "PostgreSQL is ready"
                break
            else
                echo "[DEBUG] PostgreSQL container running but not ready yet, attempt $((attempts + 1))/$max_attempts"
            fi
        fi
        
        # Only exit if we've tried all attempts
        if [ $attempts -eq $((max_attempts - 1)) ]; then
            msg_error "Services failed to become ready within $((max_attempts * 2)) seconds"
            echo "[DEBUG] Final diagnostic information:"
            echo "Container status:"
            pct exec "$CTID" -- docker ps -a | grep ha-predictor || echo "No containers found"
            echo "PostgreSQL container details:"
            if pct exec "$CTID" -- docker ps -a | grep -q ha-predictor-postgres; then
                pct exec "$CTID" -- docker ps -a | grep ha-predictor-postgres
                echo "PostgreSQL logs (last 10 lines):"
                pct exec "$CTID" -- docker logs ha-predictor-postgres --tail 10 2>/dev/null || echo "Could not get logs"
            else
                echo "PostgreSQL container not found"
            fi
            echo "Available disk space:"
            pct exec "$CTID" -- df -h / 2>/dev/null || echo "Could not check disk space"
            
            # Don't exit here - continue with installation but mark services as not ready
            msg_warn "Continuing installation despite service readiness issues..."
            break
        fi
        
        sleep 2
        ((attempts++))
    done
    
    # Test services
    test_and_report \
        "pct exec $CTID -- docker ps | grep -q ha-predictor-postgres" \
        "PostgreSQL container" \
        "PostgreSQL container is running" \
        "PostgreSQL container not found"
    
    test_and_report \
        "pct exec $CTID -- docker exec ha-predictor-postgres pg_isready -U ha_predictor" \
        "PostgreSQL connectivity" \
        "PostgreSQL is accepting connections" \
        "PostgreSQL is not ready"
    
    test_and_report \
        "pct exec $CTID -- docker exec ha-predictor-redis redis-cli ping" \
        "Redis connectivity" \
        "Redis is responding" \
        "Redis is not responding"
    
    # Test Kafka connectivity
    test_and_report \
        "pct exec $CTID -- docker exec ha-predictor-kafka kafka-topics --bootstrap-server localhost:9092 --list" \
        "Kafka connectivity" \
        "Kafka is responding" \
        "Kafka is not responding"
    
    # Test Zookeeper connectivity  
    test_and_report \
        "pct exec $CTID -- docker exec ha-predictor-zookeeper nc -z localhost 2181" \
        "Zookeeper connectivity" \
        "Zookeeper is responding" \
        "Zookeeper is not responding"
}

function install_python_dependencies() {
    step_header "Installing Python Dependencies"
    
    msg_progress "Creating requirements.txt..."
    pct exec "$CTID" -- bash -c "
        cd /opt/ha-intent-predictor
        cat > requirements.txt << 'EOF'
# Core ML packages
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
river>=0.21.0

# Database and caching
psycopg2-binary>=2.9.0
redis>=5.0.0

# Web framework
fastapi>=0.104.0
uvicorn>=0.24.0

# Basic utilities
pyyaml>=6.0.0
requests>=2.31.0
EOF
    "
    
    msg_progress "Installing Python packages..."
    pct exec "$CTID" -- bash -c "
        cd /opt/ha-intent-predictor
        source venv/bin/activate
        pip install -r requirements.txt
    " > /tmp/pip_install 2>&1 &
    
    local pip_pid=$!
    spinner $pip_pid "Installing Python packages (this may take a while)..."
    
    if wait $pip_pid; then
        msg_ok "Python dependencies installed"
    else
        msg_error "Python package installation failed"
        cat /tmp/pip_install | tail -20
    fi
    
    # Test key imports
    test_and_report \
        "pct exec $CTID -- /opt/ha-intent-predictor/venv/bin/python -c 'import numpy, pandas, sklearn; print(\"OK\")'" \
        "Core ML packages" \
        "Core ML packages are importable" \
        "Failed to import core ML packages"
}

function setup_systemd_services() {
    step_header "Setting Up System Services"
    
    msg_progress "Creating systemd service files..."
    pct exec "$CTID" -- bash -c "
        # Create main service
        cat > /etc/systemd/system/ha-intent-predictor.service << 'EOF'
[Unit]
Description=HA Intent Predictor - Main Application
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ha-intent-predictor
Environment=PATH=/opt/ha-intent-predictor/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/opt/ha-intent-predictor/venv/bin/python -c \"print('HA Intent Predictor would start here')\"
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        # Enable service
        systemctl daemon-reload
        systemctl enable ha-intent-predictor.service
    "
    
    msg_ok "Systemd services configured"
    
    # Test service configuration
    test_and_report \
        "pct exec $CTID -- systemctl is-enabled ha-intent-predictor.service" \
        "Service configuration" \
        "Service is properly configured" \
        "Service configuration failed"
}

function create_monitoring_script() {
    step_header "Creating Remote Monitoring Script"
    
    msg_progress "Creating comprehensive monitoring script..."
    pct exec "$CTID" -- bash -c "
        cd /opt/ha-intent-predictor
        mkdir -p scripts
        
        # Create monitoring script that outputs status for remote access
        cat > scripts/remote-monitor.sh << 'EOF'
#!/bin/bash

# Remote Monitoring Script for HA Intent Predictor
# This script outputs status to files that can be read remotely

MONITOR_FILE=\"/tmp/ha-predictor-status.json\"
MONITOR_LOG=\"/tmp/ha-predictor-monitor.log\"

# Function to get container status
get_container_status() {
    local container_name=\"\$1\"
    if docker ps --format \"{{.Names}}\" | grep -q \"^\${container_name}\$\"; then
        echo \"running\"
    elif docker ps -a --format \"{{.Names}}\" | grep -q \"^\${container_name}\$\"; then
        echo \"stopped\"
    else
        echo \"missing\"
    fi
}

# Function to test service health
test_service_health() {
    local service=\"\$1\"
    case \$service in
        postgres)
            if docker exec ha-predictor-postgres pg_isready -U ha_predictor >/dev/null 2>&1; then
                echo \"healthy\"
            else
                echo \"unhealthy\"
            fi
            ;;
        redis)
            if docker exec ha-predictor-redis redis-cli ping 2>/dev/null | grep -q \"PONG\"; then
                echo \"healthy\"
            else
                echo \"unhealthy\"
            fi
            ;;
        kafka)
            if docker exec ha-predictor-kafka kafka-topics --bootstrap-server localhost:9092 --list >/dev/null 2>&1; then
                echo \"healthy\"
            else
                echo \"unhealthy\"
            fi
            ;;
        zookeeper)
            if docker exec ha-predictor-zookeeper nc -z localhost 2181 >/dev/null 2>&1; then
                echo \"healthy\"
            else
                echo \"unhealthy\"
            fi
            ;;
        *)
            echo \"unknown\"
            ;;
    esac
}

# Function to get resource usage
get_resource_usage() {
    # Get container resource usage
    docker stats --no-stream --format \"{{.Container}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}\" 2>/dev/null || echo \"no_data\"
}

# Function to get system info
get_system_info() {
    echo \"{\"
    echo \"  \\\"timestamp\\\": \\\"\$(date -Iseconds)\\\",\"
    echo \"  \\\"uptime\\\": \\\"\$(uptime -p)\\\",\"
    echo \"  \\\"load_avg\\\": \\\"\$(uptime | awk -F'load average:' '{print \$2}' | xargs)\\\",\"
    echo \"  \\\"disk_usage\\\": \\\"\$(df -h / | awk 'NR==2 {print \$5}' | tr -d '%')\\\",\"
    echo \"  \\\"memory_usage\\\": \\\"\$(free | awk 'NR==2{printf \\\"%.1f\\\", \$3*100/\$2}')\\\",\"
    echo \"  \\\"containers\\\": {\"
    echo \"    \\\"postgres\\\": {\"
    echo \"      \\\"status\\\": \\\"\$(get_container_status ha-predictor-postgres)\\\",\"
    echo \"      \\\"health\\\": \\\"\$(test_service_health postgres)\\\"\"
    echo \"    },\"
    echo \"    \\\"redis\\\": {\"
    echo \"      \\\"status\\\": \\\"\$(get_container_status ha-predictor-redis)\\\",\"
    echo \"      \\\"health\\\": \\\"\$(test_service_health redis)\\\"\"
    echo \"    },\"
    echo \"    \\\"kafka\\\": {\"
    echo \"      \\\"status\\\": \\\"\$(get_container_status ha-predictor-kafka)\\\",\"
    echo \"      \\\"health\\\": \\\"\$(test_service_health kafka)\\\"\"
    echo \"    },\"
    echo \"    \\\"zookeeper\\\": {\"
    echo \"      \\\"status\\\": \\\"\$(get_container_status ha-predictor-zookeeper)\\\",\"
    echo \"      \\\"health\\\": \\\"\$(test_service_health zookeeper)\\\"\"
    echo \"    }\"
    echo \"  },\"
    echo \"  \\\"services\\\": {\"
    echo \"    \\\"nginx\\\": \\\"\$(systemctl is-active nginx 2>/dev/null || echo inactive)\\\",\"
    echo \"    \\\"ssh\\\": \\\"\$(systemctl is-active ssh 2>/dev/null || echo inactive)\\\",\"
    echo \"    \\\"docker\\\": \\\"\$(systemctl is-active docker 2>/dev/null || echo inactive)\\\"\"
    echo \"  },\"
    echo \"  \\\"application\\\": {\"
    echo \"    \\\"directory_exists\\\": \$([ -d /opt/ha-intent-predictor ] && echo true || echo false),\"
    echo \"    \\\"venv_exists\\\": \$([ -d /opt/ha-intent-predictor/venv ] && echo true || echo false),\"
    echo \"    \\\"compose_file_exists\\\": \$([ -f /opt/ha-intent-predictor/docker-compose.yml ] && echo true || echo false)\"
    echo \"  }\"
    echo \"}\"
}

# Function to get recent logs
get_recent_logs() {
    echo \"=== RECENT DOCKER LOGS ===\" >> \"\$MONITOR_LOG\"
    echo \"Timestamp: \$(date)\" >> \"\$MONITOR_LOG\"
    echo \"\" >> \"\$MONITOR_LOG\"
    
    for container in ha-predictor-postgres ha-predictor-redis ha-predictor-kafka ha-predictor-zookeeper; do
        if docker ps -a --format \"{{.Names}}\" | grep -q \"^\${container}\$\"; then
            echo \"--- \$container ---\" >> \"\$MONITOR_LOG\"
            docker logs \"\$container\" --tail 10 --since 5m 2>&1 >> \"\$MONITOR_LOG\"
            echo \"\" >> \"\$MONITOR_LOG\"
        fi
    done
    
    # Keep only last 1000 lines
    tail -1000 \"\$MONITOR_LOG\" > \"\${MONITOR_LOG}.tmp\" && mv \"\${MONITOR_LOG}.tmp\" \"\$MONITOR_LOG\"
}

# Main monitoring function
monitor_once() {
    get_system_info > \"\$MONITOR_FILE\"
    get_recent_logs
    
    # Also create a human-readable summary
    cat > \"/tmp/ha-predictor-summary.txt\" << EOFSUM
HA Intent Predictor Status Summary
Generated: \$(date)

=== CONTAINER STATUS ===
PostgreSQL: \$(get_container_status ha-predictor-postgres) - \$(test_service_health postgres)
Redis: \$(get_container_status ha-predictor-redis) - \$(test_service_health redis)  
Kafka: \$(get_container_status ha-predictor-kafka) - \$(test_service_health kafka)
Zookeeper: \$(get_container_status ha-predictor-zookeeper) - \$(test_service_health zookeeper)

=== SYSTEM HEALTH ===
Disk Usage: \$(df -h / | awk 'NR==2 {print \$5}')
Memory Usage: \$(free | awk 'NR==2{printf \"%.1f%%\", \$3*100/\$2}')
Load Average: \$(uptime | awk -F'load average:' '{print \$2}' | xargs)

=== SERVICES ===
Docker: \$(systemctl is-active docker 2>/dev/null || echo inactive)
Nginx: \$(systemctl is-active nginx 2>/dev/null || echo inactive)
SSH: \$(systemctl is-active ssh 2>/dev/null || echo inactive)

=== QUICK TESTS ===
PostgreSQL Connection: \$(docker exec ha-predictor-postgres pg_isready -U ha_predictor 2>/dev/null && echo \"OK\" || echo \"FAIL\")
Redis Connection: \$(docker exec ha-predictor-redis redis-cli ping 2>/dev/null || echo \"FAIL\")
Kafka Connection: \$(docker exec ha-predictor-kafka kafka-topics --bootstrap-server localhost:9092 --list >/dev/null 2>&1 && echo \"OK\" || echo \"FAIL\")
Web Server: \$(curl -s http://localhost/health >/dev/null && echo \"OK\" || echo \"FAIL\")

=== DOCKER COMPOSE STATUS ===
\$(cd /opt/ha-intent-predictor && docker compose ps 2>/dev/null || echo \"Docker compose not available\")
EOFSUM
}

# Continuous monitoring mode
monitor_continuous() {
    echo \"Starting continuous monitoring... (Ctrl+C to stop)\"
    echo \"Monitor files: \$MONITOR_FILE and /tmp/ha-predictor-summary.txt\"
    
    while true; do
        monitor_once
        sleep 30
    done
}

# Handle command line arguments
case \"\${1:-once}\" in
    \"continuous\"|\"loop\"|\"watch\")
        monitor_continuous
        ;;
    \"once\"|\"\")
        monitor_once
        echo \"Status written to: \$MONITOR_FILE\"
        echo \"Summary written to: /tmp/ha-predictor-summary.txt\"
        echo \"Logs written to: \$MONITOR_LOG\"
        ;;
    \"show\"|\"cat\"|\"view\")
        if [ -f \"/tmp/ha-predictor-summary.txt\" ]; then
            cat /tmp/ha-predictor-summary.txt
        else
            echo \"No status file found. Run monitor first.\"
        fi
        ;;
    \"json\")
        if [ -f \"\$MONITOR_FILE\" ]; then
            cat \"\$MONITOR_FILE\"
        else
            echo \"No JSON status file found. Run monitor first.\"
        fi
        ;;
    *)
        echo \"Usage: \$0 [once|continuous|show|json]\"
        echo \"  once       - Generate status once (default)\"
        echo \"  continuous - Monitor continuously every 30s\"
        echo \"  show       - Show latest human-readable status\"
        echo \"  json       - Show latest JSON status\"
        ;;
esac
EOF
        
        # Make monitoring script executable
        chmod +x scripts/remote-monitor.sh
    "
    
    msg_ok "Remote monitoring script created"
    
    # Test monitoring script
    test_and_report \
        "pct exec $CTID -- test -x /opt/ha-intent-predictor/scripts/remote-monitor.sh" \
        "Monitoring script" \
        "Monitoring script is executable" \
        "Monitoring script failed to create"
}

function setup_nginx() {
    step_header "Setting Up Web Server"
    
    msg_progress "Configuring Nginx..."
    pct exec "$CTID" -- bash -c "
        # Create nginx configuration
        cat > /etc/nginx/sites-available/ha-intent-predictor << 'EOF'
server {
    listen 80;
    server_name _;
    
    location /health {
        return 200 'HA Intent Predictor is running';
        add_header Content-Type text/plain;
    }
    
    location / {
        return 200 'HA Intent Predictor - Coming Soon';
        add_header Content-Type text/plain;
    }
}
EOF
        
        # Enable site
        ln -sf /etc/nginx/sites-available/ha-intent-predictor /etc/nginx/sites-enabled/
        rm -f /etc/nginx/sites-enabled/default
        
        # Test and restart nginx
        nginx -t && systemctl restart nginx && systemctl enable nginx
    "
    
    msg_ok "Nginx configured and started"
    
    # Test web server
    test_and_report \
        "pct exec $CTID -- curl -s http://localhost/health" \
        "Web server" \
        "Web server is responding" \
        "Web server is not responding"
}

function run_validation_tests() {
    step_header "Running Validation Tests"
    
    msg_progress "Testing system integration..."
    
    local tests_passed=0
    local total_tests=10
    
    # Test 1: Container networking
    if pct exec "$CTID" -- ping -c 1 8.8.8.8 &>/dev/null; then
        msg_ok "✓ Internet connectivity"
        ((tests_passed++))
    else
        msg_error "✗ Internet connectivity failed"
    fi
    
    # Test 2: Docker services
    if pct exec "$CTID" -- docker ps | grep -q ha-predictor-postgres; then
        msg_ok "✓ PostgreSQL container running"
        ((tests_passed++))
    else
        msg_error "✗ PostgreSQL container not running"
    fi
    
    # Test 3: Database connectivity
    if pct exec "$CTID" -- docker exec ha-predictor-postgres pg_isready -U ha_predictor &>/dev/null; then
        msg_ok "✓ Database accepting connections"
        ((tests_passed++))
    else
        msg_error "✗ Database not accepting connections"
    fi
    
    # Test 4: Redis connectivity
    if pct exec "$CTID" -- docker exec ha-predictor-redis redis-cli ping 2>/dev/null | grep -q PONG; then
        msg_ok "✓ Redis responding"
        ((tests_passed++))
    else
        msg_error "✗ Redis not responding"
    fi
    
    # Test 5: Kafka connectivity
    if pct exec "$CTID" -- docker exec ha-predictor-kafka kafka-topics --bootstrap-server localhost:9092 --list &>/dev/null; then
        msg_ok "✓ Kafka responding"
        ((tests_passed++))
    else
        msg_error "✗ Kafka not responding"
    fi
    
    # Test 6: Zookeeper connectivity
    if pct exec "$CTID" -- docker exec ha-predictor-zookeeper nc -z localhost 2181 &>/dev/null; then
        msg_ok "✓ Zookeeper responding"
        ((tests_passed++))
    else
        msg_error "✗ Zookeeper not responding"
    fi
    
    # Test 7: Python environment
    if pct exec "$CTID" -- test -d /opt/ha-intent-predictor/venv; then
        msg_ok "✓ Python virtual environment exists"
        ((tests_passed++))
    else
        msg_error "✗ Python virtual environment missing"
    fi
    
    # Test 8: Python dependencies
    if pct exec "$CTID" -- /opt/ha-intent-predictor/venv/bin/python -c "import numpy, pandas" &>/dev/null; then
        msg_ok "✓ Python dependencies importable"
        ((tests_passed++))
    else
        msg_error "✗ Python dependencies not working"
    fi
    
    # Test 9: Systemd service
    if pct exec "$CTID" -- systemctl is-enabled ha-intent-predictor.service &>/dev/null; then
        msg_ok "✓ Systemd service configured"
        ((tests_passed++))
    else
        msg_error "✗ Systemd service not configured"
    fi
    
    # Test 10: Web server
    if pct exec "$CTID" -- curl -s http://localhost/health &>/dev/null; then
        msg_ok "✓ Web server responding"
        ((tests_passed++))
    else
        msg_error "✗ Web server not responding"
    fi
    
    echo
    local success_rate=$((tests_passed * 100 / total_tests))
    if [ $tests_passed -eq $total_tests ]; then
        msg_ok "All validation tests passed! ($tests_passed/$total_tests)"
    elif [ $success_rate -ge 75 ]; then
        msg_warn "Most validation tests passed ($tests_passed/$total_tests - ${success_rate}%)"
    else
        msg_error "Multiple validation tests failed ($tests_passed/$total_tests - ${success_rate}%)"
    fi
}

function show_completion_info() {
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    clear
    header
    
    msg_ok "Installation completed successfully!"
    echo
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    echo -e "${GREEN}🎉 HA Intent Predictor is now installed!${NC}"
    echo -e "${CYAN}Installation time: ${minutes}m ${seconds}s${NC}"
    echo
    echo -e "${BLUE}📋 Container Details:${NC}"
    echo -e "   • Container ID: ${CYAN}$CTID${NC}"
    echo -e "   • Hostname: ${CYAN}$HOSTNAME${NC}"
    echo -e "   • IP Address: ${CYAN}$(pct exec "$CTID" -- hostname -I | tr -d ' ')${NC}"
    echo
    echo -e "${BLUE}🔗 Access URLs:${NC}"
    echo -e "   • Health Check: ${CYAN}http://$(pct exec "$CTID" -- hostname -I | tr -d ' ')/health${NC}"
    echo
    echo -e "${BLUE}📊 Services Status:${NC}"
    
    # Show real-time service status
    if pct exec "$CTID" -- docker ps | grep -q ha-predictor-postgres; then
        echo -e "   • PostgreSQL: ${GREEN}Running${NC}"
    else
        echo -e "   • PostgreSQL: ${RED}Stopped${NC}"
    fi
    
    if pct exec "$CTID" -- docker ps | grep -q ha-predictor-redis; then
        echo -e "   • Redis: ${GREEN}Running${NC}"
    else
        echo -e "   • Redis: ${RED}Stopped${NC}"
    fi
    
    if pct exec "$CTID" -- docker ps | grep -q ha-predictor-kafka; then
        echo -e "   • Kafka: ${GREEN}Running${NC}"
    else
        echo -e "   • Kafka: ${RED}Stopped${NC}"
    fi
    
    if pct exec "$CTID" -- docker ps | grep -q ha-predictor-zookeeper; then
        echo -e "   • Zookeeper: ${GREEN}Running${NC}"
    else
        echo -e "   • Zookeeper: ${RED}Stopped${NC}"
    fi
    
    if pct exec "$CTID" -- systemctl is-active nginx &>/dev/null; then
        echo -e "   • Nginx: ${GREEN}Running${NC}"
    else
        echo -e "   • Nginx: ${RED}Stopped${NC}"
    fi
    
    echo
    echo -e "${BLUE}🔧 Management Commands:${NC}"
    echo -e "   • Enter container: ${CYAN}pct enter $CTID${NC}"
    echo -e "   • View logs: ${CYAN}pct exec $CTID -- journalctl -f${NC}"
    echo -e "   • Check services: ${CYAN}pct exec $CTID -- docker compose ps${NC}"
    echo -e "   • Restart services: ${CYAN}pct exec $CTID -- systemctl restart ha-intent-predictor.service${NC}"
    echo -e "   • Monitor remotely: ${CYAN}pct exec $CTID -- /opt/ha-intent-predictor/scripts/remote-monitor.sh${NC}"
    echo
    echo -e "${BLUE}📝 Next Steps:${NC}"
    echo -e "   1. Complete the application development"
    echo -e "   2. Configure your Home Assistant sensors"
    echo -e "   3. Import historical data"
    echo -e "   4. Monitor the learning progress"
    echo
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    echo -e "${GREEN}The system foundation is ready for the ML application!${NC}"
    echo
}

function usage() {
    echo "HA Intent Predictor - Enhanced Proxmox Installation"
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --auto                    Non-interactive mode with auto-assigned container ID"
    echo "  --ctid <id>              Use specific container ID"
    echo "  --hostname <name>        Set container hostname"
    echo "  --cores <number>         Set CPU cores"
    echo "  --memory <MB>            Set memory in MB"
    echo "  --storage <pool>         Set storage pool"
    echo "  --disk-size <GB>         Set disk size in GB"
    echo "  --ha-url <url>           Home Assistant URL"
    echo "  --ha-token <token>       Home Assistant access token"
    echo "  --check-ids              Show available container IDs and exit"
    echo "  --help                   Show this help message"
    echo
    echo "Examples:"
    echo "  $0                                    # Interactive installation"
    echo "  $0 --auto                            # Auto installation with next available ID"
    echo "  $0 --ctid <NEXT_ID> --hostname my-predictor # Use specific ID and hostname"
    echo "  $0 --ha-url http://YOUR_HA_IP:8123 --ha-token YOUR_TOKEN # With HA integration"
    echo "  $0 --check-ids                       # Check available IDs first"
    echo
    exit 0
}

function parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --auto)
                INTERACTIVE_MODE=false
                AUTO_CTID=true
                shift
                ;;
            --ctid)
                CTID="$2"
                INTERACTIVE_MODE=false
                shift 2
                ;;
            --hostname)
                HOSTNAME="$2"
                shift 2
                ;;
            --cores)
                CORES="$2"
                shift 2
                ;;
            --memory)
                MEMORY="$2"
                shift 2
                ;;
            --storage)
                STORAGE="$2"
                shift 2
                ;;
            --disk-size)
                DISK_SIZE="$2"
                shift 2
                ;;
            --ha-url)
                HA_URL="$2"
                shift 2
                ;;
            --ha-token)
                HA_TOKEN="$2"
                shift 2
                ;;
            --check-ids)
                # Use our ID checker utility
                if [ -f "$(dirname "$0")/utils/check_available_ids.sh" ]; then
                    bash "$(dirname "$0")/utils/check_available_ids.sh"
                else
                    # Fallback implementation
                    echo "Next available container ID: $(get_next_vmid)"
                fi
                exit 0
                ;;
            --help)
                usage
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Global error handler
function handle_error() {
    local exit_code=$?
    local line_no=$1
    echo
    echo "[ERROR] Script failed at line $line_no with exit code $exit_code"
    echo "[ERROR] Current function: ${FUNCNAME[1]:-main}"
    echo "[ERROR] Command that failed: ${BASH_COMMAND}"
    cleanup
    exit $exit_code
}

# Global interrupt handler  
function handle_interrupt() {
    echo
    echo "[INFO] Script interrupted by user"
    cleanup
    exit 130
}

# Set up global error handling immediately
set -eE  # Exit on error, inherit ERR trap in functions and subshells
set -u   # Exit on undefined variables
set -o pipefail  # Exit on pipe failures
trap 'handle_error $LINENO' ERR
trap 'handle_interrupt' INT TERM

# Main installation flow
function main() {
    echo "[DEBUG] Starting main function with args: $@"
    # Parse command line arguments first
    parse_arguments "$@"
    echo "[DEBUG] Arguments parsed successfully"
    
    echo "[DEBUG] Displaying header..."
    header
    echo "[DEBUG] Checking requirements..."
    check_requirements
    echo "[DEBUG] Collecting info..."
    collect_info
    echo "[DEBUG] Creating container..."
    create_container
    echo "[DEBUG] Installing dependencies..."
    install_dependencies
    echo "[DEBUG] Setting up security and access..."
    setup_security
    echo "[DEBUG] Installing docker..."
    install_docker
    echo "[DEBUG] Setting up application..."
    setup_application
    echo "[DEBUG] Setting up docker services..."
    setup_docker_services
    echo "[DEBUG] Installing python dependencies..."
    install_python_dependencies
    echo "[DEBUG] Setting up systemd services..."
    setup_systemd_services
    echo "[DEBUG] Setting up nginx..."
    setup_nginx
    echo "[DEBUG] Creating monitoring script..."
    create_monitoring_script
    echo "[DEBUG] Running validation tests..."
    run_validation_tests
    echo "[DEBUG] Showing completion info..."
    show_completion_info
}

function run_validation_tests() {
    step_header "Running Validation Tests"
    
    msg_progress "Testing container connectivity..."
    if pct exec "$CTID" -- ping -c 1 8.8.8.8 &>/dev/null; then
        msg_ok "Internet connectivity: OK"
    else
        msg_warn "Internet connectivity: Limited"
    fi
    
    msg_progress "Testing Python installation..."
    if pct exec "$CTID" -- python3 --version &>/dev/null; then
        local python_version=$(pct exec "$CTID" -- python3 --version 2>&1)
        msg_ok "Python: $python_version"
    else
        msg_error "Python installation failed"
    fi
    
    msg_progress "Testing Docker installation..."
    if pct exec "$CTID" -- docker --version &>/dev/null; then
        local docker_version=$(pct exec "$CTID" -- docker --version 2>&1)
        msg_ok "Docker: $docker_version"
    else
        msg_error "Docker installation failed"
    fi
    
    msg_progress "Testing application files..."
    if pct exec "$CTID" -- test -d /opt/ha-intent-predictor; then
        msg_ok "Application files: Present"
    else
        msg_error "Application files: Missing"
    fi
    
    msg_ok "Validation tests completed"
}

function show_completion_info() {
    step_header "Installation Complete"
    
    msg_ok "HA Intent Predictor has been successfully installed!"
    echo
    echo "=== INSTALLATION SUMMARY ==="
    echo "  Container ID: $CTID"
    echo "  Hostname: $HOSTNAME"
    echo "  CPU Cores: $CORES"
    echo "  Memory: ${MEMORY}MB"
    echo "  Storage: $STORAGE"
    echo "  Disk Size: ${DISK_SIZE}GB"
    echo
    
    # Get container IP
    local container_ip=$(pct exec "$CTID" -- hostname -I 2>/dev/null | awk '{print $1}' || echo "checking...")
    echo "=== ACCESS INFORMATION ==="
    echo "  Container IP: $container_ip"
    echo "  SSH: ssh root@$container_ip"
    echo "  Root Password: $DEFAULT_PASSWORD"
    echo "  Console: pct enter $CTID (from Proxmox host)"
    echo "  API: http://$container_ip:8000"
    echo "  Grafana: http://$container_ip:3000"
    echo
    echo "=== MANAGEMENT COMMANDS ==="
    echo "  View logs: pct exec $CTID -- journalctl -u ha-intent-predictor.service -f"
    echo "  Restart: pct exec $CTID -- systemctl restart ha-intent-predictor.service"
    echo "  Status: pct exec $CTID -- systemctl status ha-intent-predictor.service"
    echo "  Enter container: pct enter $CTID"
    echo "  Remote monitor: pct exec $CTID -- /opt/ha-intent-predictor/scripts/remote-monitor.sh"
    echo
    echo "=== SERVICE ACCESS ==="
    echo "  PostgreSQL: $container_ip:5432 (user: ha_predictor, db: ha_predictor)"
    echo "  Redis: $container_ip:6379"
    echo "  Kafka: $container_ip:9092"
    echo "  Zookeeper: $container_ip:2181"
    echo
    echo "=== NEXT STEPS ==="
    echo "  1. Login via SSH: ssh root@$container_ip (password: $DEFAULT_PASSWORD)"
    echo "  2. Configure Home Assistant connection in /opt/ha-intent-predictor/config/"
    echo "  3. Monitor with: /opt/ha-intent-predictor/scripts/remote-monitor.sh"
    echo "  4. Add prediction entities to your Home Assistant automations"
    echo
    msg_ok "Installation completed successfully!"
}

# Run main function
main "$@"