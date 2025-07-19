#!/usr/bin/env bash

# HA Intent Predictor - Proxmox LXC Installation Script
# Based on Proxmox Community Scripts architecture
# Usage: bash -c "$(wget -qLO - https://github.com/yourusername/ha-intent-predictor/raw/main/deploy/proxmox-install.sh)"

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Safety tracking
CONTAINER_CREATED="false"

# HA Connection details
HA_URL=""
HA_TOKEN=""

function header() {
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

function msg_info() {
    local msg="$1"
    echo -e "${BLUE}[INFO]${NC} $msg"
}

function msg_ok() {
    local msg="$1"
    echo -e "${GREEN}[OK]${NC} $msg"
}

function msg_error() {
    local msg="$1"
    echo -e "${RED}[ERROR]${NC} $msg"
}

function msg_warn() {
    local msg="$1"
    echo -e "${YELLOW}[WARN]${NC} $msg"
}

function cleanup() {
    # Only cleanup if we created the container in this session
    if [ -n "$CTID" ] && [ "$CONTAINER_CREATED" = "true" ] && pct status "$CTID" &>/dev/null; then
        msg_warn "Cleaning up failed installation for container $CTID..."
        
        # Double-check this is our container by checking if it has our hostname
        local container_hostname=$(pct exec "$CTID" -- hostname 2>/dev/null || echo "")
        if [ "$container_hostname" = "$HOSTNAME" ]; then
            msg_info "Stopping and removing container $CTID (hostname: $container_hostname)"
            pct stop "$CTID" 2>/dev/null || true
            pct destroy "$CTID" 2>/dev/null || true
            msg_ok "Cleanup completed"
        else
            msg_error "Safety check failed: Container $CTID hostname '$container_hostname' does not match expected '$HOSTNAME'"
            msg_error "Manual cleanup required. Container $CTID was NOT removed for safety."
        fi
    elif [ -n "$CTID" ]; then
        msg_info "No cleanup needed - container $CTID was not created in this session"
    fi
}

function check_requirements() {
    msg_info "Checking Proxmox requirements..."
    
    # Check if running on Proxmox
    if ! command -v pct &> /dev/null; then
        msg_error "This script must be run on a Proxmox server"
        exit 1
    fi
    
    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        msg_error "This script must be run as root"
        exit 1
    fi
    
    # Check for required commands
    local required_commands=("pvesh" "jq" "curl" "wget" "bc")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            msg_error "Required command '$cmd' not found. Please install it first."
            exit 1
        fi
    done
    
    # Check available storage
    local available_storage=$(pvs --noheadings -o pv_free --units g | head -1 | tr -d ' G' 2>/dev/null || echo "0")
    if (( $(echo "$available_storage < 50" | bc -l) )); then
        msg_warn "Low storage space detected. Available: ${available_storage}GB, Recommended: 50GB+"
    fi
    
    # Auto-detect available Ubuntu template
    detect_ubuntu_template
    
    msg_ok "Requirements check passed"
}

function detect_ubuntu_template() {
    msg_info "Detecting latest available Ubuntu template..."
    
    # Check local storage for Ubuntu 22.04 templates (prefer latest version)
    local available_templates=$(pveam list local 2>/dev/null | grep -E "ubuntu-22\.04.*standard.*amd64\.tar\.zst" | awk '{print $2}' | sort -V | tail -1)
    
    if [ -n "$available_templates" ]; then
        TEMPLATE="$available_templates"
        msg_ok "Found latest Ubuntu 22.04 template: $TEMPLATE"
        return 0
    fi
    
    # If Ubuntu 22.04 not found, look for any Ubuntu LTS template
    msg_info "Ubuntu 22.04 not found, searching for any Ubuntu LTS template..."
    available_templates=$(pveam list local 2>/dev/null | grep -E "ubuntu-[0-9]+\.[0-9]+.*standard.*amd64\.tar\.zst" | awk '{print $2}' | sort -V | tail -1)
    
    if [ -n "$available_templates" ]; then
        TEMPLATE="$available_templates"
        msg_ok "Found latest Ubuntu template: $TEMPLATE"
        return 0
    fi
    
    # Last resort - any Ubuntu template
    msg_info "Searching for any available Ubuntu template..."
    available_templates=$(pveam list local 2>/dev/null | grep -E "ubuntu.*amd64\.tar\.zst" | awk '{print $2}' | sort -V | tail -1)
    
    if [ -n "$available_templates" ]; then
        TEMPLATE="$available_templates"
        msg_warn "Using available Ubuntu template: $TEMPLATE"
        return 0
    fi
    
    # No Ubuntu templates found locally
    msg_error "No Ubuntu templates found in local storage."
    msg_error "Available templates:"
    pveam list local 2>/dev/null | grep -E "\.tar\.zst" | awk '{print "  - " $2}'
    msg_error ""
    msg_error "Please download an Ubuntu template first:"
    msg_error "pveam available --section system | grep ubuntu"
    msg_error "pveam download local <ubuntu-template-name>"
    exit 1
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
        ((vmid++))
        ((attempts++))
    done
    
    msg_error "Could not find available VMID after $max_attempts attempts"
    exit 1
}

function collect_info() {
    header
    
    msg_info "Collecting installation parameters..."
    
    # Get CTID
    local suggested_ctid=$(get_next_vmid)
    read -p "Enter CT ID (default: $suggested_ctid): " input_ctid
    CTID=${input_ctid:-$suggested_ctid}
    
    # Comprehensive CTID validation
    if pct status "$CTID" &>/dev/null; then
        msg_error "LXC Container ID $CTID already exists"
        exit 1
    fi
    
    if qm status "$CTID" &>/dev/null; then
        msg_error "VM ID $CTID already exists"
        exit 1
    fi
    
    # Final check using cluster resources
    if pvesh get /cluster/resources --type vm --output-format json | jq -r '.[].vmid' | grep -q "^${CTID}$" 2>/dev/null; then
        msg_error "VM/Container ID $CTID is already in use according to cluster resources"
        exit 1
    fi
    
    msg_ok "Container ID $CTID is available"
    
    # Get hostname
    read -p "Enter hostname (default: $HOSTNAME): " input_hostname
    HOSTNAME=${input_hostname:-$HOSTNAME}
    
    # Get cores
    read -p "Enter CPU cores (default: $CORES): " input_cores
    CORES=${input_cores:-$CORES}
    
    # Get memory
    read -p "Enter RAM in MB (default: $MEMORY): " input_memory
    MEMORY=${input_memory:-$MEMORY}
    
    # Get storage
    msg_info "Available storage pools:"
    pvesm status | grep -E "(local-lvm|local-zfs|local)" | awk '{print "  - " $1 " (" $2 ")"}'
    read -p "Enter storage pool (default: $STORAGE): " input_storage
    STORAGE=${input_storage:-$STORAGE}
    
    # Get disk size
    read -p "Enter disk size in GB (default: $DISK_SIZE): " input_disk_size
    DISK_SIZE=${input_disk_size:-$DISK_SIZE}
    
    # Get HA details
    echo
    msg_info "Home Assistant integration setup:"
    read -p "Enter Home Assistant URL (e.g., http://YOUR_HA_IP:8123): " HA_URL
    read -s -p "Enter Home Assistant Long-Lived Access Token: " HA_TOKEN
    echo
    
    # Validate HA connection
    if [ -n "$HA_URL" ] && [ -n "$HA_TOKEN" ]; then
        msg_info "Testing Home Assistant connection..."
        if curl -s -H "Authorization: Bearer $HA_TOKEN" "$HA_URL/api/" > /dev/null; then
            msg_ok "Home Assistant connection successful"
        else
            msg_warn "Could not connect to Home Assistant. You can configure this later."
        fi
    fi
    
    # Show summary
    echo
    msg_info "Installation Summary:"
    echo "  Container ID: $CTID"
    echo "  Hostname: $HOSTNAME"
    echo "  CPU Cores: $CORES"
    echo "  Memory: ${MEMORY}MB"
    echo "  Storage: $STORAGE"
    echo "  Disk Size: ${DISK_SIZE}GB"
    echo "  HA URL: $HA_URL"
    echo
    
    read -p "Proceed with installation? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        msg_info "Installation cancelled"
        exit 0
    fi
}

function create_container() {
    msg_info "Creating LXC container..."
    
    # Final safety check before creation
    if pct status "$CTID" &>/dev/null || qm status "$CTID" &>/dev/null; then
        msg_error "CRITICAL: Container/VM $CTID exists just before creation. Aborting to prevent data loss."
        exit 1
    fi
    
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
        --start "$START_AFTER_CREATE"; then
        
        # Mark that we successfully created this container
        CONTAINER_CREATED="true"
        msg_ok "Container created successfully"
    else
        msg_error "Failed to create container $CTID"
        exit 1
    fi
    
    # Wait for container to be ready
    msg_info "Waiting for container to be ready..."
    sleep 10
    
    # Check if container is running
    local max_wait=60
    local wait_time=0
    while [ $wait_time -lt $max_wait ]; do
        if pct status "$CTID" | grep -q "running"; then
            msg_ok "Container is running"
            return 0
        fi
        sleep 5
        wait_time=$((wait_time + 5))
        msg_info "Waiting for container to start... ($wait_time/${max_wait}s)"
    done
    
    msg_error "Container failed to start within $max_wait seconds"
    msg_error "Container status: $(pct status "$CTID" 2>/dev/null || echo 'unknown')"
    exit 1
}

function install_dependencies() {
    msg_info "Installing system dependencies..."
    
    # Update system
    pct exec "$CTID" -- bash -c "apt-get update && apt-get upgrade -y"
    
    # Install base packages
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
        unzip"
    
    msg_ok "System dependencies installed"
}

function install_docker() {
    msg_info "Installing Docker..."
    
    # Install Docker
    pct exec "$CTID" -- bash -c "
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
        echo \"deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \$(lsb_release -cs) stable\" | tee /etc/apt/sources.list.d/docker.list > /dev/null
        apt-get update
        apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        systemctl enable docker
        systemctl start docker
    "
    
    msg_ok "Docker installed"
}

function setup_application() {
    msg_info "Setting up HA Intent Predictor application..."
    
    # Create application directory
    pct exec "$CTID" -- bash -c "mkdir -p /opt/ha-intent-predictor"
    
    # Copy application files
    pct push "$CTID" "$(dirname "$0")/../" "/opt/ha-intent-predictor/" --recursive
    
    # Create configuration from template
    pct exec "$CTID" -- bash -c "
        cd /opt/ha-intent-predictor
        
        # Create config directory
        mkdir -p config/
        
        # Generate configuration
        cat > config/app.yaml << EOF
app:
  name: ha-intent-predictor
  version: 1.0.0
  debug: false
  
database:
  type: postgresql
  host: localhost
  port: 5432
  name: ha_predictor
  user: ha_predictor
  password: $(openssl rand -base64 32)
  
redis:
  host: localhost
  port: 6379
  db: 0
  
kafka:
  bootstrap_servers: localhost:9092
  topic_prefix: ha_predictor
  
home_assistant:
  url: \"$HA_URL\"
  token: \"$HA_TOKEN\"
  
logging:
  level: INFO
  file: /var/log/ha-intent-predictor/app.log
  
ml:
  model_update_interval: 300
  prediction_horizons: [15, 30, 60, 120]
  feature_cache_ttl: 300
  
rooms:
  - living_kitchen
  - bedroom
  - office
  - bathroom
  - small_bathroom
  - guest_bedroom
EOF
    "
    
    msg_ok "Application setup completed"
}

function setup_docker_services() {
    msg_info "Setting up Docker services..."
    
    # Create docker-compose.yml
    pct exec "$CTID" -- bash -c "
        cd /opt/ha-intent-predictor
        
        cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:2.11.0-pg15
    container_name: ha-predictor-postgres
    environment:
      POSTGRES_DB: ha_predictor
      POSTGRES_USER: ha_predictor
      POSTGRES_PASSWORD: \${POSTGRES_PASSWORD}
      POSTGRES_HOST_AUTH_METHOD: trust
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - \"5432:5432\"
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

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    container_name: ha-predictor-kafka
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
    ports:
      - \"9092:9092\"
    depends_on:
      - zookeeper
    volumes:
      - kafka_data:/var/lib/kafka/data
    restart: unless-stopped

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

  grafana:
    image: grafana/grafana:10.0.0
    container_name: ha-predictor-grafana
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: \${GRAFANA_PASSWORD}
      GF_USERS_ALLOW_SIGN_UP: false
    ports:
      - \"3000:3000\"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana:/etc/grafana/provisioning
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  kafka_data:
  zookeeper_data:
  zookeeper_logs:
  grafana_data:

networks:
  default:
    name: ha-predictor-network
EOF
    "
    
    # Create .env file
    pct exec "$CTID" -- bash -c "
        cd /opt/ha-intent-predictor
        
        cat > .env << EOF
POSTGRES_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 16)
EOF
    "
    
    # Create init SQL
    pct exec "$CTID" -- bash -c "
        mkdir -p /opt/ha-intent-predictor/scripts
        cat > /opt/ha-intent-predictor/scripts/init.sql << 'EOF'
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create tables
CREATE TABLE IF NOT EXISTS sensor_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    state VARCHAR(50),
    attributes JSONB,
    room VARCHAR(100),
    sensor_type VARCHAR(100),
    zone_info JSONB
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100) NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    probability FLOAT NOT NULL,
    uncertainty FLOAT NOT NULL,
    model_info JSONB,
    features JSONB
);

CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metadata JSONB
);

-- Create hypertables for time-series data
SELECT create_hypertable('sensor_events', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('predictions', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('model_performance', 'timestamp', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_sensor_events_entity_time ON sensor_events (entity_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_sensor_events_room_time ON sensor_events (room, timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_room_time ON predictions (room, timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_horizon ON predictions (horizon_minutes);
EOF
    "
    
    msg_ok "Docker services configuration created"
}

function start_services() {
    msg_info "Starting Docker services..."
    
    pct exec "$CTID" -- bash -c "
        cd /opt/ha-intent-predictor
        docker compose up -d
        
        # Wait for services to be ready
        sleep 30
        
        # Check service health
        docker compose ps
    "
    
    msg_ok "Docker services started"
}

function setup_python_application() {
    msg_info "Setting up Python application..."
    
    pct exec "$CTID" -- bash -c "
        cd /opt/ha-intent-predictor
        
        # Create virtual environment
        python3 -m venv venv
        source venv/bin/activate
        
        # Install Python dependencies
        pip install --upgrade pip
        pip install -r requirements.txt
        
        # Create log directory
        mkdir -p /var/log/ha-intent-predictor
        
        # Set permissions
        chown -R root:root /opt/ha-intent-predictor
        chmod +x /opt/ha-intent-predictor/scripts/*.py
    "
    
    msg_ok "Python application setup completed"
}

function setup_systemd_services() {
    msg_info "Setting up systemd services..."
    
    # Create systemd service files
    pct exec "$CTID" -- bash -c "
        # Main application service
        cat > /etc/systemd/system/ha-intent-predictor.service << 'EOF'
[Unit]
Description=HA Intent Predictor - Main Application
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ha-intent-predictor
Environment=PATH=/opt/ha-intent-predictor/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/opt/ha-intent-predictor/venv/bin/python -m src.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

        # Data ingestion service
        cat > /etc/systemd/system/ha-predictor-ingestion.service << 'EOF'
[Unit]
Description=HA Intent Predictor - Data Ingestion
After=network.target docker.service ha-intent-predictor.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ha-intent-predictor
Environment=PATH=/opt/ha-intent-predictor/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/opt/ha-intent-predictor/venv/bin/python -m src.ingestion.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

        # Model training service
        cat > /etc/systemd/system/ha-predictor-training.service << 'EOF'
[Unit]
Description=HA Intent Predictor - Model Training
After=network.target docker.service ha-intent-predictor.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ha-intent-predictor
Environment=PATH=/opt/ha-intent-predictor/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/opt/ha-intent-predictor/venv/bin/python -m src.learning.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

        # Enable services
        systemctl daemon-reload
        systemctl enable ha-intent-predictor.service
        systemctl enable ha-predictor-ingestion.service
        systemctl enable ha-predictor-training.service
    "
    
    msg_ok "Systemd services configured"
}

function setup_nginx() {
    msg_info "Setting up Nginx reverse proxy..."
    
    pct exec "$CTID" -- bash -c "
        # Create nginx configuration
        cat > /etc/nginx/sites-available/ha-intent-predictor << 'EOF'
server {
    listen 80;
    server_name _;
    
    # Main API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Grafana
    location /grafana/ {
        proxy_pass http://localhost:3000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Health check
    location /health {
        return 200 'OK';
        add_header Content-Type text/plain;
    }
}
EOF
        
        # Enable site
        ln -sf /etc/nginx/sites-available/ha-intent-predictor /etc/nginx/sites-enabled/
        rm -f /etc/nginx/sites-enabled/default
        
        # Test configuration
        nginx -t
        
        # Restart nginx
        systemctl restart nginx
        systemctl enable nginx
    "
    
    msg_ok "Nginx configured"
}

function import_historical_data() {
    msg_info "Starting historical data import..."
    
    if [ -n "$HA_URL" ] && [ -n "$HA_TOKEN" ]; then
        pct exec "$CTID" -- bash -c "
            cd /opt/ha-intent-predictor
            source venv/bin/activate
            
            # Run historical data import
            python scripts/historical_import.py --days 180 --batch-size 1000
        "
        msg_ok "Historical data import completed"
    else
        msg_warn "Skipping historical data import - HA credentials not provided"
    fi
}

function final_setup() {
    msg_info "Performing final setup..."
    
    # Start services
    pct exec "$CTID" -- bash -c "
        systemctl start ha-intent-predictor.service
        systemctl start ha-predictor-ingestion.service
        systemctl start ha-predictor-training.service
        
        # Wait for services to start
        sleep 10
        
        # Check service status
        systemctl status ha-intent-predictor.service --no-pager
        systemctl status ha-predictor-ingestion.service --no-pager
        systemctl status ha-predictor-training.service --no-pager
    "
    
    msg_ok "Services started successfully"
}

function show_completion_info() {
    clear
    header
    
    msg_ok "Installation completed successfully!"
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    echo -e "${GREEN}🎉 HA Intent Predictor is now running!${NC}"
    echo
    echo "📋 Container Details:"
    echo "   • Container ID: $CTID"
    echo "   • Hostname: $HOSTNAME"
    echo "   • IP Address: $(pct exec "$CTID" -- hostname -I | tr -d ' ')"
    echo
    echo "🔗 Access URLs:"
    echo "   • API: http://$(pct exec "$CTID" -- hostname -I | tr -d ' ')/api/"
    echo "   • Grafana: http://$(pct exec "$CTID" -- hostname -I | tr -d ' ')/grafana/"
    echo "   • Health: http://$(pct exec "$CTID" -- hostname -I | tr -d ' ')/health"
    echo
    echo "📊 Services:"
    echo "   • PostgreSQL + TimescaleDB: Port 5432"
    echo "   • Redis: Port 6379"
    echo "   • Kafka: Port 9092"
    echo "   • Grafana: Port 3000"
    echo "   • API: Port 8000"
    echo
    echo "🔧 Management Commands:"
    echo "   • View logs: pct exec $CTID -- journalctl -u ha-intent-predictor.service -f"
    echo "   • Restart services: pct exec $CTID -- systemctl restart ha-intent-predictor.service"
    echo "   • Check status: pct exec $CTID -- systemctl status ha-intent-predictor.service"
    echo "   • Enter container: pct enter $CTID"
    echo
    echo "📝 Next Steps:"
    echo "   1. Configure your Home Assistant sensors in /opt/ha-intent-predictor/config/sensors.yaml"
    echo "   2. Monitor the learning progress in Grafana"
    echo "   3. Add prediction entities to your Home Assistant automations"
    echo
    echo "📚 Documentation:"
    echo "   • Configuration: /opt/ha-intent-predictor/config/"
    echo "   • Logs: /var/log/ha-intent-predictor/"
    echo "   • GitHub: https://github.com/yourusername/ha-intent-predictor"
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    echo -e "${BLUE}The system is now learning from your Home Assistant data.${NC}"
    echo -e "${BLUE}Initial predictions will be available within 30 minutes.${NC}"
    echo
}

# Main installation flow
function main() {
    trap cleanup EXIT
    
    header
    check_requirements
    collect_info
    create_container
    install_dependencies
    install_docker
    setup_application
    setup_docker_services
    start_services
    setup_python_application
    setup_systemd_services
    setup_nginx
    import_historical_data
    final_setup
    show_completion_info
}

# Run main function
main "$@"