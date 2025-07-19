#!/usr/bin/env bash

# HA Intent Predictor - Check Available Container IDs
# Shows current container/VM usage and suggests next available ID

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

function header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                        PROXMOX ID AVAILABILITY CHECKER                       ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo
}

function get_next_vmid() {
    local vmid=100
    local max_vmid=999999
    
    # Find the actual next available ID starting from 100
    while [ $vmid -le $max_vmid ]; do
        if ! pct status "$vmid" &>/dev/null && ! qm status "$vmid" &>/dev/null; then
            echo "$vmid"
            return 0
        fi
        ((vmid++))
    done
    
    echo "none"
}

function show_containers() {
    echo -e "${BLUE}=== EXISTING CONTAINERS ===${NC}"
    
    if command -v pct &>/dev/null; then
        local containers=$(pct list 2>/dev/null | tail -n +2 || echo "")
        if [ -n "$containers" ]; then
            echo -e "${CYAN}VMID    Status    Name${NC}"
            echo "$containers" | while read line; do
                local vmid=$(echo "$line" | awk '{print $1}')
                local status=$(echo "$line" | awk '{print $2}')
                local name=$(echo "$line" | awk '{print $3}')
                
                case "$status" in
                    "running")
                        echo -e "${GREEN}$vmid${NC}      $status   $name"
                        ;;
                    "stopped")
                        echo -e "${YELLOW}$vmid${NC}      $status   $name"
                        ;;
                    *)
                        echo -e "${RED}$vmid${NC}      $status   $name"
                        ;;
                esac
            done
        else
            echo -e "${YELLOW}No containers found${NC}"
        fi
    else
        echo -e "${RED}pct command not found - not running on Proxmox?${NC}"
    fi
    echo
}

function show_vms() {
    echo -e "${BLUE}=== EXISTING VIRTUAL MACHINES ===${NC}"
    
    if command -v qm &>/dev/null; then
        local vms=$(qm list 2>/dev/null | tail -n +2 || echo "")
        if [ -n "$vms" ]; then
            echo -e "${CYAN}VMID    Status    Name${NC}"
            echo "$vms" | while read line; do
                local vmid=$(echo "$line" | awk '{print $1}')
                local status=$(echo "$line" | awk '{print $3}')
                local name=$(echo "$line" | awk '{print $2}')
                
                case "$status" in
                    "running")
                        echo -e "${GREEN}$vmid${NC}      $status   $name"
                        ;;
                    "stopped")
                        echo -e "${YELLOW}$vmid${NC}      $status   $name"
                        ;;
                    *)
                        echo -e "${RED}$vmid${NC}      $status   $name"
                        ;;
                esac
            done
        else
            echo -e "${YELLOW}No virtual machines found${NC}"
        fi
    else
        echo -e "${RED}qm command not found - not running on Proxmox?${NC}"
    fi
    echo
}

function show_id_ranges() {
    echo -e "${BLUE}=== ID RANGE USAGE ===${NC}"
    
    # Count usage in different ranges
    local range_100_199=0
    local range_200_299=0
    local range_300_999=0
    local range_1000_plus=0
    
    # Check containers
    if command -v pct &>/dev/null; then
        while read -r vmid status name; do
            if [ -n "$vmid" ] && [[ "$vmid" =~ ^[0-9]+$ ]]; then
                if [ "$vmid" -ge 100 ] && [ "$vmid" -le 199 ]; then
                    ((range_100_199++))
                elif [ "$vmid" -ge 200 ] && [ "$vmid" -le 299 ]; then
                    ((range_200_299++))
                elif [ "$vmid" -ge 300 ] && [ "$vmid" -le 999 ]; then
                    ((range_300_999++))
                elif [ "$vmid" -ge 1000 ]; then
                    ((range_1000_plus++))
                fi
            fi
        done < <(pct list 2>/dev/null | tail -n +2 | awk '{print $1, $2, $3}')
    fi
    
    # Check VMs
    if command -v qm &>/dev/null; then
        while read -r vmid name status; do
            if [ -n "$vmid" ] && [[ "$vmid" =~ ^[0-9]+$ ]]; then
                if [ "$vmid" -ge 100 ] && [ "$vmid" -le 199 ]; then
                    ((range_100_199++))
                elif [ "$vmid" -ge 200 ] && [ "$vmid" -le 299 ]; then
                    ((range_200_299++))
                elif [ "$vmid" -ge 300 ] && [ "$vmid" -le 999 ]; then
                    ((range_300_999++))
                elif [ "$vmid" -ge 1000 ]; then
                    ((range_1000_plus++))
                fi
            fi
        done < <(qm list 2>/dev/null | tail -n +2 | awk '{print $1, $2, $3}')
    fi
    
    echo -e "  ${CYAN}100-199${NC} (System):     $range_100_199 used"
    echo -e "  ${CYAN}200-299${NC} (User Range):  $range_200_299 used"
    echo -e "  ${CYAN}300-999${NC} (VMs):        $range_300_999 used"
    echo -e "  ${CYAN}1000+${NC}   (Custom):     $range_1000_plus used"
    echo
}

function show_next_available() {
    echo -e "${BLUE}=== NEXT AVAILABLE IDS ===${NC}"
    
    local next_id=$(get_next_vmid)
    if [ "$next_id" = "none" ]; then
        echo -e "${RED}No available IDs found!${NC}"
        echo -e "${YELLOW}Consider using higher ID ranges or cleaning up unused containers/VMs${NC}"
    else
        echo -e "  ${GREEN}Next available ID: $next_id${NC}"
        
        # Show next few available IDs
        local count=0
        local vmid=$next_id
        echo -e "  ${CYAN}Available IDs:${NC}"
        
        while [ $count -lt 10 ] && [ $vmid -le 299 ]; do
            if ! pct status "$vmid" &>/dev/null && ! qm status "$vmid" &>/dev/null; then
                echo -e "    $vmid"
                ((count++))
            fi
            ((vmid++))
        done
        
        if [ $count -eq 0 ]; then
            echo -e "    ${YELLOW}No available IDs in 100-299 range${NC}"
        fi
    fi
    echo
}

function check_specific_id() {
    local id="$1"
    
    echo -e "${BLUE}=== CHECKING ID $id ===${NC}"
    
    if ! [[ "$id" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}Invalid ID: must be numeric${NC}"
        return 1
    fi
    
    if [ "$id" -lt 100 ] || [ "$id" -gt 999999 ]; then
        echo -e "${YELLOW}Warning: ID $id is outside recommended range (100-999999)${NC}"
    fi
    
    local container_exists=false
    local vm_exists=false
    
    if pct status "$id" &>/dev/null; then
        container_exists=true
        local container_info=$(pct list | grep "^$id " || echo "")
        if [ -n "$container_info" ]; then
            local status=$(echo "$container_info" | awk '{print $2}')
            local name=$(echo "$container_info" | awk '{print $3}')
            echo -e "  ${RED}Container exists:${NC} ID $id ($status) - $name"
        else
            echo -e "  ${RED}Container exists:${NC} ID $id"
        fi
    fi
    
    if qm status "$id" &>/dev/null; then
        vm_exists=true
        local vm_info=$(qm list | grep "^$id " || echo "")
        if [ -n "$vm_info" ]; then
            local name=$(echo "$vm_info" | awk '{print $2}')
            local status=$(echo "$vm_info" | awk '{print $3}')
            echo -e "  ${RED}Virtual machine exists:${NC} ID $id ($status) - $name"
        else
            echo -e "  ${RED}Virtual machine exists:${NC} ID $id"
        fi
    fi
    
    if [ "$container_exists" = false ] && [ "$vm_exists" = false ]; then
        echo -e "  ${GREEN}Available:${NC} ID $id is free to use"
        return 0
    else
        echo -e "  ${RED}Unavailable:${NC} ID $id is already in use"
        return 1
    fi
}

function main() {
    # Check if running on Proxmox
    if ! command -v pct &>/dev/null && ! command -v qm &>/dev/null; then
        echo -e "${RED}Error: This script must be run on a Proxmox server${NC}"
        exit 1
    fi
    
    header
    
    # Parse arguments
    if [ $# -eq 1 ]; then
        # Check specific ID
        check_specific_id "$1"
        echo
        echo -e "${CYAN}Suggestion: Use 'bash $0' without arguments to see all available IDs${NC}"
    else
        # Show full overview
        show_containers
        show_vms
        show_id_ranges
        show_next_available
        
        echo -e "${CYAN}Usage:${NC}"
        echo -e "  $0           # Show this overview"
        echo -e "  $0 <id>      # Check if specific ID is available"
        echo -e "  $0 105       # Example: Check if ID 105 is available"
    fi
}

main "$@"