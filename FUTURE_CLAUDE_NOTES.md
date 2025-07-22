# Notes for Future Claude Sessions

## 🚨 CRITICAL DEVELOPMENT RULES (User's Absolute Requirements)

### **NEVER MODIFY CODE WITHOUT EXPLICIT PERMISSION**
- **ALWAYS ask before modifying existing functions, classes, or parameters**
- **NEVER simplify, alter, or "improve" code without user approval**
- **NEVER create placeholders or simplified versions** - user hates this!
- **NEVER use hardcoded values** - always use configuration
- **Only fix import errors and references when explicitly requested**
- **Preserve ALL existing functionality and architecture**

### **ALWAYS CHECK IF SOMETHING EXISTS FIRST**
- **User says "almost 100% is there"** - always search for existing implementations
- Use `find`, `ls`, `grep`, `Glob`, or `Read` tools to check existing files
- Check repository structure before implementing anything
- **Avoid duplicating existing functionality**
- **Build upon existing codebase, don't recreate**

### **READ CLAUDE.md AFTER COMPACTING**
- **CLAUDE.md contains the master implementation guide**
- All development must follow CLAUDE.md specifications exactly
- If confused about anything, check CLAUDE.md first

## 🖥️ DEPLOYMENT & ACCESS INFO

### **Container Details**
- **Location**: Proxmox LXC Container ID 200
- **IP Address**: `192.168.51.10`
- **SSH Access**: `ssh ha-predictor` (key-based, no password needed)
- **SSH Key**: `~/.ssh/ha-predictor`
- **Container User**: `root`
- **Backup Password**: `hapredictor123` (use key auth preferred)

### **Service Endpoints**
- **PostgreSQL**: `192.168.51.10:5432` (user: `ha_predictor`, db: `ha_predictor`)
- **Redis**: `192.168.51.10:6379`
- **Kafka**: `192.168.51.10:9092`
- **Zookeeper**: `192.168.51.10:2181`
- **Web Health**: `http://192.168.51.10/health`

### **CRITICAL: Always Work in Container's Virtual Environment**
```bash
ssh ha-predictor 'cd /opt/ha-intent-predictor && source venv/bin/activate && [your command]'
```

## 🔧 HOW TO DEPLOY CHANGES

### **Standard Workflow**
1. **Make changes locally**
2. **Commit to git**:
   ```bash
   git add [files]
   git commit -m "Description

   🤖 Generated with [Claude Code](https://claude.ai/code)
   
   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```
3. **Push to GitHub**:
   ```bash
   git push origin working-fixes
   ```
4. **Pull to container**:
   ```bash
   ssh ha-predictor 'cd /opt/ha-intent-predictor && git pull origin working-fixes'
   ```

### **Testing Changes**
- Always test in container's virtual environment
- Use existing scripts and tools
- Check logs: `ssh ha-predictor 'cd /opt/ha-intent-predictor && tail -f main_app_fixed.log'`

## 📁 KEY FILE LOCATIONS & NAMING

### **Import Issues - COMMON PROBLEM**
The user frequently encounters import errors due to wrong class names:

**WRONG NAMES** → **CORRECT NAMES**:
- `TimeSeriesDB` → `TimescaleDBManager`
- `FeatureStore` → `RedisFeatureStore`  
- `DataEnricher` → `DynamicFeatureDiscovery`

**Always use the correct class names** - they exist, just with different names!

### **Configuration Files**
- **Main Config**: `config/system.yaml` (in container only, gitignored locally)
- **Config Loader**: `config/config_loader.py` (exists in container)
- **Note**: `config/` directory is in `.gitignore` but exists on container

### **Bootstrap Process**
- **Complete Bootstrap**: `scripts/bootstrap_complete.py` (merged from both bootstrap files)
- **Historical Import**: `scripts/historical_import.py` (fixed imports)
- **Simple Import**: `scripts/simple_historical_import.py` (working fallback)

## 🏠 HOME ASSISTANT INTEGRATION

### **User's Setup**
- **HA URL**: `http://192.168.51.247:8123`
- **Timezone**: Europe/Bucharest (GMT+3)
- **HA sends timestamps in UTC** (already correct format)
- **People**: Anca & Vladimir (person-specific learning required)
- **Combined Space**: Living room + Kitchen = `living_kitchen`
- **Rooms**: bedroom, office, bathroom, small_bathroom, guest_bedroom

### **Sensor Configuration**
- **~98 sensors total** across presence, doors, climate, light
- **Presence sensors**: Full zones + subzones (dual system)
- **Person-specific zones**: anca_bed_side, vladimir_bed_side, anca_desk, vladimir_desk
- **Door sensors**: bathroom, bedroom, office, guest_bedroom, small_bathroom

## 🤖 ML SYSTEM ARCHITECTURE

### **Core Philosophy (from CLAUDE.md)**
- **NO ASSUMPTIONS** about patterns or schedules
- **Pure data-driven learning** from 180 days of historical data
- **Adaptive models** that update with every observation
- **Person-specific learning** (Anca vs Vladimir)
- **Cat detection** without hardcoded assumptions
- **Multi-zone analysis** using full zones + subzones

### **Key Components (All Exist)**
- **TimescaleDBManager**: Time-series database operations
- **RedisFeatureStore**: Real-time feature caching
- **AdaptiveOccupancyPredictor**: Main ML prediction engine
- **PatternDiscovery**: Discovers patterns without assumptions
- **AdaptiveCatDetector**: Learns cat vs human movement
- **DynamicHAIntegration**: Home Assistant entity management

### **System Workflow**
1. **Bootstrap**: `bootstrap_complete.py` (sets up infrastructure + imports 180 days)
2. **Main System**: `main.py` (continuous real-time operation)
3. **Data Flow**: HA → Kafka → ML Models → Predictions → HA entities

## 🐛 COMMON ISSUES & SOLUTIONS

### **Timezone Errors**
- **Problem**: "can't compare offset-naive and offset-aware datetimes"
- **Solution**: Always use `datetime.now(timezone.utc)` not `datetime.now()`
- **User said**: This is CRITICAL, not minor - will massively skew predictions

### **Import Errors**
- **Problem**: `cannot import name 'TimeSeriesDB'`
- **Solution**: Use correct names: `TimescaleDBManager`, `RedisFeatureStore`, etc.
- **Check first**: Use `Grep` to find actual class names

### **Missing Historical Data**
- **Problem**: Models show 0.0 accuracy, immediate drift detection
- **Root Cause**: No 6 months of historical data for bootstrapping
- **Solution**: Run bootstrap process first, then main system

### **Port Already in Use**
- **Problem**: `address already in use ('0.0.0.0', 8000)`
- **Check**: `ssh ha-predictor 'lsof -i :8000'` to see what's using port
- **Often**: Previous instance still running

## 💬 USER'S COMMUNICATION STYLE

### **What User Says vs Means**
- **"simple fix"** = Usually right, check for typos/naming issues first
- **"everything exists"** = Don't recreate, find the existing implementation
- **"never simplify"** = Use full existing implementations, no shortcuts
- **"check if it already exists"** = ALWAYS search before creating
- **"stop"** = Stop immediately, you're probably breaking something

### **User Gets Frustrated When**
- You create simplified/placeholder versions of existing code
- You don't check if something already exists
- You modify code without permission
- You ignore CLAUDE.md specifications
- You hardcode values instead of using config

### **User Is Happy When**
- You find and use existing implementations
- You ask permission before modifying code
- You follow CLAUDE.md exactly
- You work in the container's virtual environment
- You commit with proper git messages

## 🔍 DEBUGGING COMMANDS

### **System Status**
```bash
ssh ha-predictor '/opt/ha-intent-predictor/scripts/remote-monitor.sh show'
```

### **Service Logs**
```bash
ssh ha-predictor 'cd /opt/ha-intent-predictor && tail -f main_app_fixed.log'
ssh ha-predictor 'cd /opt/ha-intent-predictor && docker compose logs --tail 20'
```

### **Process Status**
```bash
ssh ha-predictor 'ps aux | grep main.py'
ssh ha-predictor 'ps aux | grep python'
```

### **Database Check**
```bash
ssh ha-predictor 'cd /opt/ha-intent-predictor && source venv/bin/activate && python -c "from src.storage.timeseries_db import TimescaleDBManager; print(\"✓ TimescaleDB accessible\")"'
```

## 🎯 REMEMBER THESE PATTERNS

### **When User Says "start it up"**
1. Check if already running: `ps aux | grep main.py`
2. If not running: `ssh ha-predictor 'cd /opt/ha-intent-predictor && source venv/bin/activate && python main.py'`
3. Monitor logs and report status

### **When User Reports Errors**
1. Check logs first: `tail -f main_app_fixed.log`
2. Look for import errors (common)
3. Check if using wrong class names
4. Verify virtual environment is activated
5. **Always ask before fixing** - don't assume

### **When User Wants New Features**
1. **Read CLAUDE.md first** to understand architecture
2. **Search for existing implementations** using `Grep`/`Glob`
3. **Ask permission** before modifying anything
4. **Use existing components** - don't recreate
5. **Test in container** virtual environment

## 📝 FINAL REMINDERS

- **User values working implementations and perfect code**
- **The config directory exists in container but not locally (gitignored)**
- **Historical data import is CRITICAL for ML models to work**
- **Person-specific learning (Anca & Vladimir) is a key requirement**
- **No hardcoded patterns - everything learned from data**
- **When in doubt, ask before modifying**

---

*This document should be your first reference for understanding the user's requirements and the system architecture. When you're confused, read CLAUDE.md and this file.*