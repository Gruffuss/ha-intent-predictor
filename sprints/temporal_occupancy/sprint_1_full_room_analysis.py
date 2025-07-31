#!/usr/bin/env python3
"""
Sprint 1: Temporal Occupancy Prediction - Full Room Analysis
============================================================

SPRINT 1 OBJECTIVES:
1. Load all full room occupancy sensors from sensors.yaml
2. Implement data cleaning: remove on→on and off→off within 2 seconds  
3. Analyze cleaned data: transition counts, patterns, room activity levels
4. Generate sample timeline showing multi-room occupancy patterns

Full Room Sensors (from rooms.yaml):
- bedroom: binary_sensor.bedroom_presence_sensor_full_bedroom
- living_room: binary_sensor.presence_livingroom_full  
- kitchen: binary_sensor.kitchen_pressence_full_kitchen
- office: binary_sensor.office_presence_full_office

SUCCESS CRITERIA:
- Identify rooms with sufficient data for temporal prediction
- Clean understanding of multi-room occupancy dynamics
- Foundation for temporal feature engineering in Sprint 2
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict
import sys
import os

# Add project root to path
sys.path.append('/opt/ha-intent-predictor')

from src.storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader

class Sprint1FullRoomAnalyzer:
    """Sprint 1: Full room occupancy analysis with data cleaning and pattern discovery"""
    
    def __init__(self):
        """Initialize analyzer with database connection"""
        self.config = ConfigLoader()
        db_config = self.config.get("database.timescale")
        
        self.db_url = f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        self.db = TimescaleDBManager(self.db_url)
        
        # Full room sensors (core occupancy detection)
        self.full_room_sensors = {
            'bedroom': 'binary_sensor.bedroom_presence_sensor_full_bedroom',
            'living_room': 'binary_sensor.presence_livingroom_full',
            'kitchen': 'binary_sensor.kitchen_pressence_full_kitchen', 
            'office': 'binary_sensor.office_presence_full_office'
        }
        
        self.raw_data = {}
        self.cleaned_data = {}
        self.analysis_results = {}
        
        print(f"🚀 Sprint 1: Full Room Occupancy Analysis")
        print(f"📊 Analyzing {len(self.full_room_sensors)} full room sensors")
        print(f"🎯 Objective: Clean data and discover multi-room occupancy patterns")
        
    async def initialize(self):
        """Initialize database connection"""
        await self.db.initialize()
        print("✅ Database connection established")
        
    async def load_full_room_data(self, days_back: int = 7) -> Dict[str, pd.DataFrame]:
        """Load raw data for all full room sensors"""
        print(f"\n📥 Loading {days_back} days of full room sensor data...")
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days_back)
        
        for room_name, sensor_id in self.full_room_sensors.items():
            print(f"   Loading {room_name}: {sensor_id}")
            
            query = """
            SELECT timestamp, state, entity_id
            FROM sensor_events 
            WHERE entity_id = $1 
              AND timestamp >= $2 
              AND timestamp <= $3
              AND state IN ('on', 'off')
            ORDER BY timestamp ASC
            """
            
            async with self.db.engine.begin() as conn:
                from sqlalchemy import text
                result = await conn.execute(
                    text(query), 
                    [sensor_id, start_time, end_time]
                )
                rows = result.fetchall()
            
            if rows:
                df = pd.DataFrame(rows, columns=['timestamp', 'state', 'entity_id'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['state_binary'] = (df['state'] == 'on').astype(int)
                self.raw_data[room_name] = df
                print(f"     ✅ {len(df):,} events loaded for {room_name}")
            else:
                print(f"     ⚠️ No data found for {room_name}")
                
        print(f"✅ Raw data loaded for {len(self.raw_data)} rooms")
        return self.raw_data
        
    def clean_sensor_data(self, df: pd.DataFrame, min_duration_seconds: int = 2) -> pd.DataFrame:
        """
        Clean sensor data by removing consecutive duplicate states within min_duration_seconds
        
        Args:
            df: DataFrame with columns [timestamp, state, state_binary]
            min_duration_seconds: Minimum duration between state changes (default 2 seconds)
            
        Returns:
            Cleaned DataFrame with proper on→off→on sequences only
        """
        if len(df) == 0:
            return df
            
        cleaned_rows = []
        last_state = None
        last_timestamp = None
        
        for _, row in df.iterrows():
            current_state = row['state']
            current_timestamp = row['timestamp']
            
            # Always keep first event
            if last_state is None:
                cleaned_rows.append(row)
                last_state = current_state
                last_timestamp = current_timestamp
                continue
                
            # If state changed, check minimum duration
            if current_state != last_state:
                time_diff = (current_timestamp - last_timestamp).total_seconds()
                
                # Only keep if enough time has passed
                if time_diff >= min_duration_seconds:
                    cleaned_rows.append(row)
                    last_state = current_state
                    last_timestamp = current_timestamp
                # Otherwise skip this event (too quick transition)
            
            # Skip duplicate states (on→on or off→off)
            
        return pd.DataFrame(cleaned_rows) if cleaned_rows else pd.DataFrame(columns=df.columns)
        
    async def clean_all_sensor_data(self):
        """Apply data cleaning to all loaded sensor data"""
        print(f"\n🧹 Cleaning sensor data (removing transitions < 2 seconds)...")
        
        cleaning_stats = {}
        
        for room_name, raw_df in self.raw_data.items():
            print(f"   Cleaning {room_name}...")
            
            # Apply cleaning
            cleaned_df = self.clean_sensor_data(raw_df, min_duration_seconds=2)
            
            # Calculate statistics
            original_count = len(raw_df)
            cleaned_count = len(cleaned_df)
            removed_count = original_count - cleaned_count
            removal_percentage = (removed_count / original_count * 100) if original_count > 0 else 0
            
            cleaning_stats[room_name] = {
                'original_events': original_count,
                'cleaned_events': cleaned_count,
                'removed_events': removed_count,
                'removal_percentage': removal_percentage
            }
            
            self.cleaned_data[room_name] = cleaned_df
            
            print(f"     📊 {original_count:,} → {cleaned_count:,} events ({removed_count:,} removed, {removal_percentage:.1f}%)")
            
        print(f"✅ Data cleaning completed for {len(self.cleaned_data)} rooms")
        return cleaning_stats
        
    def analyze_room_activity(self) -> Dict:
        """Analyze activity patterns for each room"""
        print(f"\n📈 Analyzing room activity patterns...")
        
        room_analysis = {}
        
        for room_name, df in self.cleaned_data.items():
            if len(df) == 0:
                continue
                
            print(f"   Analyzing {room_name}...")
            
            # Calculate transitions
            transitions = []
            occupancy_durations = []
            vacancy_durations = []
            
            current_occupancy_start = None
            current_vacancy_start = None
            
            for i in range(len(df)):
                current_state = df.iloc[i]['state']
                current_time = df.iloc[i]['timestamp']
                
                if current_state == 'on':  # Occupancy starts
                    transitions.append('vacant→occupied')
                    if current_vacancy_start is not None:
                        vacancy_duration = (current_time - current_vacancy_start).total_seconds() / 60  # minutes
                        vacancy_durations.append(vacancy_duration)
                    current_occupancy_start = current_time
                    current_vacancy_start = None
                    
                elif current_state == 'off':  # Occupancy ends
                    transitions.append('occupied→vacant')
                    if current_occupancy_start is not None:
                        occupancy_duration = (current_time - current_occupancy_start).total_seconds() / 60  # minutes
                        occupancy_durations.append(occupancy_duration)
                    current_vacancy_start = current_time
                    current_occupancy_start = None
            
            # Calculate daily patterns
            df_copy = df.copy()
            df_copy['hour'] = df_copy['timestamp'].dt.hour
            df_copy['day_of_week'] = df_copy['timestamp'].dt.day_name()
            
            # Hourly activity (count of events)
            hourly_activity = df_copy.groupby('hour').size()
            
            # Daily activity
            daily_activity = df_copy.groupby('day_of_week').size()
            
            room_analysis[room_name] = {
                'total_events': len(df),
                'total_transitions': len(transitions),
                'occupancy_events': len([t for t in transitions if 'occupied' in t]),
                'vacancy_events': len([t for t in transitions if 'vacant' in t]),
                'avg_occupancy_duration_min': np.mean(occupancy_durations) if occupancy_durations else 0,
                'avg_vacancy_duration_min': np.mean(vacancy_durations) if vacancy_durations else 0,
                'max_occupancy_duration_min': max(occupancy_durations) if occupancy_durations else 0,
                'max_vacancy_duration_min': max(vacancy_durations) if vacancy_durations else 0,
                'hourly_activity': hourly_activity.to_dict(),
                'daily_activity': daily_activity.to_dict(),
                'transitions': transitions,
                'occupancy_durations': occupancy_durations,
                'vacancy_durations': vacancy_durations
            }
            
            print(f"     📊 {len(transitions):,} transitions, avg occupancy: {np.mean(occupancy_durations):.1f}min" if occupancy_durations else "     📊 No occupancy data")
            
        print(f"✅ Activity analysis completed for {len(room_analysis)} rooms")
        self.analysis_results = room_analysis
        return room_analysis
        
    def generate_multi_room_timeline(self, hours_back: int = 24) -> None:
        """Generate timeline visualization showing multi-room occupancy overlaps"""
        print(f"\n📊 Generating multi-room occupancy timeline ({hours_back}h)...")
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours_back)
        
        fig, axes = plt.subplots(len(self.cleaned_data), 1, figsize=(15, 2*len(self.cleaned_data)), sharex=True)
        if len(self.cleaned_data) == 1:
            axes = [axes]
            
        room_colors = {
            'bedroom': '#FF6B6B',      # Red
            'living_room': '#4ECDC4',  # Teal  
            'kitchen': '#45B7D1',      # Blue
            'office': '#96CEB4'        # Green
        }
        
        overlaps = []  # Track simultaneous occupancy
        timeline_data = {}
        
        for idx, (room_name, df) in enumerate(self.cleaned_data.items()):
            if len(df) == 0:
                continue
                
            # Filter to time window
            mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
            df_window = df[mask].copy()
            
            if len(df_window) == 0:
                continue
                
            # Create occupancy timeline
            timeline = []
            current_state = 'off'  # Assume vacant at start
            
            for _, row in df_window.iterrows():
                timeline.append({
                    'timestamp': row['timestamp'],
                    'state': current_state,
                    'transition': f"{current_state}→{row['state']}"
                })
                current_state = row['state']
                
            # Add final state until end
            timeline.append({
                'timestamp': end_time,
                'state': current_state,
                'transition': f"{current_state}→end"
            })
            
            timeline_data[room_name] = timeline
            
            # Plot room timeline
            ax = axes[idx]
            
            for i in range(len(timeline) - 1):
                start_ts = timeline[i]['timestamp']
                end_ts = timeline[i+1]['timestamp']
                state = timeline[i]['state']
                
                if state == 'on':
                    ax.barh(0, (end_ts - start_ts).total_seconds() / 3600, 
                           left=(start_ts - start_time).total_seconds() / 3600,
                           height=0.6, color=room_colors.get(room_name, '#888888'), 
                           alpha=0.7, label='Occupied' if i == 0 else "")
                           
            ax.set_ylabel(room_name.replace('_', ' ').title(), fontsize=10)
            ax.set_ylim(-0.5, 0.5)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, hours_back)
            
        # Set up x-axis
        axes[-1].set_xlabel('Hours ago')
        axes[-1].set_xlim(0, hours_back)
        
        # Invert x-axis so recent time is on the right
        for ax in axes:
            ax.invert_xaxis()
            
        plt.suptitle(f'Multi-Room Occupancy Timeline - Last {hours_back} Hours', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = '/opt/ha-intent-predictor/sprints/temporal_occupancy/sprint_1_timeline.png'
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Timeline saved to: {plot_path}")
        
        # Analyze overlaps
        self.analyze_room_overlaps(timeline_data)
        
    def analyze_room_overlaps(self, timeline_data: Dict) -> None:
        """Analyze simultaneous occupancy patterns across rooms"""
        print(f"\n🔄 Analyzing multi-room occupancy overlaps...")
        
        # Find time periods with multiple room occupancy
        overlap_periods = []
        
        # Sample every minute to check occupancy state
        start_time = min([min([event['timestamp'] for event in timeline]) 
                         for timeline in timeline_data.values() if timeline])
        end_time = max([max([event['timestamp'] for event in timeline]) 
                       for timeline in timeline_data.values() if timeline])
        
        current_time = start_time
        overlap_count = 0
        total_minutes = 0
        
        while current_time <= end_time:
            occupied_rooms = []
            
            for room_name, timeline in timeline_data.items():
                # Find current state for this room at current_time
                current_state = 'off'  # default
                for i in range(len(timeline) - 1):
                    if timeline[i]['timestamp'] <= current_time < timeline[i+1]['timestamp']:
                        current_state = timeline[i]['state']
                        break
                        
                if current_state == 'on':
                    occupied_rooms.append(room_name)
                    
            if len(occupied_rooms) > 1:
                overlap_count += 1
                overlap_periods.append({
                    'timestamp': current_time,
                    'occupied_rooms': occupied_rooms,
                    'room_count': len(occupied_rooms)
                })
                
            total_minutes += 1
            current_time += timedelta(minutes=1)
            
        overlap_percentage = (overlap_count / total_minutes * 100) if total_minutes > 0 else 0
        
        print(f"   📊 Multi-room occupancy: {overlap_count:,} minutes ({overlap_percentage:.1f}% of time)")
        
        if overlap_periods:
            # Most common room combinations
            room_combinations = defaultdict(int)
            for period in overlap_periods:
                combo = tuple(sorted(period['occupied_rooms']))
                room_combinations[combo] += 1
                
            print(f"   🏆 Most common room combinations:")
            for combo, count in sorted(room_combinations.items(), key=lambda x: x[1], reverse=True)[:5]:
                combo_str = " + ".join(combo)
                percentage = count / overlap_count * 100
                print(f"     • {combo_str}: {count:,} minutes ({percentage:.1f}%)")
                
    def generate_data_quality_report(self, cleaning_stats: Dict) -> None:
        """Generate comprehensive data quality report"""
        print(f"\n📋 SPRINT 1 DATA QUALITY REPORT")
        print(f"=" * 50)
        
        print(f"\n🏠 FULL ROOM SENSORS ANALYZED:")
        for room_name, sensor_id in self.full_room_sensors.items():
            print(f"   • {room_name}: {sensor_id}")
            
        print(f"\n🧹 DATA CLEANING RESULTS:")
        total_original = sum(stats['original_events'] for stats in cleaning_stats.values())
        total_cleaned = sum(stats['cleaned_events'] for stats in cleaning_stats.values())
        total_removed = total_original - total_cleaned
        
        print(f"   📊 Overall: {total_original:,} → {total_cleaned:,} events")
        print(f"   🗑️ Removed: {total_removed:,} events ({total_removed/total_original*100:.1f}%)")
        
        for room_name, stats in cleaning_stats.items():
            print(f"   • {room_name}: {stats['original_events']:,} → {stats['cleaned_events']:,} " +
                  f"(-{stats['removed_events']:,}, {stats['removal_percentage']:.1f}%)")
                  
        print(f"\n📈 ROOM ACTIVITY ANALYSIS:")
        if self.analysis_results:
            for room_name, analysis in self.analysis_results.items():
                print(f"   • {room_name}:")
                print(f"     - Transitions: {analysis['total_transitions']:,}")
                print(f"     - Avg occupancy: {analysis['avg_occupancy_duration_min']:.1f} minutes")
                print(f"     - Max occupancy: {analysis['max_occupancy_duration_min']:.1f} minutes")
                print(f"     - Avg vacancy: {analysis['avg_vacancy_duration_min']:.1f} minutes")
                
        print(f"\n🎯 SPRINT 1 SUCCESS CRITERIA:")
        rooms_with_data = len([r for r in self.cleaned_data.values() if len(r) > 0])
        print(f"   ✅ Rooms with sufficient data: {rooms_with_data}/{len(self.full_room_sensors)}")
        print(f"   ✅ Data cleaning implemented: Remove transitions < 2 seconds")
        print(f"   ✅ Multi-room patterns analyzed: Timeline and overlap analysis complete")
        print(f"   ✅ Foundation for Sprint 2: Clean dataset ready for temporal features")
        
        # Recommendations for Sprint 2
        print(f"\n🚀 SPRINT 2 RECOMMENDATIONS:")
        
        # Identify most active rooms
        active_rooms = []
        if self.analysis_results:
            for room_name, analysis in self.analysis_results.items():
                if analysis['total_transitions'] > 10:  # Arbitrary threshold
                    active_rooms.append((room_name, analysis['total_transitions']))
                    
        active_rooms.sort(key=lambda x: x[1], reverse=True)
        
        if active_rooms:
            print(f"   🏆 Focus on most active rooms:")
            for room_name, transitions in active_rooms[:3]:
                print(f"     • {room_name}: {transitions:,} transitions")
        else:
            print(f"   ⚠️ Limited activity detected - extend analysis period")
            
        print(f"   🔮 Next features to implement:")
        print(f"     • Temporal patterns: hourly/daily occupancy cycles")
        print(f"     • Sequence patterns: room-to-room transition sequences")
        print(f"     • Duration modeling: occupancy/vacancy length prediction")
        print(f"     • Overlap features: multi-room occupancy correlation")
        
async def main():
    """Run Sprint 1 full room analysis"""
    analyzer = Sprint1FullRoomAnalyzer()
    
    try:
        # Initialize
        await analyzer.initialize()
        
        # Load data (7 days for comprehensive analysis)
        await analyzer.load_full_room_data(days_back=7)
        
        # Clean data
        cleaning_stats = await analyzer.clean_all_sensor_data()
        
        # Analyze patterns
        analyzer.analyze_room_activity()
        
        # Generate timeline
        analyzer.generate_multi_room_timeline(hours_back=24)
        
        # Generate report
        analyzer.generate_data_quality_report(cleaning_stats)
        
        print(f"\n🎉 SPRINT 1 COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved to: /opt/ha-intent-predictor/sprints/temporal_occupancy/")
        
    except Exception as e:
        print(f"❌ Sprint 1 failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await analyzer.db.close()

if __name__ == "__main__":
    asyncio.run(main())