#!/usr/bin/env python3
"""
Time Series Query System
System for fetching daily, weekly, and monthly time series data
"""

import sys
import os
import pandas as pd
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TimeSeriesQuerySystem:
    def __init__(self):
        self.db_path = 'data_prototype/metadata.db'
        self.excel_file = 'uploads/Energy Consumption Daily Report MHS Ele - Copy.xlsx'
        
    def detect_time_series_sheets(self) -> List[Dict[str, Any]]:
        """Detect time series sheets in the Excel file"""
        try:
            excel_data = pd.read_excel(self.excel_file, sheet_name=None)
            time_series_sheets = []
            
            for sheet_name, df in excel_data.items():
                if self._is_time_series_sheet(df):
                    time_series_sheets.append({
                        'sheet_name': sheet_name,
                        'date_range': self._extract_date_range(df),
                        'equipment_count': len(df),
                        'columns': list(df.columns)
                    })
            
            return time_series_sheets
            
        except Exception as e:
            print(f"Error detecting time series sheets: {e}")
            return []
    
    def _is_time_series_sheet(self, df: pd.DataFrame) -> bool:
        """Check if a sheet contains time series data"""
        # Check for date columns
        date_columns = []
        for col in df.columns:
            if isinstance(col, datetime) or 'date' in str(col).lower():
                date_columns.append(col)
        
        # Check for difference columns
        diff_columns = [col for col in df.columns if 'difference' in str(col).lower()]
        
        # Must have date columns and difference columns
        return len(date_columns) > 0 and len(diff_columns) > 0
    
    def _extract_date_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract date range from time series sheet"""
        date_columns = []
        for col in df.columns:
            if isinstance(col, datetime):
                date_columns.append(col)
        
        if date_columns:
            return {
                'start_date': min(date_columns),
                'end_date': max(date_columns),
                'total_days': len(date_columns)
            }
        return {}
    
    def fetch_daily_data(self, equipment: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Fetch daily time series data for specific equipment"""
        try:
            # Load time series sheets
            time_series_sheets = self.detect_time_series_sheets()
            
            if not time_series_sheets:
                return {"error": "No time series sheets found"}
            
            # Find relevant sheets based on date range
            relevant_sheets = self._find_relevant_sheets(time_series_sheets, start_date, end_date)
            
            if not relevant_sheets:
                return {"error": "No relevant time series sheets found for the date range"}
            
            # Extract data from relevant sheets
            daily_data = []
            for sheet_info in relevant_sheets:
                sheet_data = self._extract_equipment_data(sheet_info['sheet_name'], equipment)
                if sheet_data:
                    daily_data.extend(sheet_data)
            
            return {
                "equipment": equipment,
                "data_points": len(daily_data),
                "date_range": {
                    "start": min([d['date'] for d in daily_data]) if daily_data else None,
                    "end": max([d['date'] for d in daily_data]) if daily_data else None
                },
                "daily_readings": daily_data
            }
            
        except Exception as e:
            return {"error": f"Error fetching daily data: {e}"}
    
    def _find_relevant_sheets(self, time_series_sheets: List[Dict], start_date: str, end_date: str) -> List[Dict]:
        """Find sheets relevant to the date range"""
        relevant_sheets = []
        
        for sheet in time_series_sheets:
            if 'date_range' in sheet and sheet['date_range']:
                sheet_start = sheet['date_range']['start_date']
                sheet_end = sheet['date_range']['end_date']
                
                # Check if sheet overlaps with requested date range
                if self._date_ranges_overlap(sheet_start, sheet_end, start_date, end_date):
                    relevant_sheets.append(sheet)
        
        return relevant_sheets
    
    def _date_ranges_overlap(self, sheet_start, sheet_end, req_start, req_end) -> bool:
        """Check if date ranges overlap"""
        if not req_start or not req_end:
            return True  # If no date range specified, include all sheets
        
        try:
            req_start_dt = pd.to_datetime(req_start)
            req_end_dt = pd.to_datetime(req_end)
            
            return (sheet_start <= req_end_dt) and (sheet_end >= req_start_dt)
        except:
            return True
    
    def _extract_equipment_data(self, sheet_name: str, equipment: str) -> List[Dict]:
        """Extract equipment data from a specific sheet"""
        try:
            df = pd.read_excel(self.excel_file, sheet_name=sheet_name)
            
            # Find equipment row
            equipment_row = None
            for idx, row in df.iterrows():
                if equipment.lower() in str(row.iloc[0]).lower():
                    equipment_row = row
                    break
            
            if equipment_row is None:
                return []
            
            # Extract date columns and values
            daily_data = []
            for col in df.columns:
                if isinstance(col, datetime):
                    value = equipment_row[col]
                    if pd.notna(value):
                        daily_data.append({
                            'date': col.strftime('%Y-%m-%d'),
                            'reading': float(value),
                            'sheet': sheet_name
                        })
            
            return daily_data
            
        except Exception as e:
            print(f"Error extracting equipment data: {e}")
            return []
    
    def fetch_weekly_data(self, equipment: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Fetch weekly aggregated data"""
        daily_data = self.fetch_daily_data(equipment, start_date, end_date)
        
        if 'error' in daily_data:
            return daily_data
        
        # Aggregate daily data into weekly
        weekly_data = self._aggregate_to_weekly(daily_data['daily_readings'])
        
        return {
            "equipment": equipment,
            "period": "weekly",
            "data_points": len(weekly_data),
            "weekly_readings": weekly_data
        }
    
    def _aggregate_to_weekly(self, daily_readings: List[Dict]) -> List[Dict]:
        """Aggregate daily readings to weekly"""
        weekly_data = []
        
        # Group by week
        current_week = None
        week_readings = []
        
        for reading in daily_readings:
            date = pd.to_datetime(reading['date'])
            week_start = date - timedelta(days=date.weekday())
            
            if current_week is None or week_start != current_week:
                if current_week is not None and week_readings:
                    weekly_data.append(self._calculate_weekly_summary(week_readings, current_week))
                
                current_week = week_start
                week_readings = [reading]
            else:
                week_readings.append(reading)
        
        # Add last week
        if week_readings:
            weekly_data.append(self._calculate_weekly_summary(week_readings, current_week))
        
        return weekly_data
    
    def _calculate_weekly_summary(self, week_readings: List[Dict], week_start: datetime) -> Dict:
        """Calculate weekly summary from daily readings"""
        readings = [r['reading'] for r in week_readings]
        
        return {
            'week_start': week_start.strftime('%Y-%m-%d'),
            'week_end': (week_start + timedelta(days=6)).strftime('%Y-%m-%d'),
            'total_consumption': sum(readings),
            'average_daily': sum(readings) / len(readings),
            'min_reading': min(readings),
            'max_reading': max(readings),
            'daily_count': len(readings)
        }
    
    def fetch_monthly_data(self, equipment: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Fetch monthly aggregated data"""
        daily_data = self.fetch_daily_data(equipment, start_date, end_date)
        
        if 'error' in daily_data:
            return daily_data
        
        # Aggregate daily data into monthly
        monthly_data = self._aggregate_to_monthly(daily_data['daily_readings'])
        
        return {
            "equipment": equipment,
            "period": "monthly",
            "data_points": len(monthly_data),
            "monthly_readings": monthly_data
        }
    
    def _aggregate_to_monthly(self, daily_readings: List[Dict]) -> List[Dict]:
        """Aggregate daily readings to monthly"""
        monthly_data = []
        
        # Group by month
        current_month = None
        month_readings = []
        
        for reading in daily_readings:
            date = pd.to_datetime(reading['date'])
            month_start = date.replace(day=1)
            
            if current_month is None or month_start != current_month:
                if current_month is not None and month_readings:
                    monthly_data.append(self._calculate_monthly_summary(month_readings, current_month))
                
                current_month = month_start
                month_readings = [reading]
            else:
                month_readings.append(reading)
        
        # Add last month
        if month_readings:
            monthly_data.append(self._calculate_monthly_summary(month_readings, current_month))
        
        return monthly_data
    
    def _calculate_monthly_summary(self, month_readings: List[Dict], month_start: datetime) -> Dict:
        """Calculate monthly summary from daily readings"""
        readings = [r['reading'] for r in month_readings]
        
        return {
            'month': month_start.strftime('%Y-%m'),
            'total_consumption': sum(readings),
            'average_daily': sum(readings) / len(readings),
            'min_reading': min(readings),
            'max_reading': max(readings),
            'daily_count': len(readings)
        }

def test_time_series_system():
    """Test the time series query system"""
    print("TESTING TIME SERIES QUERY SYSTEM")
    print("=" * 50)
    
    ts_system = TimeSeriesQuerySystem()
    
    # Test 1: Detect time series sheets
    print("1. Detecting time series sheets...")
    sheets = ts_system.detect_time_series_sheets()
    print(f"Found {len(sheets)} time series sheets:")
    for sheet in sheets[:3]:  # Show first 3
        print(f"  - {sheet['sheet_name']}: {sheet['date_range']}")
    
    # Test 2: Fetch daily data
    print("\n2. Testing daily data fetch...")
    daily_data = ts_system.fetch_daily_data("I/C Panel Numerical Relay")
    if 'error' not in daily_data:
        print(f"Daily data for I/C Panel Numerical Relay:")
        print(f"  Data points: {daily_data['data_points']}")
        print(f"  Date range: {daily_data['date_range']}")
        if daily_data['daily_readings']:
            print(f"  Sample reading: {daily_data['daily_readings'][0]}")
    else:
        print(f"Error: {daily_data['error']}")
    
    # Test 3: Fetch weekly data
    print("\n3. Testing weekly data fetch...")
    weekly_data = ts_system.fetch_weekly_data("I/C Panel Numerical Relay")
    if 'error' not in weekly_data:
        print(f"Weekly data for I/C Panel Numerical Relay:")
        print(f"  Data points: {weekly_data['data_points']}")
        if weekly_data['weekly_readings']:
            print(f"  Sample week: {weekly_data['weekly_readings'][0]}")
    else:
        print(f"Error: {weekly_data['error']}")
    
    return True

if __name__ == "__main__":
    test_time_series_system()
