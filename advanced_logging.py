"""
Advanced logging system for the LangChain agentic dashboard.
Provides structured logging, user activity tracking, and system monitoring.
"""

import logging
import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid

@dataclass
class LogEntry:
    """Structured log entry for database storage"""
    id: str
    timestamp: str
    level: str
    component: str
    user_id: Optional[str]
    session_id: Optional[str]
    action: str
    message: str
    metadata: Dict[str, Any]
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

class AdvancedLogger:
    """
    Advanced logging system with database storage, user tracking, and analytics.
    """
    
    def __init__(self, db_path: str = "logs.db", log_file: str = "app.log"):
        self.db_path = db_path
        self.log_file = log_file
        self._init_database()
        self._setup_file_logging()
        
        # Component-specific loggers
        self.ingestion_logger = logging.getLogger("ingestion")
        self.embedding_logger = logging.getLogger("embedding")
        self.query_logger = logging.getLogger("query")
        self.ui_logger = logging.getLogger("ui")
        self.system_logger = logging.getLogger("system")
        
    def _init_database(self):
        """Initialize SQLite database for structured logging"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                component TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                action TEXT NOT NULL,
                message TEXT NOT NULL,
                metadata TEXT,
                duration_ms REAL,
                success BOOLEAN DEFAULT TRUE,
                error_message TEXT
            )
        """)
        
        # Create user sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_queries INTEGER DEFAULT 0,
                total_files_uploaded INTEGER DEFAULT 0,
                total_embeddings INTEGER DEFAULT 0,
                session_metadata TEXT
            )
        """)
        
        # Create user profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                email TEXT,
                preferences TEXT,
                created_at TEXT NOT NULL,
                last_active TEXT,
                total_sessions INTEGER DEFAULT 0,
                total_queries INTEGER DEFAULT 0,
                favorite_queries TEXT,
                custom_filters TEXT
            )
        """)
        
        # Create performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                component TEXT NOT NULL,
                operation TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                success BOOLEAN NOT NULL,
                resource_usage TEXT,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _setup_file_logging(self):
        """Setup file-based logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
    
    def log_activity(self, 
                    level: str,
                    component: str,
                    action: str,
                    message: str,
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    duration_ms: Optional[float] = None,
                    success: bool = True,
                    error_message: Optional[str] = None):
        """Log structured activity to database and file"""
        
        log_entry = LogEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            level=level,
            component=component,
            user_id=user_id,
            session_id=session_id,
            action=action,
            message=message,
            metadata=metadata or {},
            duration_ms=duration_ms,
            success=success,
            error_message=error_message
        )
        
        # Store in database
        self._store_log_entry(log_entry)
        
        # Log to file
        logger = getattr(self, f"{component}_logger", self.system_logger)
        log_message = f"[{action}] {message}"
        if metadata:
            log_message += f" | Metadata: {json.dumps(metadata)}"
        if duration_ms:
            log_message += f" | Duration: {duration_ms:.2f}ms"
        if not success and error_message:
            log_message += f" | Error: {error_message}"
            
        if level == "ERROR":
            logger.error(log_message)
        elif level == "WARNING":
            logger.warning(log_message)
        elif level == "INFO":
            logger.info(log_message)
        else:
            logger.debug(log_message)
    
    def _store_log_entry(self, log_entry: LogEntry):
        """Store log entry in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO logs (id, timestamp, level, component, user_id, session_id, 
                            action, message, metadata, duration_ms, success, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log_entry.id,
            log_entry.timestamp,
            log_entry.level,
            log_entry.component,
            log_entry.user_id,
            log_entry.session_id,
            log_entry.action,
            log_entry.message,
            json.dumps(log_entry.metadata),
            log_entry.duration_ms,
            log_entry.success,
            log_entry.error_message
        ))
        
        conn.commit()
        conn.close()
    
    def start_session(self, user_id: str, session_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new user session"""
        session_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO user_sessions (session_id, user_id, start_time, session_metadata)
            VALUES (?, ?, ?, ?)
        """, (session_id, user_id, datetime.now().isoformat(), json.dumps(session_metadata or {})))
        
        conn.commit()
        conn.close()
        
        self.log_activity("INFO", "ui", "session_start", f"User {user_id} started session", 
                         user_id=user_id, session_id=session_id, metadata=session_metadata)
        
        return session_id
    
    def end_session(self, session_id: str):
        """End a user session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE user_sessions 
            SET end_time = ?
            WHERE session_id = ?
        """, (datetime.now().isoformat(), session_id))
        
        conn.commit()
        conn.close()
        
        self.log_activity("INFO", "ui", "session_end", f"Session {session_id} ended", 
                         session_id=session_id)
    
    def log_file_upload(self, user_id: str, session_id: str, file_path: str, 
                       file_size: int, processing_time: float, success: bool, 
                       error_message: Optional[str] = None):
        """Log file upload activity"""
        metadata = {
            "file_path": file_path,
            "file_size": file_size,
            "file_extension": Path(file_path).suffix
        }
        
        self.log_activity(
            "INFO" if success else "ERROR",
            "ingestion",
            "file_upload",
            f"File upload: {Path(file_path).name}",
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            duration_ms=processing_time * 1000,
            success=success,
            error_message=error_message
        )
        
        if success:
            # Update session stats
            self._update_session_stats(session_id, "total_files_uploaded", 1)
    
    def log_embedding_generation(self, user_id: str, session_id: str, 
                               document_count: int, processing_time: float, 
                               success: bool, error_message: Optional[str] = None):
        """Log embedding generation activity"""
        metadata = {
            "document_count": document_count,
            "documents_per_second": document_count / processing_time if processing_time > 0 else 0
        }
        
        self.log_activity(
            "INFO" if success else "ERROR",
            "embedding",
            "embedding_generation",
            f"Generated embeddings for {document_count} documents",
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            duration_ms=processing_time * 1000,
            success=success,
            error_message=error_message
        )
        
        if success:
            # Update session stats
            self._update_session_stats(session_id, "total_embeddings", document_count)
    
    def log_query_execution(self, user_id: str, session_id: str, query: str,
                          route: str, processing_time: float, success: bool,
                          result_count: int = 0, error_message: Optional[str] = None):
        """Log query execution activity"""
        metadata = {
            "query": query,
            "route": route,
            "result_count": result_count,
            "query_length": len(query)
        }
        
        self.log_activity(
            "INFO" if success else "ERROR",
            "query",
            "query_execution",
            f"Query executed via {route}: {query[:50]}...",
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            duration_ms=processing_time * 1000,
            success=success,
            error_message=error_message
        )
        
        if success:
            # Update session stats
            self._update_session_stats(session_id, "total_queries", 1)
    
    def log_slm_call(self, user_id: str, session_id: str, model: str, 
                    input_length: int, output_length: int, processing_time: float,
                    success: bool, error_message: Optional[str] = None):
        """Log SLM/LLM API calls"""
        metadata = {
            "model": model,
            "input_length": input_length,
            "output_length": output_length,
            "tokens_per_second": (input_length + output_length) / processing_time if processing_time > 0 else 0
        }
        
        self.log_activity(
            "INFO" if success else "ERROR",
            "system",
            "llm_call",
            f"SLM/LLM call to {model}",
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            duration_ms=processing_time * 1000,
            success=success,
            error_message=error_message
        )
    
    def _update_session_stats(self, session_id: str, field: str, increment: int):
        """Update session statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            UPDATE user_sessions 
            SET {field} = {field} + ?
            WHERE session_id = ?
        """, (increment, session_id))
        
        conn.commit()
        conn.close()
    
    def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get analytics for a specific user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get user activity summary
        cursor.execute("""
            SELECT 
                COUNT(*) as total_queries,
                COUNT(CASE WHEN success = 1 THEN 1 END) as successful_queries,
                AVG(duration_ms) as avg_query_time,
                COUNT(DISTINCT DATE(timestamp)) as active_days
            FROM logs 
            WHERE user_id = ? AND timestamp >= datetime('now', '-{} days')
        """.format(days), (user_id,))
        
        activity_stats = cursor.fetchone()
        
        # Get most common queries
        cursor.execute("""
            SELECT metadata, COUNT(*) as frequency
            FROM logs 
            WHERE user_id = ? AND action = 'query_execution' AND timestamp >= datetime('now', '-{} days')
            GROUP BY metadata
            ORDER BY frequency DESC
            LIMIT 10
        """.format(days), (user_id,))
        
        common_queries = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_queries": activity_stats[0] or 0,
            "successful_queries": activity_stats[1] or 0,
            "avg_query_time_ms": activity_stats[2] or 0,
            "active_days": activity_stats[3] or 0,
            "common_queries": [{"query": json.loads(q[0]).get("query", ""), "frequency": q[1]} for q in common_queries]
        }
    
    def get_system_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get system performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get performance metrics
        cursor.execute("""
            SELECT 
                component,
                AVG(duration_ms) as avg_duration,
                COUNT(*) as total_operations,
                COUNT(CASE WHEN success = 1 THEN 1 END) as successful_operations
            FROM logs 
            WHERE timestamp >= datetime('now', '-{} hours')
            GROUP BY component
        """.format(hours))
        
        component_stats = cursor.fetchall()
        
        # Get error rates
        cursor.execute("""
            SELECT 
                component,
                COUNT(*) as total_errors
            FROM logs 
            WHERE success = 0 AND timestamp >= datetime('now', '-{} hours')
            GROUP BY component
        """.format(hours))
        
        error_stats = cursor.fetchall()
        
        conn.close()
        
        return {
            "component_stats": [
                {
                    "component": stat[0],
                    "avg_duration_ms": stat[1] or 0,
                    "total_operations": stat[2] or 0,
                    "success_rate": (stat[3] or 0) / (stat[2] or 1) * 100
                }
                for stat in component_stats
            ],
            "error_stats": [
                {
                    "component": stat[0],
                    "total_errors": stat[1] or 0
                }
                for stat in error_stats
            ]
        }

# Global logger instance
advanced_logger = AdvancedLogger()
