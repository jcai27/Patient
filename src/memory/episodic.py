"""Episodic memory for per-user notes."""
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.config import DATABASE_URL


class EpisodicMemory:
    """Stores episodic notes per user."""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = db_path
        else:
            # Extract path from SQLAlchemy URL
            if DATABASE_URL.startswith("sqlite:///"):
                self.db_path = DATABASE_URL.replace("sqlite:///", "")
            else:
                self.db_path = "persona_memory.db"
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodic_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                bullet TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                rolling_summary TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                conversation_turns INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT,
                turn_index INTEGER NOT NULL,
                user_message TEXT,
                assistant_response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, turn_index)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_turns_session
            ON conversation_turns (session_id, turn_index)
        """)
        
        conn.commit()
        conn.close()
    
    def add_note(self, user_id: str, bullet: str, metadata: Optional[Dict[str, Any]] = None):
        """Add an episodic note for a user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_str = None
        if metadata:
            import json
            metadata_str = json.dumps(metadata)
        
        cursor.execute("""
            INSERT INTO episodic_notes (user_id, bullet, metadata)
            VALUES (?, ?, ?)
        """, (user_id, bullet, metadata_str))
        
        conn.commit()
        conn.close()
    
    def get_user_notes(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent episodic notes for a user."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, bullet, created_at, metadata
            FROM episodic_notes
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        notes = []
        for row in rows:
            metadata = None
            if row["metadata"]:
                import json
                metadata = json.loads(row["metadata"])
            
            notes.append({
                "id": row["id"],
                "bullet": row["bullet"],
                "created_at": row["created_at"],
                "metadata": metadata,
            })
        
        return notes
    
    def update_summary(self, session_id: str, user_id: str, summary: str, turns: int):
        """Update conversation summary for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO conversation_summaries 
            (session_id, user_id, rolling_summary, conversation_turns, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (session_id, user_id, summary, turns))
        
        conn.commit()
        conn.close()
    
    def get_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation summary for a session."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, user_id, rolling_summary, updated_at, conversation_turns
            FROM conversation_summaries
            WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "session_id": row["session_id"],
                "user_id": row["user_id"],
                "rolling_summary": row["rolling_summary"],
                "updated_at": row["updated_at"],
                "conversation_turns": row["conversation_turns"],
            }
        
        return None

    def append_turn(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_response: str,
    ):
        """Persist a user/assistant exchange for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COALESCE(MAX(turn_index), -1)
            FROM conversation_turns
            WHERE session_id = ?
            """,
            (session_id,),
        )
        last_index = cursor.fetchone()[0]
        next_index = last_index + 1

        cursor.execute(
            """
            INSERT INTO conversation_turns (
                session_id,
                user_id,
                turn_index,
                user_message,
                assistant_response
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, user_id, next_index, user_message, assistant_response),
        )

        conn.commit()
        conn.close()

    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = 20,
    ) -> List[Dict[str, Optional[str]]]:
        """
        Retrieve recent conversation history for a session.

        Args:
            session_id: Conversation identifier
            limit: Maximum number of turns to return (oldest-first).
                   Pass None to retrieve the full history.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        order_clause = "ORDER BY turn_index ASC"
        params: List[Any] = [session_id]

        if limit is not None:
            order_clause = "ORDER BY turn_index DESC"
            query = f"""
                SELECT turn_index, user_message, assistant_response
                FROM conversation_turns
                WHERE session_id = ?
                {order_clause}
                LIMIT ?
            """
            params.append(limit)
        else:
            query = f"""
                SELECT turn_index, user_message, assistant_response
                FROM conversation_turns
                WHERE session_id = ?
                {order_clause}
            """

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.close()

        if limit is not None:
            rows = list(reversed(rows))

        history: List[Dict[str, Optional[str]]] = []
        for row in rows:
            history.append({
                "user": row["user_message"],
                "assistant": row["assistant_response"],
            })
        return history

