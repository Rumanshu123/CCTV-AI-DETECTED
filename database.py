import sqlite3
import uuid
import numpy as np
from datetime import datetime

class FaceDatabase:
    def __init__(self, db_path='faces.db'):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        
    def _create_tables(self):
        cursor = self.conn.cursor()
        
        # Drop existing tables if they exist (for clean start)
        cursor.execute("DROP TABLE IF EXISTS faces")
        cursor.execute("DROP TABLE IF EXISTS detections")
        
        # Create new tables with correct schema
        cursor.execute('''
            CREATE TABLE faces (
                face_id TEXT PRIMARY KEY,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                best_image BLOB,
                best_confidence REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT,
                timestamp TIMESTAMP,
                image BLOB,
                confidence REAL,
                x INTEGER,
                y INTEGER,
                width INTEGER,
                height INTEGER,
                FOREIGN KEY (face_id) REFERENCES faces (face_id)
            )
        ''')
        self.conn.commit()
        
    def _ensure_contiguous(self, image):
        """Ensure the image is C-contiguous"""
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        return image
        
    def add_face(self, image, confidence, x, y, width, height, timestamp):
        face_id = str(uuid.uuid4())
        cursor = self.conn.cursor()
        
        # Ensure image is contiguous
        image = self._ensure_contiguous(image)
        
        # Insert into faces table
        cursor.execute('''
            INSERT INTO faces (face_id, first_seen, last_seen, best_image, best_confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (face_id, timestamp, timestamp, sqlite3.Binary(image.tobytes()), confidence))
        
        # Insert into detections table
        cursor.execute('''
            INSERT INTO detections 
            (face_id, timestamp, image, confidence, x, y, width, height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (face_id, timestamp, sqlite3.Binary(image.tobytes()), confidence, x, y, width, height))
        
        self.conn.commit()
        return face_id
    
    def update_face(self, face_id, image, confidence, x, y, width, height, timestamp):
        cursor = self.conn.cursor()
        
        # Ensure image is contiguous
        image = self._ensure_contiguous(image)
        
        # Update faces table if this is a better image
        cursor.execute('''
            UPDATE faces 
            SET last_seen = ?,
                best_image = CASE WHEN ? > best_confidence THEN ? ELSE best_image END,
                best_confidence = CASE WHEN ? > best_confidence THEN ? ELSE best_confidence END
            WHERE face_id = ?
        ''', (timestamp, confidence, sqlite3.Binary(image.tobytes()), confidence, confidence, face_id))
        
        # Insert into detections table
        cursor.execute('''
            INSERT INTO detections 
            (face_id, timestamp, image, confidence, x, y, width, height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (face_id, timestamp, sqlite3.Binary(image.tobytes()), confidence, x, y, width, height))
        
        self.conn.commit()
    
    def close(self):
        self.conn.close()