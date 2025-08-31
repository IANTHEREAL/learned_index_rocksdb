#!/usr/bin/env python3
"""
Learned Index Performance Dashboard
Real-time monitoring and visualization of model performance metrics
"""

import json
import time
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from flask import Flask, render_template, jsonify, request
import plotly.graph_objs as go
import plotly.utils

app = Flask(__name__)

class MetricsDatabase:
    """SQLite database for storing performance metrics"""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Performance events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                key_value INTEGER NOT NULL,
                predicted_block INTEGER NOT NULL,
                actual_block INTEGER NOT NULL,
                confidence REAL NOT NULL,
                was_correct INTEGER NOT NULL,
                prediction_error_bytes REAL NOT NULL
            )
        ''')
        
        # Windowed metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS windowed_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                window_start_ms INTEGER NOT NULL,
                window_end_ms INTEGER NOT NULL,
                total_predictions INTEGER NOT NULL,
                correct_predictions INTEGER NOT NULL,
                accuracy_rate REAL NOT NULL,
                average_confidence REAL NOT NULL,
                average_error_bytes REAL NOT NULL,
                throughput_qps REAL NOT NULL
            )
        ''')
        
        # Model health table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_health (
                model_id TEXT PRIMARY KEY,
                last_training_timestamp_ms INTEGER NOT NULL,
                total_queries_served INTEGER NOT NULL,
                current_accuracy REAL NOT NULL,
                accuracy_trend_7d REAL NOT NULL,
                accuracy_trend_1h REAL NOT NULL,
                is_degrading INTEGER NOT NULL,
                needs_retraining INTEGER NOT NULL,
                last_retrain_timestamp_ms INTEGER NOT NULL,
                retrain_count INTEGER NOT NULL
            )
        ''')
        
        # Retraining events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS retraining_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                trigger_reason TEXT NOT NULL,
                success INTEGER NOT NULL,
                new_accuracy REAL,
                training_samples INTEGER,
                training_duration_ms INTEGER,
                error_message TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_prediction_event(self, model_id: str, timestamp_ms: int, key_value: int,
                              predicted_block: int, actual_block: int, confidence: float,
                              was_correct: bool, prediction_error_bytes: float):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO prediction_events 
            (model_id, timestamp_ms, key_value, predicted_block, actual_block, 
             confidence, was_correct, prediction_error_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_id, timestamp_ms, key_value, predicted_block, actual_block,
              confidence, int(was_correct), prediction_error_bytes))
        conn.commit()
        conn.close()
    
    def insert_windowed_metrics(self, model_id: str, window_start_ms: int, window_end_ms: int,
                              total_predictions: int, correct_predictions: int, accuracy_rate: float,
                              average_confidence: float, average_error_bytes: float, throughput_qps: float):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO windowed_metrics 
            (model_id, window_start_ms, window_end_ms, total_predictions, correct_predictions,
             accuracy_rate, average_confidence, average_error_bytes, throughput_qps)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_id, window_start_ms, window_end_ms, total_predictions, correct_predictions,
              accuracy_rate, average_confidence, average_error_bytes, throughput_qps))
        conn.commit()
        conn.close()
    
    def update_model_health(self, model_id: str, last_training_timestamp_ms: int,
                          total_queries_served: int, current_accuracy: float,
                          accuracy_trend_7d: float, accuracy_trend_1h: float,
                          is_degrading: bool, needs_retraining: bool,
                          last_retrain_timestamp_ms: int, retrain_count: int):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO model_health 
            (model_id, last_training_timestamp_ms, total_queries_served, current_accuracy,
             accuracy_trend_7d, accuracy_trend_1h, is_degrading, needs_retraining,
             last_retrain_timestamp_ms, retrain_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_id, last_training_timestamp_ms, total_queries_served, current_accuracy,
              accuracy_trend_7d, accuracy_trend_1h, int(is_degrading), int(needs_retraining),
              last_retrain_timestamp_ms, retrain_count))
        conn.commit()
        conn.close()
    
    def get_recent_metrics(self, model_id: str, hours: int = 24) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = int(time.time() * 1000) - (hours * 3600 * 1000)
        
        cursor.execute('''
            SELECT * FROM windowed_metrics 
            WHERE model_id = ? AND window_start_ms >= ?
            ORDER BY window_start_ms
        ''', (model_id, cutoff_time))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_all_models(self) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT model_id FROM model_health')
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_model_health(self, model_id: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM model_health WHERE model_id = ?', (model_id,))
        
        row = cursor.fetchone()
        if row:
            columns = [description[0] for description in cursor.description]
            result = dict(zip(columns, row))
        else:
            result = None
        
        conn.close()
        return result

class DashboardDataGenerator:
    """Generates synthetic data for demonstration purposes"""
    
    def __init__(self, db: MetricsDatabase):
        self.db = db
        self.running = False
        self.thread = None
    
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._generate_data)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _generate_data(self):
        import random
        import math
        
        models = ["sst_file_1.sst", "sst_file_2.sst", "sst_file_3.sst"]
        
        while self.running:
            timestamp_ms = int(time.time() * 1000)
            
            for model_id in models:
                # Generate synthetic prediction events
                for _ in range(random.randint(50, 200)):
                    key_value = random.randint(1000, 100000)
                    predicted_block = random.randint(0, 99)
                    actual_block = random.randint(0, 99)
                    
                    # Simulate accuracy degradation over time
                    base_accuracy = 0.95
                    time_factor = (timestamp_ms % (24 * 3600 * 1000)) / (24 * 3600 * 1000)
                    accuracy = base_accuracy - 0.1 * math.sin(time_factor * 2 * math.pi)
                    
                    was_correct = random.random() < accuracy
                    if was_correct:
                        actual_block = predicted_block
                    
                    confidence = random.uniform(0.7, 1.0) if was_correct else random.uniform(0.3, 0.8)
                    prediction_error = 0 if was_correct else random.uniform(100, 4000)
                    
                    self.db.insert_prediction_event(
                        model_id, timestamp_ms, key_value, predicted_block, actual_block,
                        confidence, was_correct, prediction_error
                    )
                
                # Generate windowed metrics
                window_start = timestamp_ms - 60000  # 1 minute window
                total_preds = random.randint(100, 500)
                correct_preds = int(total_preds * accuracy)
                
                self.db.insert_windowed_metrics(
                    model_id, window_start, timestamp_ms, total_preds, correct_preds,
                    accuracy, random.uniform(0.7, 0.9), random.uniform(50, 200),
                    total_preds / 60.0  # QPS
                )
                
                # Update model health
                trend_7d = random.uniform(-0.05, 0.05)
                trend_1h = random.uniform(-0.02, 0.02)
                is_degrading = accuracy < 0.9 or trend_1h < -0.01
                needs_retraining = accuracy < 0.85
                
                self.db.update_model_health(
                    model_id, timestamp_ms - 3600000, total_preds * 100, accuracy,
                    trend_7d, trend_1h, is_degrading, needs_retraining,
                    timestamp_ms - 1800000, random.randint(0, 5)
                )
            
            time.sleep(30)  # Update every 30 seconds

# Global instances
db = MetricsDatabase()
data_generator = DashboardDataGenerator(db)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/models')
def get_models():
    models = db.get_all_models()
    return jsonify(models)

@app.route('/api/metrics/<model_id>')
def get_model_metrics(model_id):
    hours = int(request.args.get('hours', 24))
    metrics = db.get_recent_metrics(model_id, hours)
    return jsonify(metrics)

@app.route('/api/health/<model_id>')
def get_model_health(model_id):
    health = db.get_model_health(model_id)
    return jsonify(health)

@app.route('/api/dashboard_data')
def get_dashboard_data():
    models = db.get_all_models()
    dashboard_data = {}
    
    for model_id in models:
        health = db.get_model_health(model_id)
        metrics = db.get_recent_metrics(model_id, 1)  # Last hour
        
        if health and metrics:
            dashboard_data[model_id] = {
                'health': health,
                'recent_metrics': metrics[-10:] if len(metrics) > 10 else metrics  # Last 10 points
            }
    
    return jsonify(dashboard_data)

@app.route('/api/accuracy_chart/<model_id>')
def get_accuracy_chart(model_id):
    hours = int(request.args.get('hours', 24))
    metrics = db.get_recent_metrics(model_id, hours)
    
    if not metrics:
        return jsonify({'error': 'No data found'})
    
    timestamps = [datetime.fromtimestamp(m['window_start_ms'] / 1000) for m in metrics]
    accuracy_values = [m['accuracy_rate'] * 100 for m in metrics]
    
    trace = go.Scatter(
        x=timestamps,
        y=accuracy_values,
        mode='lines+markers',
        name='Accuracy %',
        line=dict(color='#2E8B57', width=2)
    )
    
    layout = go.Layout(
        title=f'Model Accuracy Over Time - {model_id}',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Accuracy %', range=[0, 100]),
        hovermode='closest'
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/api/throughput_chart/<model_id>')
def get_throughput_chart(model_id):
    hours = int(request.args.get('hours', 24))
    metrics = db.get_recent_metrics(model_id, hours)
    
    if not metrics:
        return jsonify({'error': 'No data found'})
    
    timestamps = [datetime.fromtimestamp(m['window_start_ms'] / 1000) for m in metrics]
    throughput_values = [m['throughput_qps'] for m in metrics]
    
    trace = go.Scatter(
        x=timestamps,
        y=throughput_values,
        mode='lines+markers',
        name='Throughput (QPS)',
        line=dict(color='#4169E1', width=2)
    )
    
    layout = go.Layout(
        title=f'Throughput Over Time - {model_id}',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Queries Per Second'),
        hovermode='closest'
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/start_demo')
def start_demo():
    data_generator.start()
    return jsonify({'status': 'Demo data generation started'})

@app.route('/stop_demo')
def stop_demo():
    data_generator.stop()
    return jsonify({'status': 'Demo data generation stopped'})

if __name__ == '__main__':
    print("Starting Learned Index Performance Dashboard...")
    print("Visit http://localhost:5000 to view the dashboard")
    print("Use /start_demo endpoint to begin generating demo data")
    
    app.run(debug=True, host='0.0.0.0', port=5000)