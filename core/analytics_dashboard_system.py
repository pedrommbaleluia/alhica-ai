#!/usr/bin/env python3
"""
Alhica AI - Sistema Completo de Analytics e Dashboard
Sistema avançado para análise de dados, métricas e visualizações em tempo real

Copyright (c) 2024 Alhica AI Team
"""

import os
import json
import logging
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from flask import Flask, render_template_string, jsonify, request
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import psutil
import redis

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/alhica-ai/analytics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MetricData:
    """Dados de uma métrica específica"""
    name: str
    value: Union[int, float, str]
    timestamp: datetime
    category: str
    metadata: Dict[str, Any]
    unit: str = ""
    trend: Optional[str] = None  # up, down, stable

@dataclass
class AnalyticsReport:
    """Relatório de analytics"""
    report_id: str
    title: str
    description: str
    generated_at: datetime
    data: Dict[str, Any]
    charts: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]

class AnalyticsDashboardSystem:
    """Sistema completo de analytics e dashboard"""
    
    def __init__(self, db_path: str = "/opt/alhica-ai/data/analytics.db"):
        self.db_path = db_path
        self.redis_client = None
        self.metrics_cache = {}
        self.real_time_data = defaultdict(list)
        self.dashboard_app = None
        
        # Configurar Redis se disponível
        self._setup_redis()
        
        # Configurar base de dados
        self._setup_database()
        
        # Inicializar coleta de métricas
        self.metrics_collector = MetricsCollector(self)
        
        # Configurar dashboard web
        self._setup_dashboard()
        
        logger.info("📊 Sistema de analytics inicializado")
    
    def _setup_redis(self):
        """Configurar Redis para cache de métricas"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis conectado para cache de métricas")
        except Exception as e:
            logger.warning(f"Redis não disponível: {e}")
            self.redis_client = None
    
    def _setup_database(self):
        """Configurar base de dados para analytics"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                # Tabela de métricas
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        value TEXT NOT NULL,
                        category TEXT NOT NULL,
                        unit TEXT DEFAULT '',
                        metadata TEXT DEFAULT '{}',
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabela de eventos do sistema
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS system_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        event_data TEXT NOT NULL,
                        severity TEXT DEFAULT 'info',
                        source TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabela de performance de modelos
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        execution_time REAL,
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabela de uso de recursos
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS resource_usage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        resource_type TEXT NOT NULL,
                        usage_value REAL NOT NULL,
                        max_value REAL,
                        unit TEXT DEFAULT '',
                        hostname TEXT DEFAULT 'localhost',
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabela de sessões de utilizador
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        session_duration INTEGER,
                        commands_executed INTEGER DEFAULT 0,
                        success_rate REAL DEFAULT 0.0,
                        start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                        end_time DATETIME
                    )
                ''')
                
                # Tabela de relatórios
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS analytics_reports (
                        report_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        description TEXT,
                        report_data TEXT NOT NULL,
                        generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        report_type TEXT DEFAULT 'general'
                    )
                ''')
                
                # Índices para performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_category ON metrics(category)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp ON model_performance(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_resource_usage_timestamp ON resource_usage(timestamp)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Erro ao configurar base de dados: {e}")
    
    def _setup_dashboard(self):
        """Configurar dashboard web"""
        self.dashboard_app = Flask(__name__)
        
        @self.dashboard_app.route('/')
        def dashboard_home():
            return render_template_string(self._get_dashboard_template())
        
        @self.dashboard_app.route('/api/metrics')
        def api_metrics():
            metrics = self.get_current_metrics()
            return jsonify(metrics)
        
        @self.dashboard_app.route('/api/charts/<chart_type>')
        def api_charts(chart_type):
            chart_data = self.generate_chart(chart_type)
            return jsonify(chart_data)
        
        @self.dashboard_app.route('/api/reports')
        def api_reports():
            reports = self.get_recent_reports()
            return jsonify(reports)
        
        @self.dashboard_app.route('/api/realtime')
        def api_realtime():
            data = self.get_realtime_data()
            return jsonify(data)
    
    def record_metric(self, name: str, value: Union[int, float, str], 
                     category: str, unit: str = "", metadata: Dict[str, Any] = None):
        """Registar métrica"""
        metric = MetricData(
            name=name,
            value=value,
            timestamp=datetime.now(),
            category=category,
            metadata=metadata or {},
            unit=unit
        )
        
        # Salvar na base de dados
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO metrics (name, value, category, unit, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    metric.name,
                    json.dumps(metric.value),
                    metric.category,
                    metric.unit,
                    json.dumps(metric.metadata)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao salvar métrica: {e}")
        
        # Cache em Redis se disponível
        if self.redis_client:
            try:
                cache_key = f"metric:{category}:{name}"
                self.redis_client.setex(cache_key, 3600, json.dumps(asdict(metric), default=str))
            except Exception as e:
                logger.warning(f"Erro ao cachear métrica: {e}")
        
        # Adicionar aos dados em tempo real
        self.real_time_data[category].append(metric)
        
        # Manter apenas últimos 100 pontos por categoria
        if len(self.real_time_data[category]) > 100:
            self.real_time_data[category] = self.real_time_data[category][-100:]
    
    def record_system_event(self, event_type: str, event_data: Dict[str, Any], 
                           severity: str = "info", source: str = "system"):
        """Registar evento do sistema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO system_events (event_type, event_data, severity, source)
                    VALUES (?, ?, ?, ?)
                ''', (event_type, json.dumps(event_data), severity, source))
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao registar evento: {e}")
    
    def record_model_performance(self, model_name: str, metric_name: str, 
                                metric_value: float, execution_time: float = None,
                                input_tokens: int = None, output_tokens: int = None):
        """Registar performance de modelo"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO model_performance 
                    (model_name, metric_name, metric_value, execution_time, input_tokens, output_tokens)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (model_name, metric_name, metric_value, execution_time, input_tokens, output_tokens))
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao registar performance: {e}")
    
    def record_resource_usage(self, resource_type: str, usage_value: float, 
                             max_value: float = None, unit: str = "", hostname: str = "localhost"):
        """Registar uso de recursos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO resource_usage (resource_type, usage_value, max_value, unit, hostname)
                    VALUES (?, ?, ?, ?, ?)
                ''', (resource_type, usage_value, max_value, unit, hostname))
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao registar uso de recursos: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Obter métricas atuais"""
        metrics = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Métricas por categoria
                cursor.execute('''
                    SELECT category, COUNT(*) as count, 
                           AVG(CASE WHEN value REGEXP '^[0-9]+\.?[0-9]*$' THEN CAST(value as REAL) END) as avg_value
                    FROM metrics 
                    WHERE timestamp > datetime('now', '-1 hour')
                    GROUP BY category
                ''')
                
                categories = {}
                for row in cursor.fetchall():
                    categories[row[0]] = {
                        'count': row[1],
                        'average': round(row[2], 2) if row[2] else 0
                    }
                
                metrics['categories'] = categories
                
                # Eventos recentes
                cursor.execute('''
                    SELECT event_type, COUNT(*) as count
                    FROM system_events 
                    WHERE timestamp > datetime('now', '-1 hour')
                    GROUP BY event_type
                    ORDER BY count DESC
                    LIMIT 10
                ''')
                
                events = dict(cursor.fetchall())
                metrics['recent_events'] = events
                
                # Performance dos modelos
                cursor.execute('''
                    SELECT model_name, AVG(metric_value) as avg_performance, AVG(execution_time) as avg_time
                    FROM model_performance 
                    WHERE timestamp > datetime('now', '-1 hour')
                    GROUP BY model_name
                ''')
                
                models = {}
                for row in cursor.fetchall():
                    models[row[0]] = {
                        'performance': round(row[1], 3) if row[1] else 0,
                        'avg_time': round(row[2], 3) if row[2] else 0
                    }
                
                metrics['model_performance'] = models
                
        except Exception as e:
            logger.error(f"Erro ao obter métricas: {e}")
        
        return metrics
    
    def generate_chart(self, chart_type: str) -> Dict[str, Any]:
        """Gerar dados para gráfico"""
        try:
            if chart_type == "resource_usage":
                return self._generate_resource_usage_chart()
            elif chart_type == "model_performance":
                return self._generate_model_performance_chart()
            elif chart_type == "user_activity":
                return self._generate_user_activity_chart()
            elif chart_type == "system_events":
                return self._generate_system_events_chart()
            elif chart_type == "realtime_metrics":
                return self._generate_realtime_metrics_chart()
            else:
                return {"error": f"Tipo de gráfico não suportado: {chart_type}"}
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico {chart_type}: {e}")
            return {"error": str(e)}
    
    def _generate_resource_usage_chart(self) -> Dict[str, Any]:
        """Gerar gráfico de uso de recursos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query('''
                    SELECT resource_type, usage_value, timestamp
                    FROM resource_usage 
                    WHERE timestamp > datetime('now', '-24 hours')
                    ORDER BY timestamp
                ''', conn)
            
            if df.empty:
                return {"data": [], "layout": {}, "message": "Sem dados de recursos"}
            
            fig = px.line(df, x='timestamp', y='usage_value', color='resource_type',
                         title='Uso de Recursos (24h)')
            
            return {
                "data": fig.data,
                "layout": fig.layout,
                "type": "line"
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico de recursos: {e}")
            return {"error": str(e)}
    
    def _generate_model_performance_chart(self) -> Dict[str, Any]:
        """Gerar gráfico de performance dos modelos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query('''
                    SELECT model_name, AVG(metric_value) as avg_performance, 
                           AVG(execution_time) as avg_time, COUNT(*) as requests
                    FROM model_performance 
                    WHERE timestamp > datetime('now', '-24 hours')
                    GROUP BY model_name
                ''', conn)
            
            if df.empty:
                return {"data": [], "layout": {}, "message": "Sem dados de performance"}
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Performance Média', 'Tempo de Execução'),
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Bar(x=df['model_name'], y=df['avg_performance'], name='Performance'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=df['model_name'], y=df['avg_time'], name='Tempo (s)'),
                row=2, col=1
            )
            
            fig.update_layout(title_text="Performance dos Modelos IA")
            
            return {
                "data": fig.data,
                "layout": fig.layout,
                "type": "subplot"
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico de performance: {e}")
            return {"error": str(e)}
    
    def _generate_user_activity_chart(self) -> Dict[str, Any]:
        """Gerar gráfico de atividade de utilizadores"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query('''
                    SELECT DATE(start_time) as date, COUNT(*) as sessions,
                           AVG(commands_executed) as avg_commands,
                           AVG(success_rate) as avg_success_rate
                    FROM user_sessions 
                    WHERE start_time > datetime('now', '-30 days')
                    GROUP BY DATE(start_time)
                    ORDER BY date
                ''', conn)
            
            if df.empty:
                return {"data": [], "layout": {}, "message": "Sem dados de utilizadores"}
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Sessões por Dia', 'Taxa de Sucesso'),
                specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['sessions'], name='Sessões', mode='lines+markers'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['avg_commands'], name='Comandos Médios', mode='lines'),
                row=1, col=1, secondary_y=True
            )
            
            fig.add_trace(
                go.Bar(x=df['date'], y=df['avg_success_rate'], name='Taxa de Sucesso'),
                row=2, col=1
            )
            
            fig.update_layout(title_text="Atividade de Utilizadores (30 dias)")
            
            return {
                "data": fig.data,
                "layout": fig.layout,
                "type": "subplot"
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico de utilizadores: {e}")
            return {"error": str(e)}
    
    def _generate_system_events_chart(self) -> Dict[str, Any]:
        """Gerar gráfico de eventos do sistema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query('''
                    SELECT event_type, severity, COUNT(*) as count
                    FROM system_events 
                    WHERE timestamp > datetime('now', '-24 hours')
                    GROUP BY event_type, severity
                    ORDER BY count DESC
                ''', conn)
            
            if df.empty:
                return {"data": [], "layout": {}, "message": "Sem eventos do sistema"}
            
            fig = px.sunburst(df, path=['severity', 'event_type'], values='count',
                             title='Eventos do Sistema por Severidade (24h)')
            
            return {
                "data": fig.data,
                "layout": fig.layout,
                "type": "sunburst"
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico de eventos: {e}")
            return {"error": str(e)}
    
    def _generate_realtime_metrics_chart(self) -> Dict[str, Any]:
        """Gerar gráfico de métricas em tempo real"""
        try:
            data = []
            
            for category, metrics in self.real_time_data.items():
                if not metrics:
                    continue
                
                timestamps = [m.timestamp for m in metrics[-20:]]  # Últimos 20 pontos
                values = []
                
                for m in metrics[-20:]:
                    try:
                        # Tentar converter para número
                        if isinstance(m.value, (int, float)):
                            values.append(m.value)
                        else:
                            values.append(float(m.value))
                    except (ValueError, TypeError):
                        values.append(0)
                
                if values:
                    data.append({
                        'x': timestamps,
                        'y': values,
                        'name': category,
                        'type': 'scatter',
                        'mode': 'lines+markers'
                    })
            
            layout = {
                'title': 'Métricas em Tempo Real',
                'xaxis': {'title': 'Tempo'},
                'yaxis': {'title': 'Valor'},
                'showlegend': True
            }
            
            return {
                "data": data,
                "layout": layout,
                "type": "realtime"
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico em tempo real: {e}")
            return {"error": str(e)}
    
    def generate_report(self, report_type: str = "general", 
                       start_date: datetime = None, end_date: datetime = None) -> AnalyticsReport:
        """Gerar relatório de analytics"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
        
        report_id = f"report_{int(datetime.now().timestamp())}"
        
        try:
            # Coletar dados para o relatório
            report_data = self._collect_report_data(start_date, end_date)
            
            # Gerar insights
            insights = self._generate_insights(report_data)
            
            # Gerar recomendações
            recommendations = self._generate_recommendations(report_data)
            
            # Gerar gráficos
            charts = [
                self.generate_chart("resource_usage"),
                self.generate_chart("model_performance"),
                self.generate_chart("user_activity"),
                self.generate_chart("system_events")
            ]
            
            report = AnalyticsReport(
                report_id=report_id,
                title=f"Relatório de Analytics - {report_type.title()}",
                description=f"Relatório gerado para o período de {start_date.date()} a {end_date.date()}",
                generated_at=datetime.now(),
                data=report_data,
                charts=charts,
                insights=insights,
                recommendations=recommendations
            )
            
            # Salvar relatório
            self._save_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            raise
    
    def _collect_report_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Coletar dados para relatório"""
        data = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Estatísticas gerais
                cursor.execute('''
                    SELECT COUNT(*) FROM metrics 
                    WHERE timestamp BETWEEN ? AND ?
                ''', (start_date.isoformat(), end_date.isoformat()))
                data['total_metrics'] = cursor.fetchone()[0]
                
                cursor.execute('''
                    SELECT COUNT(*) FROM system_events 
                    WHERE timestamp BETWEEN ? AND ?
                ''', (start_date.isoformat(), end_date.isoformat()))
                data['total_events'] = cursor.fetchone()[0]
                
                cursor.execute('''
                    SELECT COUNT(*) FROM user_sessions 
                    WHERE start_time BETWEEN ? AND ?
                ''', (start_date.isoformat(), end_date.isoformat()))
                data['total_sessions'] = cursor.fetchone()[0]
                
                # Performance dos modelos
                cursor.execute('''
                    SELECT model_name, AVG(metric_value), AVG(execution_time), COUNT(*)
                    FROM model_performance 
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY model_name
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                model_stats = {}
                for row in cursor.fetchall():
                    model_stats[row[0]] = {
                        'avg_performance': round(row[1], 3) if row[1] else 0,
                        'avg_execution_time': round(row[2], 3) if row[2] else 0,
                        'total_requests': row[3]
                    }
                data['model_statistics'] = model_stats
                
                # Uso de recursos
                cursor.execute('''
                    SELECT resource_type, AVG(usage_value), MAX(usage_value), COUNT(*)
                    FROM resource_usage 
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY resource_type
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                resource_stats = {}
                for row in cursor.fetchall():
                    resource_stats[row[0]] = {
                        'avg_usage': round(row[1], 2) if row[1] else 0,
                        'max_usage': round(row[2], 2) if row[2] else 0,
                        'measurements': row[3]
                    }
                data['resource_statistics'] = resource_stats
                
        except Exception as e:
            logger.error(f"Erro ao coletar dados do relatório: {e}")
        
        return data
    
    def _generate_insights(self, report_data: Dict[str, Any]) -> List[str]:
        """Gerar insights baseados nos dados"""
        insights = []
        
        try:
            # Insights sobre modelos
            if 'model_statistics' in report_data:
                model_stats = report_data['model_statistics']
                
                if model_stats:
                    # Modelo mais usado
                    most_used = max(model_stats.items(), key=lambda x: x[1]['total_requests'])
                    insights.append(f"O modelo mais utilizado foi {most_used[0]} com {most_used[1]['total_requests']} requisições")
                    
                    # Modelo mais rápido
                    fastest = min(model_stats.items(), key=lambda x: x[1]['avg_execution_time'])
                    insights.append(f"O modelo mais rápido foi {fastest[0]} com tempo médio de {fastest[1]['avg_execution_time']}s")
            
            # Insights sobre recursos
            if 'resource_statistics' in report_data:
                resource_stats = report_data['resource_statistics']
                
                for resource, stats in resource_stats.items():
                    if stats['max_usage'] > stats['avg_usage'] * 1.5:
                        insights.append(f"Picos de uso detectados em {resource}: máximo {stats['max_usage']}% vs média {stats['avg_usage']}%")
            
            # Insights sobre atividade
            if report_data.get('total_sessions', 0) > 0:
                insights.append(f"Total de {report_data['total_sessions']} sessões de utilizador registadas")
            
            if report_data.get('total_events', 0) > 0:
                insights.append(f"Sistema gerou {report_data['total_events']} eventos durante o período")
            
        except Exception as e:
            logger.error(f"Erro ao gerar insights: {e}")
        
        return insights
    
    def _generate_recommendations(self, report_data: Dict[str, Any]) -> List[str]:
        """Gerar recomendações baseadas nos dados"""
        recommendations = []
        
        try:
            # Recomendações sobre recursos
            if 'resource_statistics' in report_data:
                resource_stats = report_data['resource_statistics']
                
                for resource, stats in resource_stats.items():
                    if stats['avg_usage'] > 80:
                        recommendations.append(f"Considere aumentar recursos de {resource} - uso médio de {stats['avg_usage']}%")
                    elif stats['avg_usage'] < 20:
                        recommendations.append(f"Recursos de {resource} subutilizados - considere otimização")
            
            # Recomendações sobre modelos
            if 'model_statistics' in report_data:
                model_stats = report_data['model_statistics']
                
                slow_models = [name for name, stats in model_stats.items() if stats['avg_execution_time'] > 5.0]
                if slow_models:
                    recommendations.append(f"Modelos com performance lenta detectados: {', '.join(slow_models)} - considere otimização")
            
            # Recomendações gerais
            if report_data.get('total_events', 0) > 1000:
                recommendations.append("Alto volume de eventos detectado - considere implementar filtros ou agregação")
            
            if not report_data.get('total_sessions', 0):
                recommendations.append("Nenhuma sessão de utilizador detectada - verificar conectividade e logs")
            
        except Exception as e:
            logger.error(f"Erro ao gerar recomendações: {e}")
        
        return recommendations
    
    def _save_report(self, report: AnalyticsReport):
        """Salvar relatório na base de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO analytics_reports (report_id, title, description, report_data)
                    VALUES (?, ?, ?, ?)
                ''', (
                    report.report_id,
                    report.title,
                    report.description,
                    json.dumps(asdict(report), default=str)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao salvar relatório: {e}")
    
    def get_recent_reports(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obter relatórios recentes"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT report_id, title, description, generated_at
                    FROM analytics_reports 
                    ORDER BY generated_at DESC 
                    LIMIT ?
                ''', (limit,))
                
                reports = []
                for row in cursor.fetchall():
                    reports.append({
                        'report_id': row[0],
                        'title': row[1],
                        'description': row[2],
                        'generated_at': row[3]
                    })
                
                return reports
                
        except Exception as e:
            logger.error(f"Erro ao obter relatórios: {e}")
            return []
    
    def get_realtime_data(self) -> Dict[str, Any]:
        """Obter dados em tempo real"""
        data = {}
        
        # Métricas em tempo real
        for category, metrics in self.real_time_data.items():
            if metrics:
                latest = metrics[-1]
                data[category] = {
                    'value': latest.value,
                    'timestamp': latest.timestamp.isoformat(),
                    'unit': latest.unit
                }
        
        # Estatísticas do sistema
        data['system'] = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'active_sessions': len(self.real_time_data),
            'timestamp': datetime.now().isoformat()
        }
        
        return data
    
    def _get_dashboard_template(self) -> str:
        """Template HTML para dashboard"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Alhica AI - Analytics Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; margin: -20px -20px 20px -20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
        .metric-label { color: #7f8c8d; margin-top: 5px; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .realtime-indicator { display: inline-block; width: 10px; height: 10px; background: #2ecc71; border-radius: 50%; margin-right: 5px; animation: blink 2s infinite; }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0.3; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Alhica AI - Analytics Dashboard</h1>
        <p><span class="realtime-indicator"></span>Monitorização em Tempo Real</p>
    </div>
    
    <div class="metrics-grid" id="metricsGrid">
        <!-- Métricas serão carregadas aqui -->
    </div>
    
    <div class="chart-container">
        <h3>📊 Uso de Recursos</h3>
        <div id="resourceChart"></div>
    </div>
    
    <div class="chart-container">
        <h3>🤖 Performance dos Modelos IA</h3>
        <div id="modelChart"></div>
    </div>
    
    <div class="chart-container">
        <h3>👥 Atividade de Utilizadores</h3>
        <div id="userChart"></div>
    </div>
    
    <div class="chart-container">
        <h3>⚡ Métricas em Tempo Real</h3>
        <div id="realtimeChart"></div>
    </div>

    <script>
        function loadMetrics() {
            $.get('/api/metrics', function(data) {
                updateMetricsGrid(data);
            });
        }
        
        function updateMetricsGrid(data) {
            let html = '';
            
            // Métricas de categorias
            if (data.categories) {
                for (let category in data.categories) {
                    html += `
                        <div class="metric-card">
                            <div class="metric-value">${data.categories[category].count}</div>
                            <div class="metric-label">${category} (última hora)</div>
                        </div>
                    `;
                }
            }
            
            // Performance dos modelos
            if (data.model_performance) {
                for (let model in data.model_performance) {
                    html += `
                        <div class="metric-card">
                            <div class="metric-value">${data.model_performance[model].performance}</div>
                            <div class="metric-label">${model} - Performance</div>
                        </div>
                    `;
                }
            }
            
            $('#metricsGrid').html(html);
        }
        
        function loadChart(chartType, containerId) {
            $.get(`/api/charts/${chartType}`, function(data) {
                if (data.error) {
                    $(`#${containerId}`).html(`<p>Erro: ${data.error}</p>`);
                } else if (data.message) {
                    $(`#${containerId}`).html(`<p>${data.message}</p>`);
                } else {
                    Plotly.newPlot(containerId, data.data, data.layout);
                }
            });
        }
        
        function loadRealtimeData() {
            $.get('/api/realtime', function(data) {
                // Atualizar métricas do sistema
                if (data.system) {
                    updateSystemMetrics(data.system);
                }
            });
        }
        
        function updateSystemMetrics(system) {
            let html = `
                <div class="metric-card">
                    <div class="metric-value">${system.cpu_percent}%</div>
                    <div class="metric-label">CPU</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${system.memory_percent}%</div>
                    <div class="metric-label">Memória</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${system.disk_percent}%</div>
                    <div class="metric-label">Disco</div>
                </div>
            `;
            $('#metricsGrid').prepend(html);
        }
        
        // Carregar dados iniciais
        $(document).ready(function() {
            loadMetrics();
            loadChart('resource_usage', 'resourceChart');
            loadChart('model_performance', 'modelChart');
            loadChart('user_activity', 'userChart');
            loadChart('realtime_metrics', 'realtimeChart');
            
            // Atualizar dados em tempo real
            setInterval(function() {
                loadMetrics();
                loadRealtimeData();
                loadChart('realtime_metrics', 'realtimeChart');
            }, 30000); // A cada 30 segundos
        });
    </script>
</body>
</html>
        '''
    
    def start_dashboard(self, host: str = "0.0.0.0", port: int = 8080):
        """Iniciar dashboard web"""
        def run_dashboard():
            self.dashboard_app.run(host=host, port=port, debug=False)
        
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        
        logger.info(f"Dashboard iniciado em http://{host}:{port}")

class MetricsCollector:
    """Coletor automático de métricas do sistema"""
    
    def __init__(self, analytics_system: AnalyticsDashboardSystem):
        self.analytics = analytics_system
        self.running = False
        self.collection_thread = None
        self.collection_interval = 60  # segundos
    
    def start_collection(self):
        """Iniciar coleta automática de métricas"""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info("Coleta automática de métricas iniciada")
    
    def stop_collection(self):
        """Parar coleta de métricas"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
        
        logger.info("Coleta de métricas parada")
    
    def _collection_loop(self):
        """Loop principal de coleta"""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Erro na coleta de métricas: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Coletar métricas do sistema"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.analytics.record_resource_usage("cpu", cpu_percent, 100, "%")
            
            # Memória
            memory = psutil.virtual_memory()
            self.analytics.record_resource_usage("memory", memory.percent, 100, "%")
            
            # Disco
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.analytics.record_resource_usage("disk", disk_percent, 100, "%")
            
            # Rede
            network = psutil.net_io_counters()
            self.analytics.record_metric("network_bytes_sent", network.bytes_sent, "network", "bytes")
            self.analytics.record_metric("network_bytes_recv", network.bytes_recv, "network", "bytes")
            
            # Processos
            process_count = len(psutil.pids())
            self.analytics.record_metric("process_count", process_count, "system", "count")
            
        except Exception as e:
            logger.error(f"Erro ao coletar métricas do sistema: {e}")

def main():
    """Função principal para teste"""
    analytics = AnalyticsDashboardSystem()
    
    # Iniciar coleta de métricas
    analytics.metrics_collector.start_collection()
    
    # Registar algumas métricas de teste
    analytics.record_metric("test_metric", 42, "test", "units")
    analytics.record_model_performance("qwen", "accuracy", 0.95, 2.3, 100, 50)
    analytics.record_system_event("test_event", {"message": "Sistema iniciado"}, "info", "test")
    
    # Gerar relatório
    report = analytics.generate_report("test")
    print(f"Relatório gerado: {report.report_id}")
    print(f"Insights: {len(report.insights)}")
    print(f"Recomendações: {len(report.recommendations)}")
    
    # Iniciar dashboard
    analytics.start_dashboard(port=8080)
    
    print("Analytics dashboard iniciado em http://localhost:8080")
    print("Pressione Ctrl+C para parar...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        analytics.metrics_collector.stop_collection()
        print("Sistema parado")

if __name__ == "__main__":
    main()

