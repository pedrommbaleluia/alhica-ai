#!/usr/bin/env python3
"""
Alhica AI - Integração dos Modelos IA Híbridos
A Primeira Plataforma do Mundo com IA Conversacional + SSH Automático

Integração dos modelos:
- Qwen 3 25B: Compreensão de linguagem natural
- DeepSeek-Coder: Geração de código e scripts
- WizardCoder: Automação e workflows complexos

Copyright (c) 2024 Alhica AI Team
"""

import os
import json
import asyncio
import aiohttp
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import logging
import sqlite3
from pathlib import Path
import subprocess
import psutil
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle

# Dependências para processamento de linguagem
import re
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('alhica_ai_models')

@dataclass
class ModelConfig:
    """Configuração de modelo IA"""
    name: str
    endpoint: str
    model_id: str
    speciality: str
    max_tokens: int
    temperature: float
    timeout: int
    health_check_url: str
    description: str
    capabilities: List[str]
    priority: int  # 1 = highest priority

@dataclass
class ModelResponse:
    """Resposta de modelo IA"""
    model_id: str
    response: str
    confidence: float
    processing_time: float
    tokens_used: int
    cost_estimate: float
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class TaskAnalysis:
    """Análise de tarefa para seleção de modelo"""
    task_type: str  # coding, automation, general, security, analysis
    complexity: str  # low, medium, high, expert
    risk_level: str  # low, medium, high, critical
    estimated_tokens: int
    preferred_models: List[str]
    fallback_models: List[str]
    requires_context: bool
    execution_context: Dict[str, Any]

class ModelHealthMonitor:
    """Monitor de saúde dos modelos"""
    
    def __init__(self, models_config: Dict[str, ModelConfig]):
        self.models_config = models_config
        self.health_status = {}
        self.performance_metrics = {}
        self.last_check = {}
        self.check_interval = 30  # segundos
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Iniciar monitorização contínua"""
        def monitor_worker():
            while True:
                try:
                    self._check_all_models()
                    time.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"Erro no monitor de saúde: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        monitor_thread.start()
        logger.info("Monitor de saúde dos modelos iniciado")
    
    def _check_all_models(self):
        """Verificar saúde de todos os modelos"""
        for model_id, config in self.models_config.items():
            try:
                start_time = time.time()
                
                # Verificar endpoint de saúde
                response = requests.get(
                    config.health_check_url,
                    timeout=5
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    self.health_status[model_id] = {
                        'status': 'healthy',
                        'response_time': response_time,
                        'last_check': datetime.now(),
                        'error': None
                    }
                else:
                    self.health_status[model_id] = {
                        'status': 'unhealthy',
                        'response_time': response_time,
                        'last_check': datetime.now(),
                        'error': f"HTTP {response.status_code}"
                    }
                
                # Atualizar métricas de performance
                self._update_performance_metrics(model_id, response_time, True)
                
            except Exception as e:
                self.health_status[model_id] = {
                    'status': 'error',
                    'response_time': None,
                    'last_check': datetime.now(),
                    'error': str(e)
                }
                
                self._update_performance_metrics(model_id, None, False)
    
    def _update_performance_metrics(self, model_id: str, response_time: Optional[float], success: bool):
        """Atualizar métricas de performance"""
        if model_id not in self.performance_metrics:
            self.performance_metrics[model_id] = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'avg_response_time': 0.0,
                'response_times': [],
                'uptime_percentage': 100.0,
                'last_24h_requests': []
            }
        
        metrics = self.performance_metrics[model_id]
        metrics['total_requests'] += 1
        
        if success:
            metrics['successful_requests'] += 1
            if response_time:
                metrics['response_times'].append(response_time)
                # Manter apenas últimas 100 medições
                if len(metrics['response_times']) > 100:
                    metrics['response_times'] = metrics['response_times'][-100:]
                
                metrics['avg_response_time'] = sum(metrics['response_times']) / len(metrics['response_times'])
        else:
            metrics['failed_requests'] += 1
        
        # Calcular uptime
        if metrics['total_requests'] > 0:
            metrics['uptime_percentage'] = (metrics['successful_requests'] / metrics['total_requests']) * 100
        
        # Registar pedido nas últimas 24h
        metrics['last_24h_requests'].append({
            'timestamp': datetime.now(),
            'success': success,
            'response_time': response_time
        })
        
        # Limpar pedidos antigos (>24h)
        cutoff = datetime.now() - timedelta(hours=24)
        metrics['last_24h_requests'] = [
            req for req in metrics['last_24h_requests'] 
            if req['timestamp'] > cutoff
        ]
    
    def get_healthy_models(self) -> List[str]:
        """Obter lista de modelos saudáveis"""
        healthy = []
        for model_id, status in self.health_status.items():
            if status.get('status') == 'healthy':
                healthy.append(model_id)
        return healthy
    
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Obter métricas de performance de um modelo"""
        return self.performance_metrics.get(model_id, {})
    
    def get_best_model_for_task(self, task_type: str, exclude_models: List[str] = None) -> Optional[str]:
        """Obter melhor modelo para uma tarefa"""
        exclude_models = exclude_models or []
        healthy_models = self.get_healthy_models()
        
        # Filtrar modelos excluídos
        available_models = [m for m in healthy_models if m not in exclude_models]
        
        if not available_models:
            return None
        
        # Mapear tipos de tarefa para modelos preferenciais
        task_preferences = {
            'coding': ['deepseek', 'wizardcoder', 'qwen'],
            'automation': ['wizardcoder', 'deepseek', 'qwen'],
            'general': ['qwen', 'deepseek', 'wizardcoder'],
            'security': ['qwen', 'deepseek', 'wizardcoder'],
            'analysis': ['qwen', 'wizardcoder', 'deepseek']
        }
        
        preferred_order = task_preferences.get(task_type, ['qwen', 'deepseek', 'wizardcoder'])
        
        # Selecionar primeiro modelo disponível na ordem de preferência
        for model_id in preferred_order:
            if model_id in available_models:
                return model_id
        
        # Fallback: primeiro modelo disponível
        return available_models[0] if available_models else None

class TaskAnalyzer:
    """Analisador de tarefas para seleção inteligente de modelos"""
    
    def __init__(self):
        self.task_patterns = self._load_task_patterns()
        self.complexity_indicators = self._load_complexity_indicators()
        self.risk_indicators = self._load_risk_indicators()
    
    def _load_task_patterns(self) -> Dict[str, List[str]]:
        """Carregar padrões de tarefas"""
        return {
            'coding': [
                'escrever código', 'criar script', 'programar', 'desenvolver',
                'implementar', 'codificar', 'função', 'classe', 'algoritmo',
                'debug', 'corrigir bug', 'otimizar código', 'refatorar'
            ],
            'automation': [
                'automatizar', 'workflow', 'pipeline', 'orquestrar',
                'agendar', 'batch', 'cron', 'deploy', 'ci/cd',
                'provisionar', 'configurar automaticamente'
            ],
            'security': [
                'segurança', 'vulnerabilidade', 'auditoria', 'firewall',
                'encriptar', 'autenticação', 'autorização', 'ssl', 'tls',
                'certificado', 'chave', 'password', 'hash'
            ],
            'analysis': [
                'analisar', 'relatório', 'estatística', 'métrica',
                'dashboard', 'monitorizar', 'log', 'performance',
                'diagnóstico', 'troubleshoot'
            ],
            'general': [
                'explicar', 'ajudar', 'como', 'o que é', 'tutorial',
                'guia', 'documentação', 'exemplo', 'demonstrar'
            ]
        }
    
    def _load_complexity_indicators(self) -> Dict[str, List[str]]:
        """Carregar indicadores de complexidade"""
        return {
            'low': [
                'simples', 'básico', 'rápido', 'pequeno', 'fácil',
                'listar', 'mostrar', 'verificar status'
            ],
            'medium': [
                'configurar', 'instalar', 'atualizar', 'modificar',
                'personalizar', 'integrar', 'conectar'
            ],
            'high': [
                'complexo', 'avançado', 'múltiplos', 'enterprise',
                'produção', 'crítico', 'migrar', 'transformar'
            ],
            'expert': [
                'arquitetura', 'otimização avançada', 'machine learning',
                'big data', 'distribuído', 'escalável', 'alta disponibilidade'
            ]
        }
    
    def _load_risk_indicators(self) -> Dict[str, List[str]]:
        """Carregar indicadores de risco"""
        return {
            'low': [
                'ler', 'listar', 'mostrar', 'verificar', 'status',
                'informação', 'consultar', 'visualizar'
            ],
            'medium': [
                'criar', 'adicionar', 'modificar', 'atualizar',
                'configurar', 'instalar', 'reiniciar serviço'
            ],
            'high': [
                'deletar', 'remover', 'parar', 'desativar',
                'alterar configuração crítica', 'sudo', 'root'
            ],
            'critical': [
                'formatar', 'rm -rf', 'dd', 'fdisk', 'mkfs',
                'shutdown', 'reboot', 'destruir', 'apagar sistema'
            ]
        }
    
    def analyze_task(self, user_input: str, context: Dict[str, Any] = None) -> TaskAnalysis:
        """Analisar tarefa do utilizador"""
        user_input_lower = user_input.lower()
        
        # Determinar tipo de tarefa
        task_type = self._determine_task_type(user_input_lower)
        
        # Determinar complexidade
        complexity = self._determine_complexity(user_input_lower, context)
        
        # Determinar nível de risco
        risk_level = self._determine_risk_level(user_input_lower)
        
        # Estimar tokens necessários
        estimated_tokens = self._estimate_tokens(user_input, complexity)
        
        # Selecionar modelos preferenciais
        preferred_models, fallback_models = self._select_models(task_type, complexity, risk_level)
        
        # Verificar se requer contexto
        requires_context = self._requires_context(user_input_lower, task_type)
        
        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity,
            risk_level=risk_level,
            estimated_tokens=estimated_tokens,
            preferred_models=preferred_models,
            fallback_models=fallback_models,
            requires_context=requires_context,
            execution_context=context or {}
        )
    
    def _determine_task_type(self, user_input: str) -> str:
        """Determinar tipo de tarefa"""
        scores = {}
        
        for task_type, patterns in self.task_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in user_input:
                    score += 1
            scores[task_type] = score
        
        # Retornar tipo com maior pontuação
        if scores:
            return max(scores, key=scores.get)
        
        return 'general'
    
    def _determine_complexity(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Determinar complexidade da tarefa"""
        scores = {}
        
        for complexity_level, indicators in self.complexity_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator in user_input:
                    score += 1
            scores[complexity_level] = score
        
        # Considerar contexto
        if context:
            if context.get('multiple_servers', False):
                scores['high'] = scores.get('high', 0) + 2
            if context.get('production_environment', False):
                scores['high'] = scores.get('high', 0) + 1
        
        # Retornar complexidade com maior pontuação
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return 'medium'  # padrão
    
    def _determine_risk_level(self, user_input: str) -> str:
        """Determinar nível de risco"""
        scores = {}
        
        for risk_level, indicators in self.risk_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator in user_input:
                    score += 1
            scores[risk_level] = score
        
        # Retornar risco com maior pontuação
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return 'low'  # padrão
    
    def _estimate_tokens(self, user_input: str, complexity: str) -> int:
        """Estimar tokens necessários"""
        base_tokens = len(user_input.split()) * 1.3  # Aproximação
        
        complexity_multipliers = {
            'low': 1.5,
            'medium': 3.0,
            'high': 5.0,
            'expert': 8.0
        }
        
        multiplier = complexity_multipliers.get(complexity, 3.0)
        return int(base_tokens * multiplier)
    
    def _select_models(self, task_type: str, complexity: str, risk_level: str) -> Tuple[List[str], List[str]]:
        """Selecionar modelos preferenciais e fallback"""
        
        # Modelos preferenciais por tipo de tarefa
        task_preferences = {
            'coding': ['deepseek', 'wizardcoder'],
            'automation': ['wizardcoder', 'deepseek'],
            'security': ['qwen', 'deepseek'],
            'analysis': ['qwen', 'wizardcoder'],
            'general': ['qwen']
        }
        
        preferred = task_preferences.get(task_type, ['qwen'])
        
        # Ajustar baseado na complexidade
        if complexity in ['high', 'expert']:
            if 'wizardcoder' not in preferred:
                preferred.append('wizardcoder')
        
        # Ajustar baseado no risco
        if risk_level in ['high', 'critical']:
            # Para tarefas de alto risco, preferir Qwen para análise cuidadosa
            if 'qwen' not in preferred:
                preferred.insert(0, 'qwen')
        
        # Modelos fallback
        all_models = ['qwen', 'deepseek', 'wizardcoder']
        fallback = [m for m in all_models if m not in preferred]
        
        return preferred, fallback
    
    def _requires_context(self, user_input: str, task_type: str) -> bool:
        """Verificar se tarefa requer contexto adicional"""
        context_indicators = [
            'servidor', 'servidores', 'máquina', 'sistema',
            'configuração atual', 'estado', 'logs', 'histórico'
        ]
        
        return any(indicator in user_input for indicator in context_indicators)

class ModelOrchestrator:
    """Orquestrador de modelos IA"""
    
    def __init__(self, db_path: str = "/var/lib/alhica-ai/alhica.db"):
        self.db_path = db_path
        self.models_config = self._load_models_config()
        self.health_monitor = ModelHealthMonitor(self.models_config)
        self.task_analyzer = TaskAnalyzer()
        self.response_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=3)
        self._init_database()
    
    def _load_models_config(self) -> Dict[str, ModelConfig]:
        """Carregar configuração dos modelos"""
        return {
            'qwen': ModelConfig(
                name='Qwen 3 25B',
                endpoint='http://localhost:5001/v1/chat/completions',
                model_id='qwen3-25b',
                speciality='general',
                max_tokens=4096,
                temperature=0.1,
                timeout=60,
                health_check_url='http://localhost:5001/health',
                description='Modelo generalista para compreensão de linguagem natural e análise',
                capabilities=['natural_language', 'analysis', 'reasoning', 'explanation'],
                priority=1
            ),
            'deepseek': ModelConfig(
                name='DeepSeek-Coder',
                endpoint='http://localhost:5002/v1/chat/completions',
                model_id='deepseek-coder',
                speciality='coding',
                max_tokens=8192,
                temperature=0.05,
                timeout=90,
                health_check_url='http://localhost:5002/health',
                description='Especialista em geração de código, scripts e soluções técnicas',
                capabilities=['code_generation', 'debugging', 'optimization', 'technical_solutions'],
                priority=2
            ),
            'wizardcoder': ModelConfig(
                name='WizardCoder',
                endpoint='http://localhost:5003/v1/chat/completions',
                model_id='wizardcoder',
                speciality='automation',
                max_tokens=6144,
                temperature=0.1,
                timeout=120,
                health_check_url='http://localhost:5003/health',
                description='Mestre em automação, workflows complexos e orquestração',
                capabilities=['automation', 'workflow_design', 'orchestration', 'complex_tasks'],
                priority=3
            )
        }
    
    def _init_database(self):
        """Inicializar tabelas de base de dados"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS model_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_id TEXT,
                    model_id TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    model_response TEXT NOT NULL,
                    task_type TEXT,
                    complexity TEXT,
                    risk_level TEXT,
                    confidence REAL,
                    processing_time REAL,
                    tokens_used INTEGER,
                    cost_estimate REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                );
                
                CREATE TABLE IF NOT EXISTS model_performance_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    response_time REAL,
                    success BOOLEAN,
                    error_type TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS model_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id INTEGER,
                    user_id INTEGER,
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                    feedback_text TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (interaction_id) REFERENCES model_interactions (id),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_model_interactions_timestamp ON model_interactions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_model_interactions_model ON model_interactions(model_id);
                CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp ON model_performance_log(timestamp);
            """)
    
    async def process_request(self, user_input: str, user_id: int = None, 
                            session_id: str = None, context: Dict[str, Any] = None) -> ModelResponse:
        """Processar pedido do utilizador"""
        
        # Analisar tarefa
        task_analysis = self.task_analyzer.analyze_task(user_input, context)
        
        logger.info(f"Task analysis: {task_analysis.task_type}, {task_analysis.complexity}, {task_analysis.risk_level}")
        
        # Verificar cache
        cache_key = self._generate_cache_key(user_input, context)
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            if datetime.now() - cached_response.timestamp < timedelta(minutes=30):
                logger.info("Returning cached response")
                return cached_response
        
        # Selecionar modelo
        selected_model = self._select_best_model(task_analysis)
        
        if not selected_model:
            raise Exception("Nenhum modelo disponível")
        
        # Processar com modelo selecionado
        try:
            response = await self._query_model(
                selected_model, user_input, task_analysis, context
            )
            
            # Salvar interação
            self._save_interaction(
                user_id, session_id, selected_model, user_input, 
                response, task_analysis, True
            )
            
            # Cache da resposta
            self.response_cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Erro ao processar com {selected_model}: {e}")
            
            # Tentar modelo fallback
            fallback_models = [m for m in task_analysis.fallback_models 
                             if m in self.health_monitor.get_healthy_models()]
            
            if fallback_models:
                try:
                    response = await self._query_model(
                        fallback_models[0], user_input, task_analysis, context
                    )
                    
                    self._save_interaction(
                        user_id, session_id, fallback_models[0], user_input, 
                        response, task_analysis, True
                    )
                    
                    return response
                    
                except Exception as fallback_error:
                    logger.error(f"Erro no fallback {fallback_models[0]}: {fallback_error}")
            
            # Salvar erro
            self._save_interaction(
                user_id, session_id, selected_model, user_input, 
                None, task_analysis, False, str(e)
            )
            
            raise Exception(f"Todos os modelos falharam: {e}")
    
    def _select_best_model(self, task_analysis: TaskAnalysis) -> Optional[str]:
        """Selecionar melhor modelo para a tarefa"""
        healthy_models = self.health_monitor.get_healthy_models()
        
        # Tentar modelos preferenciais primeiro
        for model_id in task_analysis.preferred_models:
            if model_id in healthy_models:
                return model_id
        
        # Tentar modelos fallback
        for model_id in task_analysis.fallback_models:
            if model_id in healthy_models:
                return model_id
        
        # Último recurso: qualquer modelo saudável
        return healthy_models[0] if healthy_models else None
    
    async def _query_model(self, model_id: str, user_input: str, 
                          task_analysis: TaskAnalysis, context: Dict[str, Any] = None) -> ModelResponse:
        """Consultar modelo específico"""
        
        config = self.models_config[model_id]
        start_time = time.time()
        
        # Construir prompt do sistema
        system_prompt = self._build_system_prompt(model_id, task_analysis, context)
        
        # Preparar payload
        payload = {
            "model": config.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            "temperature": config.temperature,
            "max_tokens": min(config.max_tokens, task_analysis.estimated_tokens * 2),
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config.endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.timeout)
                ) as response:
                    
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
                    data = await response.json()
                    
                    processing_time = time.time() - start_time
                    
                    # Extrair resposta
                    ai_response = data['choices'][0]['message']['content']
                    tokens_used = data.get('usage', {}).get('total_tokens', 0)
                    
                    # Calcular confiança baseada no modelo e tarefa
                    confidence = self._calculate_confidence(model_id, task_analysis, ai_response)
                    
                    # Estimar custo (simplificado)
                    cost_estimate = tokens_used * 0.0001  # $0.0001 por token
                    
                    # Registar performance
                    self._log_model_performance(model_id, processing_time, True)
                    
                    return ModelResponse(
                        model_id=model_id,
                        response=ai_response,
                        confidence=confidence,
                        processing_time=processing_time,
                        tokens_used=tokens_used,
                        cost_estimate=cost_estimate,
                        metadata={
                            'task_type': task_analysis.task_type,
                            'complexity': task_analysis.complexity,
                            'risk_level': task_analysis.risk_level,
                            'model_name': config.name
                        },
                        timestamp=datetime.now()
                    )
                    
        except Exception as e:
            processing_time = time.time() - start_time
            self._log_model_performance(model_id, processing_time, False, str(e))
            raise e
    
    def _build_system_prompt(self, model_id: str, task_analysis: TaskAnalysis, 
                           context: Dict[str, Any] = None) -> str:
        """Construir prompt do sistema"""
        
        config = self.models_config[model_id]
        
        base_prompt = f"""Você é {config.name}, parte da Alhica AI - a primeira plataforma do mundo com IA conversacional + execução SSH automática.

Suas especialidades: {', '.join(config.capabilities)}

Tarefa atual:
- Tipo: {task_analysis.task_type}
- Complexidade: {task_analysis.complexity}
- Nível de risco: {task_analysis.risk_level}

Diretrizes importantes:
1. Sempre priorize segurança e boas práticas
2. Forneça explicações claras e detalhadas
3. Para comandos de alto risco, inclua avisos e confirmações
4. Use comandos testados e seguros
5. Implemente verificações de erro quando apropriado
6. Documente suas ações e decisões

"""

        # Adicionar prompts específicos por modelo
        if model_id == 'qwen':
            base_prompt += """
Como modelo generalista, foque em:
- Compreensão precisa da intenção do utilizador
- Análise cuidadosa de riscos e implicações
- Explicações claras e educativas
- Sugestões de melhores práticas
"""
        
        elif model_id == 'deepseek':
            base_prompt += """
Como especialista em código, foque em:
- Geração de código limpo e eficiente
- Implementação de boas práticas de programação
- Debugging e otimização
- Soluções técnicas robustas
- Comentários e documentação no código
"""
        
        elif model_id == 'wizardcoder':
            base_prompt += """
Como mestre em automação, foque em:
- Workflows eficientes e escaláveis
- Orquestração de tarefas complexas
- Automação inteligente
- Tratamento de erros e recuperação
- Monitorização e logging
"""
        
        # Adicionar contexto se disponível
        if context:
            base_prompt += f"\nContexto atual:\n{json.dumps(context, indent=2)}"
        
        return base_prompt
    
    def _calculate_confidence(self, model_id: str, task_analysis: TaskAnalysis, response: str) -> float:
        """Calcular confiança da resposta"""
        confidence = 0.5  # base
        
        # Ajustar baseado na especialidade do modelo
        config = self.models_config[model_id]
        if task_analysis.task_type == config.speciality:
            confidence += 0.3
        
        # Ajustar baseado na complexidade
        complexity_adjustments = {
            'low': 0.2,
            'medium': 0.1,
            'high': -0.1,
            'expert': -0.2
        }
        confidence += complexity_adjustments.get(task_analysis.complexity, 0)
        
        # Ajustar baseado no comprimento e qualidade da resposta
        if len(response) > 100:
            confidence += 0.1
        if len(response) > 500:
            confidence += 0.1
        
        # Verificar se resposta contém código ou comandos estruturados
        if '```' in response or 'sudo' in response or 'systemctl' in response:
            confidence += 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_cache_key(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Gerar chave de cache"""
        cache_data = {
            'input': user_input,
            'context': context or {}
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def _save_interaction(self, user_id: Optional[int], session_id: Optional[str],
                         model_id: str, user_input: str, response: Optional[ModelResponse],
                         task_analysis: TaskAnalysis, success: bool, error_message: str = None):
        """Salvar interação na base de dados"""
        
        with sqlite3.connect(self.db_path) as conn:
            if response:
                conn.execute("""
                    INSERT INTO model_interactions (
                        user_id, session_id, model_id, user_input, model_response,
                        task_type, complexity, risk_level, confidence, processing_time,
                        tokens_used, cost_estimate, success
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, session_id, model_id, user_input, response.response,
                    task_analysis.task_type, task_analysis.complexity, task_analysis.risk_level,
                    response.confidence, response.processing_time, response.tokens_used,
                    response.cost_estimate, success
                ))
            else:
                conn.execute("""
                    INSERT INTO model_interactions (
                        user_id, session_id, model_id, user_input, model_response,
                        task_type, complexity, risk_level, success, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, session_id, model_id, user_input, "",
                    task_analysis.task_type, task_analysis.complexity, task_analysis.risk_level,
                    success, error_message
                ))
    
    def _log_model_performance(self, model_id: str, response_time: float, 
                              success: bool, error_type: str = None):
        """Registar performance do modelo"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO model_performance_log (model_id, response_time, success, error_type)
                VALUES (?, ?, ?, ?)
            """, (model_id, response_time, success, error_type))
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Obter estatísticas dos modelos"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Estatísticas gerais
            cursor.execute("""
                SELECT 
                    model_id,
                    COUNT(*) as total_interactions,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_interactions,
                    AVG(confidence) as avg_confidence,
                    AVG(processing_time) as avg_processing_time,
                    SUM(tokens_used) as total_tokens,
                    SUM(cost_estimate) as total_cost
                FROM model_interactions 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY model_id
            """)
            
            model_stats = {}
            for row in cursor.fetchall():
                model_stats[row['model_id']] = dict(row)
            
            # Adicionar dados do monitor de saúde
            for model_id in self.models_config.keys():
                if model_id not in model_stats:
                    model_stats[model_id] = {
                        'total_interactions': 0,
                        'successful_interactions': 0,
                        'avg_confidence': 0,
                        'avg_processing_time': 0,
                        'total_tokens': 0,
                        'total_cost': 0
                    }
                
                # Adicionar dados de saúde
                health_data = self.health_monitor.health_status.get(model_id, {})
                performance_data = self.health_monitor.get_model_performance(model_id)
                
                model_stats[model_id].update({
                    'health_status': health_data.get('status', 'unknown'),
                    'last_health_check': health_data.get('last_check'),
                    'uptime_percentage': performance_data.get('uptime_percentage', 0),
                    'avg_response_time': performance_data.get('avg_response_time', 0)
                })
            
            return model_stats
    
    def submit_feedback(self, interaction_id: int, user_id: int, 
                       rating: int, feedback_text: str = None):
        """Submeter feedback sobre interação"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO model_feedback (interaction_id, user_id, rating, feedback_text)
                VALUES (?, ?, ?, ?)
            """, (interaction_id, user_id, rating, feedback_text))
    
    def get_health_status(self) -> Dict[str, Any]:
        """Obter status de saúde dos modelos"""
        return {
            'models': self.health_monitor.health_status,
            'healthy_models': self.health_monitor.get_healthy_models(),
            'performance_metrics': self.health_monitor.performance_metrics
        }

# Função principal para integração
async def main():
    """Função principal para teste"""
    orchestrator = ModelOrchestrator()
    
    print("🤖 Alhica AI Model Orchestrator inicializado!")
    
    # Testar análise de tarefa
    test_inputs = [
        "Instalar Docker no servidor de produção",
        "Criar um script Python para backup automático",
        "Verificar status dos serviços",
        "Configurar firewall com regras avançadas",
        "Explicar como funciona o SSH"
    ]
    
    for test_input in test_inputs:
        print(f"\n📝 Testando: {test_input}")
        
        try:
            response = await orchestrator.process_request(test_input, user_id=1)
            print(f"✅ Modelo usado: {response.model_id}")
            print(f"🎯 Confiança: {response.confidence:.2f}")
            print(f"⏱️ Tempo: {response.processing_time:.2f}s")
            print(f"📊 Tokens: {response.tokens_used}")
            print(f"💰 Custo: ${response.cost_estimate:.4f}")
            print(f"📝 Resposta: {response.response[:200]}...")
            
        except Exception as e:
            print(f"❌ Erro: {e}")
    
    # Mostrar estatísticas
    print("\n📊 Estatísticas dos modelos:")
    stats = orchestrator.get_model_stats()
    for model_id, data in stats.items():
        print(f"\n{model_id}:")
        print(f"  Status: {data.get('health_status', 'unknown')}")
        print(f"  Interações: {data.get('total_interactions', 0)}")
        print(f"  Uptime: {data.get('uptime_percentage', 0):.1f}%")

if __name__ == "__main__":
    asyncio.run(main())

