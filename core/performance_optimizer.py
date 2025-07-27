#!/usr/bin/env python3
"""
Alhica AI - Sistema de Otimização de Performance
Otimização automática de recursos, cache inteligente e gestão de performance

Copyright (c) 2024 Alhica AI Team
"""

import os
import sys
import json
import time
import threading
import logging
import psutil
import gc
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
from transformers import AutoTokenizer
import redis
import sqlite3
from contextlib import contextmanager

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/alhica-ai/performance_optimizer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Métricas de performance do sistema"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: float
    disk_io_read: float
    disk_io_write: float
    network_sent: float
    network_recv: float
    model_response_time: float
    cache_hit_rate: float
    active_connections: int

@dataclass
class CacheEntry:
    """Entrada do cache"""
    key: str
    value: Any
    timestamp: float
    access_count: int
    last_access: float
    size_bytes: int
    ttl: Optional[float] = None

@dataclass
class OptimizationRule:
    """Regra de otimização"""
    name: str
    condition: str
    action: str
    priority: int
    enabled: bool = True

class SystemMonitor:
    """Monitor de sistema em tempo real"""
    
    def __init__(self, sample_interval: float = 1.0, history_size: int = 3600):
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.running = False
        self.monitor_thread = None
        self.callbacks = []
        
        # Baseline inicial
        self.baseline_metrics = self._collect_metrics()
        
    def _collect_metrics(self) -> PerformanceMetrics:
        """Coletar métricas atuais do sistema"""
        # CPU e Memória
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # GPU
        gpu_memory_percent = 0.0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated()
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
        
        # Disco I/O
        disk_io = psutil.disk_io_counters()
        disk_io_read = disk_io.read_bytes if disk_io else 0
        disk_io_write = disk_io.write_bytes if disk_io else 0
        
        # Rede
        network_io = psutil.net_io_counters()
        network_sent = network_io.bytes_sent if network_io else 0
        network_recv = network_io.bytes_recv if network_io else 0
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_memory_percent=gpu_memory_percent,
            disk_io_read=disk_io_read,
            disk_io_write=disk_io_write,
            network_sent=network_sent,
            network_recv=network_recv,
            model_response_time=0.0,  # Será atualizado externamente
            cache_hit_rate=0.0,       # Será atualizado externamente
            active_connections=0      # Será atualizado externamente
        )
    
    def start_monitoring(self):
        """Iniciar monitorização contínua"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔍 Monitor de sistema iniciado")
    
    def stop_monitoring(self):
        """Parar monitorização"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Monitor de sistema parado")
    
    def _monitor_loop(self):
        """Loop principal de monitorização"""
        while self.running:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Executar callbacks
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Erro em callback de monitorização: {e}")
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"Erro no loop de monitorização: {e}")
                time.sleep(self.sample_interval)
    
    def add_callback(self, callback):
        """Adicionar callback para métricas"""
        self.callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Obter métricas atuais"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return self._collect_metrics()
    
    def get_average_metrics(self, window_seconds: int = 300) -> Optional[PerformanceMetrics]:
        """Obter métricas médias numa janela de tempo"""
        if not self.metrics_history:
            return None
        
        cutoff_time = time.time() - window_seconds
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        # Calcular médias
        avg_metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            memory_percent=sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            gpu_memory_percent=sum(m.gpu_memory_percent for m in recent_metrics) / len(recent_metrics),
            disk_io_read=sum(m.disk_io_read for m in recent_metrics) / len(recent_metrics),
            disk_io_write=sum(m.disk_io_write for m in recent_metrics) / len(recent_metrics),
            network_sent=sum(m.network_sent for m in recent_metrics) / len(recent_metrics),
            network_recv=sum(m.network_recv for m in recent_metrics) / len(recent_metrics),
            model_response_time=sum(m.model_response_time for m in recent_metrics) / len(recent_metrics),
            cache_hit_rate=sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics),
            active_connections=int(sum(m.active_connections for m in recent_metrics) / len(recent_metrics))
        )
        
        return avg_metrics
    
    def detect_anomalies(self, threshold_multiplier: float = 2.0) -> List[str]:
        """Detectar anomalias nas métricas"""
        if len(self.metrics_history) < 10:
            return []
        
        current = self.get_current_metrics()
        baseline = self.get_average_metrics(3600)  # Última hora
        
        if not current or not baseline:
            return []
        
        anomalies = []
        
        # Verificar CPU
        if current.cpu_percent > baseline.cpu_percent * threshold_multiplier:
            anomalies.append(f"CPU alta: {current.cpu_percent:.1f}% (baseline: {baseline.cpu_percent:.1f}%)")
        
        # Verificar Memória
        if current.memory_percent > baseline.memory_percent * threshold_multiplier:
            anomalies.append(f"Memória alta: {current.memory_percent:.1f}% (baseline: {baseline.memory_percent:.1f}%)")
        
        # Verificar GPU
        if current.gpu_memory_percent > baseline.gpu_memory_percent * threshold_multiplier:
            anomalies.append(f"GPU alta: {current.gpu_memory_percent:.1f}% (baseline: {baseline.gpu_memory_percent:.1f}%)")
        
        # Verificar tempo de resposta
        if current.model_response_time > baseline.model_response_time * threshold_multiplier:
            anomalies.append(f"Resposta lenta: {current.model_response_time:.2f}s (baseline: {baseline.model_response_time:.2f}s)")
        
        return anomalies

class IntelligentCache:
    """Sistema de cache inteligente multicamada"""
    
    def __init__(self, max_memory_mb: int = 1024, redis_url: str = "redis://localhost:6379/1"):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_bytes = 0
        
        # Cache em memória (L1)
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.access_order = deque()  # Para LRU
        
        # Cache Redis (L2)
        try:
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=False)
            self.redis_client.ping()
            self.redis_available = True
            logger.info("✅ Cache Redis conectado")
        except Exception as e:
            logger.warning(f"⚠️ Redis não disponível: {e}")
            self.redis_available = False
            self.redis_client = None
        
        # Cache em disco (L3)
        self.disk_cache_dir = Path("/var/cache/alhica-ai")
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Estatísticas
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_hits': 0,
            'redis_hits': 0,
            'disk_hits': 0
        }
        
        # Lock para thread safety
        self.lock = threading.RLock()
    
    def _generate_key(self, prompt: str, model: str, params: Dict) -> str:
        """Gerar chave única para cache"""
        # Normalizar parâmetros
        normalized_params = {
            'max_tokens': params.get('max_tokens', 2048),
            'temperature': round(params.get('temperature', 0.1), 2),
            'top_p': round(params.get('top_p', 0.9), 2)
        }
        
        # Criar string única
        cache_string = f"{model}:{prompt}:{json.dumps(normalized_params, sort_keys=True)}"
        
        # Hash para chave compacta
        return hashlib.sha256(cache_string.encode()).hexdigest()[:32]
    
    def _calculate_size(self, value: Any) -> int:
        """Calcular tamanho aproximado de um valor"""
        if isinstance(value, str):
            return len(value.encode('utf-8'))
        elif isinstance(value, dict):
            return len(json.dumps(value).encode('utf-8'))
        else:
            return sys.getsizeof(value)
    
    def _evict_lru(self):
        """Remover entradas menos recentemente usadas"""
        with self.lock:
            while self.current_memory_bytes > self.max_memory_bytes * 0.8:  # 80% do limite
                if not self.access_order:
                    break
                
                # Remover entrada mais antiga
                oldest_key = self.access_order.popleft()
                if oldest_key in self.memory_cache:
                    entry = self.memory_cache[oldest_key]
                    self.current_memory_bytes -= entry.size_bytes
                    del self.memory_cache[oldest_key]
                    self.stats['evictions'] += 1
    
    def get(self, prompt: str, model: str, params: Dict) -> Optional[Any]:
        """Obter valor do cache"""
        key = self._generate_key(prompt, model, params)
        
        with self.lock:
            # L1: Cache em memória
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                # Verificar TTL
                if entry.ttl and time.time() > entry.timestamp + entry.ttl:
                    del self.memory_cache[key]
                    self.current_memory_bytes -= entry.size_bytes
                else:
                    # Hit em memória
                    entry.access_count += 1
                    entry.last_access = time.time()
                    
                    # Mover para final da fila LRU
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.access_order.append(key)
                    
                    self.stats['hits'] += 1
                    self.stats['memory_hits'] += 1
                    
                    logger.debug(f"Cache hit (memória): {key[:8]}...")
                    return entry.value
            
            # L2: Cache Redis
            if self.redis_available:
                try:
                    redis_value = self.redis_client.get(f"alhica:cache:{key}")
                    if redis_value:
                        value = json.loads(redis_value.decode('utf-8'))
                        
                        # Promover para cache em memória
                        self._set_memory_cache(key, value, ttl=3600)  # 1 hora
                        
                        self.stats['hits'] += 1
                        self.stats['redis_hits'] += 1
                        
                        logger.debug(f"Cache hit (Redis): {key[:8]}...")
                        return value
                        
                except Exception as e:
                    logger.warning(f"Erro ao acessar Redis: {e}")
            
            # L3: Cache em disco
            disk_file = self.disk_cache_dir / f"{key}.json"
            if disk_file.exists():
                try:
                    with open(disk_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    # Verificar TTL
                    if cache_data.get('ttl') and time.time() > cache_data['timestamp'] + cache_data['ttl']:
                        disk_file.unlink()
                    else:
                        value = cache_data['value']
                        
                        # Promover para caches superiores
                        if self.redis_available:
                            try:
                                self.redis_client.setex(
                                    f"alhica:cache:{key}",
                                    3600,  # 1 hora
                                    json.dumps(value)
                                )
                            except Exception as e:
                                logger.warning(f"Erro ao promover para Redis: {e}")
                        
                        self._set_memory_cache(key, value, ttl=3600)
                        
                        self.stats['hits'] += 1
                        self.stats['disk_hits'] += 1
                        
                        logger.debug(f"Cache hit (disco): {key[:8]}...")
                        return value
                        
                except Exception as e:
                    logger.warning(f"Erro ao acessar cache em disco: {e}")
                    if disk_file.exists():
                        disk_file.unlink()
            
            # Miss
            self.stats['misses'] += 1
            return None
    
    def _set_memory_cache(self, key: str, value: Any, ttl: Optional[float] = None):
        """Definir entrada no cache em memória"""
        size_bytes = self._calculate_size(value)
        
        # Verificar se cabe na memória
        if size_bytes > self.max_memory_bytes:
            logger.warning(f"Valor muito grande para cache: {size_bytes} bytes")
            return
        
        # Fazer espaço se necessário
        while self.current_memory_bytes + size_bytes > self.max_memory_bytes:
            self._evict_lru()
        
        # Criar entrada
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            access_count=1,
            last_access=time.time(),
            size_bytes=size_bytes,
            ttl=ttl
        )
        
        # Adicionar ao cache
        self.memory_cache[key] = entry
        self.current_memory_bytes += size_bytes
        
        # Atualizar ordem LRU
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def set(self, prompt: str, model: str, params: Dict, value: Any, ttl: Optional[float] = None):
        """Definir valor no cache"""
        key = self._generate_key(prompt, model, params)
        
        with self.lock:
            # L1: Cache em memória
            self._set_memory_cache(key, value, ttl)
            
            # L2: Cache Redis
            if self.redis_available:
                try:
                    redis_ttl = int(ttl) if ttl else 7200  # 2 horas padrão
                    self.redis_client.setex(
                        f"alhica:cache:{key}",
                        redis_ttl,
                        json.dumps(value)
                    )
                except Exception as e:
                    logger.warning(f"Erro ao salvar no Redis: {e}")
            
            # L3: Cache em disco (para valores importantes)
            if ttl is None or ttl > 3600:  # Apenas para cache de longa duração
                try:
                    disk_file = self.disk_cache_dir / f"{key}.json"
                    cache_data = {
                        'value': value,
                        'timestamp': time.time(),
                        'ttl': ttl
                    }
                    
                    with open(disk_file, 'w') as f:
                        json.dump(cache_data, f)
                        
                except Exception as e:
                    logger.warning(f"Erro ao salvar cache em disco: {e}")
    
    def clear(self):
        """Limpar todo o cache"""
        with self.lock:
            # Limpar memória
            self.memory_cache.clear()
            self.access_order.clear()
            self.current_memory_bytes = 0
            
            # Limpar Redis
            if self.redis_available:
                try:
                    keys = self.redis_client.keys("alhica:cache:*")
                    if keys:
                        self.redis_client.delete(*keys)
                except Exception as e:
                    logger.warning(f"Erro ao limpar Redis: {e}")
            
            # Limpar disco
            try:
                for file in self.disk_cache_dir.glob("*.json"):
                    file.unlink()
            except Exception as e:
                logger.warning(f"Erro ao limpar cache em disco: {e}")
            
            logger.info("🗑️ Cache limpo")
    
    def get_stats(self) -> Dict:
        """Obter estatísticas do cache"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / max(1, total_requests)) * 100
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'memory_hits': self.stats['memory_hits'],
            'redis_hits': self.stats['redis_hits'],
            'disk_hits': self.stats['disk_hits'],
            'memory_usage_mb': self.current_memory_bytes / (1024 * 1024),
            'memory_entries': len(self.memory_cache),
            'redis_available': self.redis_available
        }

class ResourceOptimizer:
    """Otimizador de recursos do sistema"""
    
    def __init__(self, monitor: SystemMonitor, cache: IntelligentCache):
        self.monitor = monitor
        self.cache = cache
        self.optimization_rules = self._load_optimization_rules()
        self.running = False
        self.optimizer_thread = None
        
        # Histórico de otimizações
        self.optimization_history = deque(maxlen=1000)
        
        # Configurações adaptativas
        self.adaptive_config = {
            'cache_ttl_base': 3600,
            'memory_threshold': 80.0,
            'cpu_threshold': 80.0,
            'gpu_threshold': 90.0,
            'response_time_threshold': 5.0
        }
    
    def _load_optimization_rules(self) -> List[OptimizationRule]:
        """Carregar regras de otimização"""
        return [
            OptimizationRule(
                name="high_memory_usage",
                condition="memory_percent > 85",
                action="clear_cache_partial",
                priority=1
            ),
            OptimizationRule(
                name="high_cpu_usage",
                condition="cpu_percent > 90",
                action="reduce_model_precision",
                priority=2
            ),
            OptimizationRule(
                name="slow_response",
                condition="model_response_time > 10",
                action="optimize_model_loading",
                priority=3
            ),
            OptimizationRule(
                name="low_cache_hit_rate",
                condition="cache_hit_rate < 30",
                action="adjust_cache_strategy",
                priority=4
            ),
            OptimizationRule(
                name="gpu_memory_high",
                condition="gpu_memory_percent > 95",
                action="offload_to_cpu",
                priority=1
            )
        ]
    
    def start_optimization(self):
        """Iniciar otimização automática"""
        if self.running:
            return
        
        self.running = True
        self.optimizer_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimizer_thread.start()
        logger.info("🚀 Otimizador de recursos iniciado")
    
    def stop_optimization(self):
        """Parar otimização"""
        self.running = False
        if self.optimizer_thread:
            self.optimizer_thread.join(timeout=5)
        logger.info("🛑 Otimizador de recursos parado")
    
    def _optimization_loop(self):
        """Loop principal de otimização"""
        while self.running:
            try:
                self._run_optimization_cycle()
                time.sleep(30)  # Verificar a cada 30 segundos
                
            except Exception as e:
                logger.error(f"Erro no ciclo de otimização: {e}")
                time.sleep(60)  # Esperar mais tempo em caso de erro
    
    def _run_optimization_cycle(self):
        """Executar um ciclo de otimização"""
        current_metrics = self.monitor.get_current_metrics()
        if not current_metrics:
            return
        
        cache_stats = self.cache.get_stats()
        
        # Criar contexto para avaliação de regras
        context = {
            'memory_percent': current_metrics.memory_percent,
            'cpu_percent': current_metrics.cpu_percent,
            'gpu_memory_percent': current_metrics.gpu_memory_percent,
            'model_response_time': current_metrics.model_response_time,
            'cache_hit_rate': cache_stats['hit_rate']
        }
        
        # Avaliar regras de otimização
        triggered_rules = []
        for rule in self.optimization_rules:
            if not rule.enabled:
                continue
            
            try:
                if eval(rule.condition, {"__builtins__": {}}, context):
                    triggered_rules.append(rule)
            except Exception as e:
                logger.warning(f"Erro ao avaliar regra {rule.name}: {e}")
        
        # Executar ações por prioridade
        triggered_rules.sort(key=lambda r: r.priority)
        
        for rule in triggered_rules:
            try:
                self._execute_optimization_action(rule.action, context)
                
                # Registar otimização
                self.optimization_history.append({
                    'timestamp': time.time(),
                    'rule': rule.name,
                    'action': rule.action,
                    'context': context.copy()
                })
                
                logger.info(f"🔧 Otimização executada: {rule.action} (regra: {rule.name})")
                
            except Exception as e:
                logger.error(f"Erro ao executar ação {rule.action}: {e}")
    
    def _execute_optimization_action(self, action: str, context: Dict):
        """Executar ação de otimização"""
        if action == "clear_cache_partial":
            # Limpar 30% do cache menos usado
            self._clear_cache_partial(0.3)
            
        elif action == "reduce_model_precision":
            # Reduzir precisão dos modelos (implementar conforme necessário)
            logger.info("Sugestão: Considerar reduzir precisão dos modelos")
            
        elif action == "optimize_model_loading":
            # Otimizar carregamento de modelos
            self._optimize_model_loading()
            
        elif action == "adjust_cache_strategy":
            # Ajustar estratégia de cache
            self._adjust_cache_strategy(context)
            
        elif action == "offload_to_cpu":
            # Sugerir offload para CPU
            logger.info("Sugestão: Considerar mover modelos para CPU")
            
        else:
            logger.warning(f"Ação de otimização desconhecida: {action}")
    
    def _clear_cache_partial(self, percentage: float):
        """Limpar parcialmente o cache"""
        # Implementar limpeza parcial baseada em LRU
        with self.cache.lock:
            entries_to_remove = int(len(self.cache.memory_cache) * percentage)
            
            # Ordenar por último acesso
            sorted_entries = sorted(
                self.cache.memory_cache.items(),
                key=lambda x: x[1].last_access
            )
            
            for i in range(min(entries_to_remove, len(sorted_entries))):
                key, entry = sorted_entries[i]
                del self.cache.memory_cache[key]
                self.cache.current_memory_bytes -= entry.size_bytes
                
                if key in self.cache.access_order:
                    self.cache.access_order.remove(key)
            
            logger.info(f"🗑️ Cache parcialmente limpo: {entries_to_remove} entradas removidas")
    
    def _optimize_model_loading(self):
        """Otimizar carregamento de modelos"""
        # Forçar garbage collection
        gc.collect()
        
        # Limpar cache da GPU se disponível
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🧹 Limpeza de memória executada")
    
    def _adjust_cache_strategy(self, context: Dict):
        """Ajustar estratégia de cache"""
        hit_rate = context.get('cache_hit_rate', 0)
        
        if hit_rate < 20:
            # Taxa muito baixa - aumentar TTL
            self.adaptive_config['cache_ttl_base'] = min(7200, self.adaptive_config['cache_ttl_base'] * 1.5)
            logger.info(f"📈 TTL do cache aumentado para {self.adaptive_config['cache_ttl_base']}s")
            
        elif hit_rate > 80:
            # Taxa alta - pode reduzir TTL para economizar memória
            self.adaptive_config['cache_ttl_base'] = max(1800, self.adaptive_config['cache_ttl_base'] * 0.8)
            logger.info(f"📉 TTL do cache reduzido para {self.adaptive_config['cache_ttl_base']}s")
    
    def get_optimization_stats(self) -> Dict:
        """Obter estatísticas de otimização"""
        recent_optimizations = [
            opt for opt in self.optimization_history
            if time.time() - opt['timestamp'] < 3600  # Última hora
        ]
        
        action_counts = defaultdict(int)
        for opt in recent_optimizations:
            action_counts[opt['action']] += 1
        
        return {
            'total_optimizations': len(self.optimization_history),
            'recent_optimizations': len(recent_optimizations),
            'action_counts': dict(action_counts),
            'adaptive_config': self.adaptive_config.copy(),
            'rules_enabled': sum(1 for rule in self.optimization_rules if rule.enabled),
            'rules_total': len(self.optimization_rules)
        }

class PerformanceOptimizer:
    """Sistema principal de otimização de performance"""
    
    def __init__(self, base_path: str = "/opt/alhica-ai"):
        self.base_path = Path(base_path)
        
        # Componentes principais
        self.monitor = SystemMonitor()
        self.cache = IntelligentCache()
        self.resource_optimizer = ResourceOptimizer(self.monitor, self.cache)
        
        # Estado
        self.running = False
        
        # Configurações
        self.config = {
            'monitoring_interval': 1.0,
            'optimization_interval': 30.0,
            'cache_max_memory_mb': 1024,
            'enable_auto_optimization': True,
            'enable_predictive_caching': True
        }
        
        logger.info("🚀 PerformanceOptimizer inicializado")
    
    def start(self):
        """Iniciar todos os componentes"""
        if self.running:
            return
        
        self.running = True
        
        # Iniciar monitor
        self.monitor.start_monitoring()
        
        # Iniciar otimizador se habilitado
        if self.config['enable_auto_optimization']:
            self.resource_optimizer.start_optimization()
        
        logger.info("✅ PerformanceOptimizer iniciado")
    
    def stop(self):
        """Parar todos os componentes"""
        if not self.running:
            return
        
        self.running = False
        
        # Parar componentes
        self.monitor.stop_monitoring()
        self.resource_optimizer.stop_optimization()
        
        logger.info("🛑 PerformanceOptimizer parado")
    
    def get_cached_response(self, prompt: str, model: str, params: Dict) -> Optional[Any]:
        """Obter resposta do cache"""
        return self.cache.get(prompt, model, params)
    
    def cache_response(self, prompt: str, model: str, params: Dict, response: Any, ttl: Optional[float] = None):
        """Armazenar resposta no cache"""
        if ttl is None:
            ttl = self.resource_optimizer.adaptive_config['cache_ttl_base']
        
        self.cache.set(prompt, model, params, response, ttl)
    
    def get_system_health(self) -> Dict:
        """Obter saúde geral do sistema"""
        current_metrics = self.monitor.get_current_metrics()
        cache_stats = self.cache.get_stats()
        optimization_stats = self.resource_optimizer.get_optimization_stats()
        anomalies = self.monitor.detect_anomalies()
        
        # Determinar status geral
        status = "healthy"
        if anomalies:
            status = "warning"
        
        if current_metrics:
            if (current_metrics.cpu_percent > 95 or 
                current_metrics.memory_percent > 95 or
                current_metrics.gpu_memory_percent > 98):
                status = "critical"
        
        return {
            'status': status,
            'anomalies': anomalies,
            'current_metrics': asdict(current_metrics) if current_metrics else None,
            'cache_stats': cache_stats,
            'optimization_stats': optimization_stats,
            'uptime': time.time() - self.monitor.baseline_metrics.timestamp if self.monitor.baseline_metrics else 0
        }
    
    def optimize_now(self) -> Dict:
        """Executar otimização manual"""
        logger.info("🔧 Executando otimização manual...")
        
        # Executar ciclo de otimização
        self.resource_optimizer._run_optimization_cycle()
        
        # Limpeza adicional
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Retornar estatísticas atualizadas
        return self.get_system_health()

def main():
    """Função principal para teste standalone"""
    optimizer = PerformanceOptimizer()
    
    try:
        # Iniciar otimizador
        optimizer.start()
        
        # Simular algumas operações
        logger.info("🧪 Testando sistema de otimização...")
        
        # Teste de cache
        test_response = {"text": "Resposta de teste", "tokens": 10}
        optimizer.cache_response("teste", "qwen", {"temperature": 0.1}, test_response)
        
        cached = optimizer.get_cached_response("teste", "qwen", {"temperature": 0.1})
        if cached:
            logger.info("✅ Cache funcionando corretamente")
        
        # Mostrar estatísticas
        health = optimizer.get_system_health()
        logger.info(f"Saúde do sistema: {health['status']}")
        logger.info(f"Cache hit rate: {health['cache_stats']['hit_rate']:.1f}%")
        
        # Manter rodando por um tempo para teste
        time.sleep(60)
        
    except KeyboardInterrupt:
        logger.info("Interrompido pelo utilizador")
    finally:
        optimizer.stop()

if __name__ == "__main__":
    main()

