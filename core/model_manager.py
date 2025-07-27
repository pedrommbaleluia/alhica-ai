#!/usr/bin/env python3
"""
Alhica AI - Gestor de Modelos IA em Produção
Sistema avançado de carregamento, gestão e otimização de modelos IA

Copyright (c) 2024 Alhica AI Team
"""

import os
import sys
import json
import time
import threading
import logging
import traceback
import gc
import psutil
import signal
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager
from queue import Queue, Empty
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    GenerationConfig, StoppingCriteria, StoppingCriteriaList
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import numpy as np

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/alhica-ai/model_manager.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelState:
    """Estado de um modelo IA"""
    name: str
    status: str  # 'unloaded', 'loading', 'loaded', 'error'
    load_time: Optional[float] = None
    memory_usage_gb: float = 0.0
    gpu_memory_gb: float = 0.0
    last_used: Optional[float] = None
    error_message: Optional[str] = None
    generation_count: int = 0
    total_tokens: int = 0
    avg_response_time: float = 0.0

@dataclass
class GenerationRequest:
    """Solicitação de geração de texto"""
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    stop_sequences: List[str] = None
    context: Optional[str] = None
    speciality_hint: Optional[str] = None

@dataclass
class GenerationResponse:
    """Resposta de geração de texto"""
    text: str
    tokens_generated: int
    generation_time: float
    model_used: str
    confidence: float = 0.0
    finish_reason: str = "stop"

class CustomStoppingCriteria(StoppingCriteria):
    """Critério de parada personalizado"""
    
    def __init__(self, stop_sequences: List[str], tokenizer):
        self.stop_sequences = stop_sequences or []
        self.tokenizer = tokenizer
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_sequences:
            return False
            
        # Decodificar últimos tokens
        last_tokens = input_ids[0][-50:].tolist()  # Últimos 50 tokens
        decoded = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
        
        # Verificar se alguma sequência de parada está presente
        for stop_seq in self.stop_sequences:
            if stop_seq in decoded:
                return True
                
        return False

class ModelMemoryManager:
    """Gestor de memória para modelos"""
    
    def __init__(self, max_memory_gb: float = None):
        self.max_memory_gb = max_memory_gb or (psutil.virtual_memory().total / (1024**3) * 0.8)
        self.current_usage_gb = 0.0
        self.gpu_max_memory_gb = self._get_gpu_memory()
        self.gpu_current_usage_gb = 0.0
        
    def _get_gpu_memory(self) -> float:
        """Obter memória total da GPU"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0.0
    
    def can_load_model(self, estimated_size_gb: float) -> Tuple[bool, str]:
        """Verificar se é possível carregar um modelo"""
        # Verificar RAM
        available_ram = self.max_memory_gb - self.current_usage_gb
        if available_ram < estimated_size_gb:
            return False, f"RAM insuficiente: {available_ram:.1f}GB disponível, {estimated_size_gb:.1f}GB necessário"
        
        # Verificar GPU se disponível
        if torch.cuda.is_available():
            available_gpu = self.gpu_max_memory_gb - self.gpu_current_usage_gb
            if available_gpu < estimated_size_gb * 0.7:  # 70% do modelo na GPU
                return False, f"VRAM insuficiente: {available_gpu:.1f}GB disponível, {estimated_size_gb * 0.7:.1f}GB necessário"
        
        return True, "Memória suficiente"
    
    def register_model_load(self, model_name: str, memory_usage_gb: float, gpu_usage_gb: float = 0.0):
        """Registar carregamento de modelo"""
        self.current_usage_gb += memory_usage_gb
        self.gpu_current_usage_gb += gpu_usage_gb
        logger.info(f"Modelo {model_name} carregado: RAM {memory_usage_gb:.1f}GB, GPU {gpu_usage_gb:.1f}GB")
        logger.info(f"Uso total: RAM {self.current_usage_gb:.1f}/{self.max_memory_gb:.1f}GB, "
                   f"GPU {self.gpu_current_usage_gb:.1f}/{self.gpu_max_memory_gb:.1f}GB")
    
    def register_model_unload(self, model_name: str, memory_usage_gb: float, gpu_usage_gb: float = 0.0):
        """Registar descarregamento de modelo"""
        self.current_usage_gb = max(0, self.current_usage_gb - memory_usage_gb)
        self.gpu_current_usage_gb = max(0, self.gpu_current_usage_gb - gpu_usage_gb)
        logger.info(f"Modelo {model_name} descarregado: liberou RAM {memory_usage_gb:.1f}GB, GPU {gpu_usage_gb:.1f}GB")
    
    def get_memory_stats(self) -> Dict:
        """Obter estatísticas de memória"""
        ram_percent = (self.current_usage_gb / self.max_memory_gb) * 100
        gpu_percent = (self.gpu_current_usage_gb / self.gpu_max_memory_gb) * 100 if self.gpu_max_memory_gb > 0 else 0
        
        return {
            'ram_used_gb': self.current_usage_gb,
            'ram_max_gb': self.max_memory_gb,
            'ram_percent': ram_percent,
            'gpu_used_gb': self.gpu_current_usage_gb,
            'gpu_max_gb': self.gpu_max_memory_gb,
            'gpu_percent': gpu_percent,
            'system_ram_percent': psutil.virtual_memory().percent
        }

class ModelInstance:
    """Instância de um modelo IA carregado"""
    
    def __init__(self, name: str, config: Dict, model_path: str):
        self.name = name
        self.config = config
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded = False
        self.load_time = None
        self.memory_usage = 0.0
        self.gpu_memory_usage = 0.0
        self.generation_count = 0
        self.total_tokens = 0
        self.response_times = []
        self.lock = threading.RLock()
        
    def load(self) -> bool:
        """Carregar modelo em memória"""
        with self.lock:
            if self.loaded:
                return True
                
            try:
                start_time = time.time()
                logger.info(f"🔄 Carregando modelo {self.name}...")
                
                # Carregar configurações otimizadas
                opt_config_path = Path(self.model_path) / "alhica_config.json"
                load_kwargs = {}
                
                if opt_config_path.exists():
                    with open(opt_config_path, 'r') as f:
                        load_kwargs = json.load(f)
                        logger.info(f"Configurações otimizadas carregadas: {load_kwargs}")
                
                # Carregar tokenizer
                logger.info("📝 Carregando tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    use_fast=True
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Carregar configuração do modelo
                logger.info("⚙️ Carregando configuração...")
                model_config = AutoConfig.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                
                # Preparar argumentos de carregamento
                load_kwargs.update({
                    'pretrained_model_name_or_path': self.model_path,
                    'config': model_config,
                    'trust_remote_code': True,
                    'torch_dtype': torch.float16 if self.device == "cuda" else torch.float32,
                    'low_cpu_mem_usage': True
                })
                
                # Configurar device_map baseado no sistema
                if self.device == "cuda" and torch.cuda.is_available():
                    if torch.cuda.device_count() > 1:
                        load_kwargs['device_map'] = 'auto'
                    else:
                        load_kwargs['device_map'] = {'': 0}
                else:
                    load_kwargs['device_map'] = 'cpu'
                
                # Carregar modelo
                logger.info("🧠 Carregando modelo (pode demorar alguns minutos)...")
                
                # Tentar carregamento com diferentes estratégias
                try:
                    # Estratégia 1: Carregamento normal
                    self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
                except Exception as e1:
                    logger.warning(f"Carregamento normal falhou: {e1}")
                    try:
                        # Estratégia 2: Carregamento com quantização
                        load_kwargs['load_in_8bit'] = True
                        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
                        logger.info("✅ Modelo carregado com quantização 8-bit")
                    except Exception as e2:
                        logger.warning(f"Carregamento com quantização falhou: {e2}")
                        try:
                            # Estratégia 3: Carregamento apenas CPU
                            load_kwargs = {
                                'pretrained_model_name_or_path': self.model_path,
                                'config': model_config,
                                'trust_remote_code': True,
                                'torch_dtype': torch.float32,
                                'device_map': 'cpu',
                                'low_cpu_mem_usage': True
                            }
                            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
                            self.device = "cpu"
                            logger.info("✅ Modelo carregado em CPU")
                        except Exception as e3:
                            logger.error(f"Todas as estratégias de carregamento falharam: {e3}")
                            raise e3
                
                # Configurar para inferência
                self.model.eval()
                
                # Configuração de geração
                self.generation_config = GenerationConfig(
                    max_new_tokens=2048,
                    temperature=0.1,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Calcular uso de memória
                self._calculate_memory_usage()
                
                self.loaded = True
                self.load_time = time.time() - start_time
                
                logger.info(f"✅ Modelo {self.name} carregado com sucesso!")
                logger.info(f"⏱️ Tempo de carregamento: {self.load_time:.1f}s")
                logger.info(f"💾 Uso de memória: RAM {self.memory_usage:.1f}GB, GPU {self.gpu_memory_usage:.1f}GB")
                logger.info(f"🖥️ Dispositivo: {self.device}")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Erro ao carregar modelo {self.name}: {e}")
                logger.error(traceback.format_exc())
                self.loaded = False
                return False
    
    def _calculate_memory_usage(self):
        """Calcular uso de memória do modelo"""
        try:
            # Memória RAM
            process = psutil.Process()
            self.memory_usage = process.memory_info().rss / (1024**3)
            
            # Memória GPU
            if self.device == "cuda" and torch.cuda.is_available():
                self.gpu_memory_usage = torch.cuda.memory_allocated() / (1024**3)
            
        except Exception as e:
            logger.warning(f"Erro ao calcular uso de memória: {e}")
    
    def unload(self):
        """Descarregar modelo da memória"""
        with self.lock:
            if not self.loaded:
                return
                
            try:
                logger.info(f"🗑️ Descarregando modelo {self.name}...")
                
                # Limpar modelo
                if self.model is not None:
                    del self.model
                    self.model = None
                
                if self.tokenizer is not None:
                    del self.tokenizer
                    self.tokenizer = None
                
                # Limpar cache da GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Forçar garbage collection
                gc.collect()
                
                self.loaded = False
                logger.info(f"✅ Modelo {self.name} descarregado")
                
            except Exception as e:
                logger.error(f"❌ Erro ao descarregar modelo {self.name}: {e}")
    
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Gerar texto usando o modelo"""
        with self.lock:
            if not self.loaded:
                raise Exception(f"Modelo {self.name} não está carregado")
            
            start_time = time.time()
            
            try:
                # Preparar prompt
                prompt = self._prepare_prompt(request.prompt, request.context, request.speciality_hint)
                
                # Tokenizar
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                    padding=False
                )
                
                # Mover para dispositivo
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Configurar geração
                generation_config = GenerationConfig(
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Critérios de parada
                stopping_criteria = None
                if request.stop_sequences:
                    stopping_criteria = StoppingCriteriaList([
                        CustomStoppingCriteria(request.stop_sequences, self.tokenizer)
                    ])
                
                # Gerar
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config,
                        stopping_criteria=stopping_criteria,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                # Decodificar resposta
                generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
                response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Calcular confiança (média dos scores)
                confidence = 0.0
                if hasattr(outputs, 'scores') and outputs.scores:
                    scores = torch.stack(outputs.scores, dim=1)
                    probs = torch.softmax(scores, dim=-1)
                    max_probs = torch.max(probs, dim=-1)[0]
                    confidence = torch.mean(max_probs).item()
                
                # Determinar razão de parada
                finish_reason = "stop"
                if len(generated_tokens) >= request.max_tokens:
                    finish_reason = "length"
                elif request.stop_sequences and any(seq in response_text for seq in request.stop_sequences):
                    finish_reason = "stop_sequence"
                
                generation_time = time.time() - start_time
                tokens_generated = len(generated_tokens)
                
                # Atualizar estatísticas
                self.generation_count += 1
                self.total_tokens += tokens_generated
                self.response_times.append(generation_time)
                
                # Manter apenas últimas 100 medições
                if len(self.response_times) > 100:
                    self.response_times = self.response_times[-100:]
                
                logger.info(f"✅ Geração concluída: {tokens_generated} tokens em {generation_time:.2f}s "
                           f"({tokens_generated/generation_time:.1f} tokens/s)")
                
                return GenerationResponse(
                    text=response_text.strip(),
                    tokens_generated=tokens_generated,
                    generation_time=generation_time,
                    model_used=self.name,
                    confidence=confidence,
                    finish_reason=finish_reason
                )
                
            except Exception as e:
                logger.error(f"❌ Erro na geração com modelo {self.name}: {e}")
                raise
    
    def _prepare_prompt(self, prompt: str, context: str = None, speciality_hint: str = None) -> str:
        """Preparar prompt baseado na especialidade do modelo"""
        # Sistema de prompts baseado na especialidade
        system_prompts = {
            'general_language': "Você é um assistente IA inteligente e útil. Responda de forma clara, precisa e educativa.",
            'code_generation': "Você é um especialista em programação. Gere código limpo, eficiente, bem documentado e siga as melhores práticas.",
            'automation': "Você é um especialista em automação de sistemas. Crie scripts e workflows eficientes, seguros e bem estruturados."
        }
        
        # Determinar especialidade
        speciality = speciality_hint or self.config.get('speciality', 'general_language')
        system_prompt = system_prompts.get(speciality, system_prompts['general_language'])
        
        # Construir prompt completo
        full_prompt = f"{system_prompt}\n\n"
        
        if context:
            full_prompt += f"Contexto: {context}\n\n"
        
        full_prompt += f"Usuário: {prompt}\nAssistente:"
        
        return full_prompt
    
    def get_stats(self) -> Dict:
        """Obter estatísticas do modelo"""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
        tokens_per_second = self.total_tokens / sum(self.response_times) if self.response_times else 0.0
        
        return {
            'name': self.name,
            'loaded': self.loaded,
            'device': self.device,
            'load_time': self.load_time,
            'memory_usage_gb': self.memory_usage,
            'gpu_memory_usage_gb': self.gpu_memory_usage,
            'generation_count': self.generation_count,
            'total_tokens': self.total_tokens,
            'avg_response_time': avg_response_time,
            'tokens_per_second': tokens_per_second,
            'last_used': max(self.response_times) if self.response_times else None
        }

class ModelManager:
    """Gestor principal de modelos IA"""
    
    def __init__(self, models_config: Dict, base_path: str = "/opt/alhica-ai/models"):
        self.base_path = Path(base_path)
        self.models_config = models_config
        self.models: Dict[str, ModelInstance] = {}
        self.memory_manager = ModelMemoryManager()
        self.request_queue = Queue()
        self.worker_pool = ThreadPoolExecutor(max_workers=3)
        self.running = False
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'uptime': time.time()
        }
        
        # Configurar signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info("🚀 ModelManager inicializado")
    
    def _signal_handler(self, signum, frame):
        """Handler para sinais de sistema"""
        logger.info(f"Recebido sinal {signum}, iniciando shutdown...")
        self.shutdown()
    
    def initialize_models(self, models_to_load: List[str] = None) -> Dict[str, bool]:
        """Inicializar modelos especificados"""
        if models_to_load is None:
            models_to_load = list(self.models_config.keys())
        
        results = {}
        
        for model_key in models_to_load:
            if model_key not in self.models_config:
                logger.error(f"❌ Configuração não encontrada para modelo: {model_key}")
                results[model_key] = False
                continue
            
            config = self.models_config[model_key]
            model_path = str(self.base_path / model_key)
            
            # Verificar se modelo existe
            if not Path(model_path).exists():
                logger.error(f"❌ Modelo não encontrado: {model_path}")
                results[model_key] = False
                continue
            
            # Verificar memória disponível
            estimated_size = config.get('size_gb', 10.0)
            can_load, message = self.memory_manager.can_load_model(estimated_size)
            
            if not can_load:
                logger.error(f"❌ {message}")
                results[model_key] = False
                continue
            
            # Criar instância do modelo
            model_instance = ModelInstance(config['name'], config, model_path)
            
            # Carregar modelo
            if model_instance.load():
                self.models[model_key] = model_instance
                self.memory_manager.register_model_load(
                    config['name'],
                    model_instance.memory_usage,
                    model_instance.gpu_memory_usage
                )
                results[model_key] = True
                logger.info(f"✅ Modelo {config['name']} inicializado com sucesso")
            else:
                results[model_key] = False
                logger.error(f"❌ Falha ao inicializar modelo {config['name']}")
        
        return results
    
    def generate_text(self, model_key: str, request: GenerationRequest) -> GenerationResponse:
        """Gerar texto usando modelo específico"""
        if model_key not in self.models:
            raise Exception(f"Modelo {model_key} não está carregado")
        
        start_time = time.time()
        
        try:
            self.stats['total_requests'] += 1
            
            response = self.models[model_key].generate(request)
            
            self.stats['successful_requests'] += 1
            
            # Atualizar estatísticas
            response_time = time.time() - start_time
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['successful_requests'] - 1) + response_time) /
                self.stats['successful_requests']
            )
            
            return response
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"❌ Erro na geração de texto: {e}")
            raise
    
    def auto_select_model(self, request: GenerationRequest) -> str:
        """Selecionar automaticamente o melhor modelo para a solicitação"""
        # Análise simples baseada em palavras-chave
        prompt_lower = request.prompt.lower()
        
        # Palavras-chave para código
        code_keywords = ['código', 'script', 'função', 'class', 'def', 'import', 'install', 'configure']
        if any(keyword in prompt_lower for keyword in code_keywords):
            if 'deepseek' in self.models:
                return 'deepseek'
        
        # Palavras-chave para automação
        automation_keywords = ['automatizar', 'workflow', 'pipeline', 'deploy', 'backup', 'monitor']
        if any(keyword in prompt_lower for keyword in automation_keywords):
            if 'wizardcoder' in self.models:
                return 'wizardcoder'
        
        # Padrão: usar Qwen para tarefas gerais
        if 'qwen' in self.models:
            return 'qwen'
        
        # Fallback: primeiro modelo disponível
        if self.models:
            return list(self.models.keys())[0]
        
        raise Exception("Nenhum modelo disponível")
    
    def generate_with_auto_selection(self, request: GenerationRequest) -> GenerationResponse:
        """Gerar texto com seleção automática de modelo"""
        model_key = self.auto_select_model(request)
        logger.info(f"🤖 Modelo selecionado automaticamente: {self.models[model_key].name}")
        return self.generate_text(model_key, request)
    
    def unload_model(self, model_key: str) -> bool:
        """Descarregar modelo específico"""
        if model_key not in self.models:
            return False
        
        model = self.models[model_key]
        self.memory_manager.register_model_unload(
            model.name,
            model.memory_usage,
            model.gpu_memory_usage
        )
        
        model.unload()
        del self.models[model_key]
        
        logger.info(f"✅ Modelo {model.name} descarregado")
        return True
    
    def reload_model(self, model_key: str) -> bool:
        """Recarregar modelo específico"""
        if model_key in self.models:
            self.unload_model(model_key)
        
        return model_key in self.initialize_models([model_key]) and self.initialize_models([model_key])[model_key]
    
    def get_system_stats(self) -> Dict:
        """Obter estatísticas completas do sistema"""
        memory_stats = self.memory_manager.get_memory_stats()
        
        model_stats = {}
        for key, model in self.models.items():
            model_stats[key] = model.get_stats()
        
        return {
            'system': {
                'uptime': time.time() - self.stats['uptime'],
                'total_requests': self.stats['total_requests'],
                'successful_requests': self.stats['successful_requests'],
                'failed_requests': self.stats['failed_requests'],
                'success_rate': (self.stats['successful_requests'] / max(1, self.stats['total_requests'])) * 100,
                'avg_response_time': self.stats['avg_response_time']
            },
            'memory': memory_stats,
            'models': model_stats,
            'loaded_models': list(self.models.keys()),
            'available_models': list(self.models_config.keys())
        }
    
    def health_check(self) -> Dict:
        """Verificação de saúde do sistema"""
        health = {
            'status': 'healthy',
            'models': {},
            'memory': self.memory_manager.get_memory_stats(),
            'issues': []
        }
        
        # Verificar cada modelo
        for key, model in self.models.items():
            model_health = {
                'loaded': model.loaded,
                'responsive': False,
                'last_error': None
            }
            
            # Teste básico de responsividade
            try:
                test_request = GenerationRequest(
                    prompt="Teste",
                    max_tokens=10,
                    temperature=0.1
                )
                response = model.generate(test_request)
                model_health['responsive'] = True
            except Exception as e:
                model_health['last_error'] = str(e)
                health['issues'].append(f"Modelo {model.name} não responsivo: {e}")
            
            health['models'][key] = model_health
        
        # Verificar memória
        memory_stats = health['memory']
        if memory_stats['ram_percent'] > 90:
            health['issues'].append(f"Uso alto de RAM: {memory_stats['ram_percent']:.1f}%")
            health['status'] = 'warning'
        
        if memory_stats['gpu_percent'] > 90:
            health['issues'].append(f"Uso alto de GPU: {memory_stats['gpu_percent']:.1f}%")
            health['status'] = 'warning'
        
        # Status geral
        if health['issues']:
            if any('não responsivo' in issue for issue in health['issues']):
                health['status'] = 'critical'
            elif health['status'] != 'warning':
                health['status'] = 'warning'
        
        return health
    
    def shutdown(self):
        """Shutdown graceful do sistema"""
        logger.info("🛑 Iniciando shutdown do ModelManager...")
        
        self.running = False
        
        # Descarregar todos os modelos
        for model_key in list(self.models.keys()):
            self.unload_model(model_key)
        
        # Shutdown do worker pool
        self.worker_pool.shutdown(wait=True)
        
        logger.info("✅ ModelManager shutdown concluído")

def main():
    """Função principal para teste standalone"""
    # Configuração de exemplo
    models_config = {
        'qwen': {
            'name': 'Qwen 3 25B',
            'speciality': 'general_language',
            'size_gb': 45.0
        },
        'deepseek': {
            'name': 'DeepSeek-Coder',
            'speciality': 'code_generation',
            'size_gb': 62.0
        },
        'wizardcoder': {
            'name': 'WizardCoder',
            'speciality': 'automation',
            'size_gb': 65.0
        }
    }
    
    # Inicializar manager
    manager = ModelManager(models_config)
    
    try:
        # Inicializar modelos
        results = manager.initialize_models(['qwen'])  # Carregar apenas Qwen para teste
        
        if any(results.values()):
            logger.info("✅ Pelo menos um modelo carregado com sucesso")
            
            # Teste de geração
            request = GenerationRequest(
                prompt="Explique o que é inteligência artificial",
                max_tokens=100
            )
            
            response = manager.generate_with_auto_selection(request)
            logger.info(f"Resposta: {response.text}")
            
            # Mostrar estatísticas
            stats = manager.get_system_stats()
            logger.info(f"Estatísticas: {json.dumps(stats, indent=2)}")
        else:
            logger.error("❌ Nenhum modelo foi carregado com sucesso")
    
    except KeyboardInterrupt:
        logger.info("Interrompido pelo utilizador")
    finally:
        manager.shutdown()

if __name__ == "__main__":
    main()

