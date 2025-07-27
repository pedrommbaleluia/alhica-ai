#!/usr/bin/env python3
"""
Alhica AI - Sistema Integrado Completo
Integração de todos os componentes para criar a plataforma 100% funcional

Copyright (c) 2024 Alhica AI Team
"""

import os
import sys
import json
import logging
import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

# Importar todos os módulos desenvolvidos
from alhica_ai_core import AlhicaAICore
from alhica_ai_web import AlhicaAIWebInterface
from alhica_ai_security import AlhicaAISecurityManager
from alhica_ai_models import AlhicaAIModelManager
from natural_language_parser import NaturalLanguageParser
from intent_classification_system import IntentClassificationSystem
from conversational_context_manager import ConversationalContextManager
from analytics_dashboard_system import AnalyticsDashboardSystem
from model_manager import ModelManager
from performance_optimizer import PerformanceOptimizer

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/alhica-ai/integrated_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemStatus:
    """Status do sistema integrado"""
    core_status: str
    web_status: str
    security_status: str
    models_status: str
    nlp_status: str
    intent_status: str
    context_status: str
    analytics_status: str
    overall_status: str
    startup_time: datetime
    last_health_check: datetime

class AlhicaAIIntegratedSystem:
    """Sistema integrado completo da Alhica AI"""
    
    def __init__(self, config_path: str = "/opt/alhica-ai/config/system.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.status = None
        self.startup_time = datetime.now()
        
        # Componentes do sistema
        self.core = None
        self.web_interface = None
        self.security_manager = None
        self.model_manager = None
        self.nlp_parser = None
        self.intent_classifier = None
        self.context_manager = None
        self.analytics_system = None
        self.performance_optimizer = None
        
        # Threads de serviços
        self.service_threads = {}
        self.running = False
        
        logger.info("🚀 Inicializando Sistema Integrado Alhica AI")
    
    def _load_config(self) -> Dict[str, Any]:
        """Carregar configuração do sistema"""
        default_config = {
            "system": {
                "name": "Alhica AI",
                "version": "1.0.0",
                "environment": "production",
                "debug": False
            },
            "database": {
                "path": "/opt/alhica-ai/data/",
                "backup_interval": 3600
            },
            "security": {
                "encryption_key_path": "/opt/alhica-ai/keys/master.key",
                "session_timeout": 7200,
                "max_login_attempts": 3
            },
            "models": {
                "cache_size": "8GB",
                "auto_download": True,
                "optimization_level": "balanced"
            },
            "web": {
                "host": "0.0.0.0",
                "port": 80,
                "ssl_enabled": False,
                "ssl_cert": "",
                "ssl_key": ""
            },
            "analytics": {
                "dashboard_port": 8080,
                "metrics_retention_days": 30,
                "real_time_updates": True
            },
            "performance": {
                "auto_optimization": True,
                "resource_monitoring": True,
                "alert_thresholds": {
                    "cpu": 85,
                    "memory": 90,
                    "disk": 95
                }
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge com configuração padrão
                    return self._merge_configs(default_config, user_config)
            else:
                # Criar arquivo de configuração padrão
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            return default_config
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Merge configurações recursivamente"""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    async def initialize_system(self):
        """Inicializar todos os componentes do sistema"""
        logger.info("🔧 Inicializando componentes do sistema...")
        
        try:
            # 1. Inicializar Core
            logger.info("Inicializando Core...")
            self.core = AlhicaAICore(self.config)
            await self.core.initialize()
            
            # 2. Inicializar Security Manager
            logger.info("Inicializando Security Manager...")
            self.security_manager = AlhicaAISecurityManager(
                encryption_key_path=self.config["security"]["encryption_key_path"]
            )
            await self.security_manager.initialize()
            
            # 3. Inicializar Model Manager
            logger.info("Inicializando Model Manager...")
            self.model_manager = ModelManager()
            await self.model_manager.initialize()
            
            # 4. Inicializar Performance Optimizer
            logger.info("Inicializando Performance Optimizer...")
            self.performance_optimizer = PerformanceOptimizer()
            await self.performance_optimizer.initialize()
            
            # 5. Inicializar NLP Parser
            logger.info("Inicializando NLP Parser...")
            self.nlp_parser = NaturalLanguageParser()
            await self.nlp_parser.initialize()
            
            # 6. Inicializar Intent Classifier
            logger.info("Inicializando Intent Classifier...")
            self.intent_classifier = IntentClassificationSystem()
            await self.intent_classifier.initialize()
            
            # 7. Inicializar Context Manager
            logger.info("Inicializando Context Manager...")
            self.context_manager = ConversationalContextManager()
            
            # 8. Inicializar Analytics System
            logger.info("Inicializando Analytics System...")
            self.analytics_system = AnalyticsDashboardSystem()
            
            # 9. Inicializar Web Interface (por último)
            logger.info("Inicializando Web Interface...")
            self.web_interface = AlhicaAIWebInterface(
                core=self.core,
                security_manager=self.security_manager,
                model_manager=self.model_manager,
                nlp_parser=self.nlp_parser,
                intent_classifier=self.intent_classifier,
                context_manager=self.context_manager,
                analytics_system=self.analytics_system,
                config=self.config
            )
            await self.web_interface.initialize()
            
            logger.info("✅ Todos os componentes inicializados com sucesso")
            
        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}")
            raise
    
    async def start_services(self):
        """Iniciar todos os serviços"""
        logger.info("🚀 Iniciando serviços...")
        
        try:
            self.running = True
            
            # Iniciar Analytics Dashboard
            self.analytics_system.start_dashboard(
                host="0.0.0.0",
                port=self.config["analytics"]["dashboard_port"]
            )
            
            # Iniciar coleta de métricas
            self.analytics_system.metrics_collector.start_collection()
            
            # Iniciar otimizador de performance
            if self.config["performance"]["auto_optimization"]:
                await self.performance_optimizer.start_optimization()
            
            # Iniciar monitorização de recursos
            if self.config["performance"]["resource_monitoring"]:
                self._start_resource_monitoring()
            
            # Iniciar limpeza automática de contexto
            self._start_context_cleanup()
            
            # Iniciar Web Interface (principal)
            await self.web_interface.start_server(
                host=self.config["web"]["host"],
                port=self.config["web"]["port"]
            )
            
            logger.info("✅ Todos os serviços iniciados")
            
        except Exception as e:
            logger.error(f"❌ Erro ao iniciar serviços: {e}")
            raise
    
    def _start_resource_monitoring(self):
        """Iniciar monitorização de recursos"""
        def monitor_resources():
            import psutil
            
            while self.running:
                try:
                    # CPU
                    cpu_percent = psutil.cpu_percent(interval=1)
                    if cpu_percent > self.config["performance"]["alert_thresholds"]["cpu"]:
                        self.analytics_system.record_system_event(
                            "high_cpu_usage",
                            {"cpu_percent": cpu_percent},
                            "warning",
                            "resource_monitor"
                        )
                    
                    # Memória
                    memory = psutil.virtual_memory()
                    if memory.percent > self.config["performance"]["alert_thresholds"]["memory"]:
                        self.analytics_system.record_system_event(
                            "high_memory_usage",
                            {"memory_percent": memory.percent},
                            "warning",
                            "resource_monitor"
                        )
                    
                    # Disco
                    disk = psutil.disk_usage('/')
                    disk_percent = (disk.used / disk.total) * 100
                    if disk_percent > self.config["performance"]["alert_thresholds"]["disk"]:
                        self.analytics_system.record_system_event(
                            "high_disk_usage",
                            {"disk_percent": disk_percent},
                            "critical",
                            "resource_monitor"
                        )
                    
                    time.sleep(60)  # Verificar a cada minuto
                    
                except Exception as e:
                    logger.error(f"Erro na monitorização de recursos: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        self.service_threads['resource_monitor'] = monitor_thread
    
    def _start_context_cleanup(self):
        """Iniciar limpeza automática de contexto"""
        def cleanup_contexts():
            while self.running:
                try:
                    self.context_manager.cleanup_expired_sessions()
                    time.sleep(1800)  # Limpar a cada 30 minutos
                except Exception as e:
                    logger.error(f"Erro na limpeza de contexto: {e}")
                    time.sleep(1800)
        
        cleanup_thread = threading.Thread(target=cleanup_contexts, daemon=True)
        cleanup_thread.start()
        self.service_threads['context_cleanup'] = cleanup_thread
    
    async def process_user_message(self, user_id: str, message: str, session_id: str = None) -> Dict[str, Any]:
        """Processar mensagem do utilizador (pipeline completo)"""
        try:
            # 1. Criar ou obter sessão
            if not session_id:
                session_id = self.context_manager.create_session(user_id, message)
            else:
                self.context_manager.add_message(session_id, message, "user")
            
            # 2. Obter contexto relevante
            context = self.context_manager.get_relevant_context(session_id, message)
            
            # 3. Parse de linguagem natural
            parsed_message = await self.nlp_parser.parse_message(message, context)
            
            # 4. Classificar intenção
            intent_result = await self.intent_classifier.classify_intent(
                message, context, parsed_message
            )
            
            # 5. Atualizar contexto com intenção
            self.context_manager.update_intent(session_id, intent_result['intent'], intent_result['confidence'])
            
            # 6. Adicionar entidades ao contexto
            for entity_name, entity_value in parsed_message.get('entities', {}).items():
                self.context_manager.add_entity(
                    session_id, entity_name, entity_value, 
                    "extracted", intent_result['confidence']
                )
            
            # 7. Processar com modelo apropriado
            model_response = await self.model_manager.process_request(
                intent_result['intent'],
                message,
                context,
                parsed_message
            )
            
            # 8. Executar ação se necessário
            execution_result = None
            if intent_result['intent'] in ['install_software', 'configure_service', 'system_command']:
                execution_result = await self.core.execute_ssh_command(
                    model_response.get('command', ''),
                    model_response.get('server', 'localhost'),
                    context
                )
            
            # 9. Gerar resposta final
            final_response = await self._generate_final_response(
                model_response, execution_result, context
            )
            
            # 10. Adicionar resposta ao contexto
            self.context_manager.add_message(
                session_id, final_response['message'], "assistant",
                intent=intent_result['intent'],
                metadata=final_response.get('metadata', {})
            )
            
            # 11. Registar métricas
            self.analytics_system.record_metric(
                "user_interaction", 1, "usage", "count",
                {"intent": intent_result['intent'], "success": final_response.get('success', True)}
            )
            
            return {
                'session_id': session_id,
                'message': final_response['message'],
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'execution_result': execution_result,
                'success': final_response.get('success', True),
                'metadata': final_response.get('metadata', {})
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar mensagem: {e}")
            
            # Registar erro
            self.analytics_system.record_system_event(
                "message_processing_error",
                {"error": str(e), "user_id": user_id, "message": message},
                "error",
                "message_processor"
            )
            
            return {
                'session_id': session_id,
                'message': f"Desculpe, ocorreu um erro ao processar a sua mensagem: {str(e)}",
                'intent': 'error',
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            }
    
    async def _generate_final_response(self, model_response: Dict[str, Any], 
                                     execution_result: Dict[str, Any] = None,
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Gerar resposta final baseada no resultado do modelo e execução"""
        try:
            if execution_result:
                if execution_result.get('success', False):
                    message = f"✅ {model_response.get('message', 'Comando executado com sucesso!')}\n\n"
                    if execution_result.get('output'):
                        message += f"Resultado:\n```\n{execution_result['output']}\n```"
                    
                    return {
                        'message': message,
                        'success': True,
                        'metadata': {
                            'execution_time': execution_result.get('execution_time', 0),
                            'command': execution_result.get('command', ''),
                            'server': execution_result.get('server', '')
                        }
                    }
                else:
                    message = f"❌ Erro na execução: {execution_result.get('error', 'Erro desconhecido')}"
                    return {
                        'message': message,
                        'success': False,
                        'metadata': {
                            'error': execution_result.get('error', ''),
                            'command': execution_result.get('command', ''),
                            'server': execution_result.get('server', '')
                        }
                    }
            else:
                # Resposta apenas informativa
                return {
                    'message': model_response.get('message', 'Processado com sucesso.'),
                    'success': True,
                    'metadata': model_response.get('metadata', {})
                }
                
        except Exception as e:
            logger.error(f"Erro ao gerar resposta final: {e}")
            return {
                'message': f"Erro interno: {str(e)}",
                'success': False,
                'metadata': {'error': str(e)}
            }
    
    def get_system_status(self) -> SystemStatus:
        """Obter status do sistema"""
        try:
            status = SystemStatus(
                core_status="running" if self.core and self.core.is_running() else "stopped",
                web_status="running" if self.web_interface and self.web_interface.is_running() else "stopped",
                security_status="active" if self.security_manager else "inactive",
                models_status="loaded" if self.model_manager and self.model_manager.models_loaded() else "not_loaded",
                nlp_status="active" if self.nlp_parser else "inactive",
                intent_status="active" if self.intent_classifier else "inactive",
                context_status="active" if self.context_manager else "inactive",
                analytics_status="running" if self.analytics_system else "stopped",
                overall_status="running" if self.running else "stopped",
                startup_time=self.startup_time,
                last_health_check=datetime.now()
            )
            
            self.status = status
            return status
            
        except Exception as e:
            logger.error(f"Erro ao obter status: {e}")
            return SystemStatus(
                core_status="error",
                web_status="error",
                security_status="error",
                models_status="error",
                nlp_status="error",
                intent_status="error",
                context_status="error",
                analytics_status="error",
                overall_status="error",
                startup_time=self.startup_time,
                last_health_check=datetime.now()
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificação de saúde do sistema"""
        status = self.get_system_status()
        
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': status.overall_status,
            'uptime_seconds': (datetime.now() - status.startup_time).total_seconds(),
            'components': {
                'core': status.core_status,
                'web': status.web_status,
                'security': status.security_status,
                'models': status.models_status,
                'nlp': status.nlp_status,
                'intent': status.intent_status,
                'context': status.context_status,
                'analytics': status.analytics_status
            }
        }
        
        # Adicionar métricas de sistema
        try:
            import psutil
            health_data['system_metrics'] = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'active_connections': len(psutil.net_connections()),
                'process_count': len(psutil.pids())
            }
        except Exception as e:
            logger.warning(f"Erro ao obter métricas do sistema: {e}")
        
        # Registar health check
        self.analytics_system.record_system_event(
            "health_check",
            health_data,
            "info",
            "system_monitor"
        )
        
        return health_data
    
    async def shutdown(self):
        """Encerrar sistema graciosamente"""
        logger.info("🛑 Iniciando encerramento do sistema...")
        
        try:
            self.running = False
            
            # Parar serviços
            if self.analytics_system:
                self.analytics_system.metrics_collector.stop_collection()
            
            if self.performance_optimizer:
                await self.performance_optimizer.stop_optimization()
            
            if self.web_interface:
                await self.web_interface.shutdown()
            
            # Aguardar threads de serviço
            for name, thread in self.service_threads.items():
                logger.info(f"Aguardando thread {name}...")
                thread.join(timeout=5)
            
            # Salvar estado final
            if self.context_manager:
                # Terminar sessões ativas
                for session_id in list(self.context_manager.active_sessions.keys()):
                    self.context_manager.end_session(session_id, "system_shutdown")
            
            # Registar encerramento
            if self.analytics_system:
                self.analytics_system.record_system_event(
                    "system_shutdown",
                    {"uptime_seconds": (datetime.now() - self.startup_time).total_seconds()},
                    "info",
                    "system"
                )
            
            logger.info("✅ Sistema encerrado com sucesso")
            
        except Exception as e:
            logger.error(f"❌ Erro durante encerramento: {e}")
    
    async def run(self):
        """Executar sistema completo"""
        try:
            # Inicializar sistema
            await self.initialize_system()
            
            # Iniciar serviços
            await self.start_services()
            
            # Registar startup
            self.analytics_system.record_system_event(
                "system_startup",
                {"version": self.config["system"]["version"]},
                "info",
                "system"
            )
            
            logger.info("🎉 Alhica AI Sistema Integrado iniciado com sucesso!")
            logger.info(f"🌐 Interface Web: http://{self.config['web']['host']}:{self.config['web']['port']}")
            logger.info(f"📊 Dashboard Analytics: http://localhost:{self.config['analytics']['dashboard_port']}")
            
            # Manter sistema em execução
            while self.running:
                # Health check periódico
                await self.health_check()
                await asyncio.sleep(300)  # A cada 5 minutos
                
        except KeyboardInterrupt:
            logger.info("Interrupção do utilizador recebida")
        except Exception as e:
            logger.error(f"Erro crítico no sistema: {e}")
            raise
        finally:
            await self.shutdown()

def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Alhica AI - Sistema Integrado")
    parser.add_argument("--config", default="/opt/alhica-ai/config/system.json",
                       help="Caminho para arquivo de configuração")
    parser.add_argument("--debug", action="store_true",
                       help="Ativar modo debug")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Criar sistema integrado
    system = AlhicaAIIntegratedSystem(args.config)
    
    # Executar sistema
    try:
        asyncio.run(system.run())
    except Exception as e:
        logger.error(f"Falha crítica: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

