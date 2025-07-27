#!/usr/bin/env python3
"""
Alhica AI - Core Platform
A Primeira Plataforma do Mundo com IA Conversacional + SSH Automático

Copyright (c) 2024 Alhica AI Team
"""

import os
import sys
import json
import asyncio
import logging
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import queue
import time

# Dependências de segurança
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Dependências SSH
import paramiko
from paramiko import SSHClient, AutoAddPolicy
import socket

# Dependências web
from flask import Flask, request, jsonify, render_template_string, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room

# Dependências de IA
import openai
import requests

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/alhica-ai/core.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('alhica_ai_core')

@dataclass
class ServerConfig:
    """Configuração de servidor"""
    hostname: str
    port: int = 22
    username: str = "root"
    password: Optional[str] = None
    private_key_path: Optional[str] = None
    description: str = ""
    tags: List[str] = None
    last_seen: Optional[datetime] = None
    status: str = "unknown"  # online, offline, error, unknown
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class CommandExecution:
    """Resultado de execução de comando"""
    command: str
    server: str
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    timestamp: datetime
    user_id: str
    session_id: str

@dataclass
class AIResponse:
    """Resposta da IA"""
    model_used: str
    response: str
    confidence: float
    execution_plan: List[Dict]
    risk_level: str  # low, medium, high, critical
    requires_confirmation: bool
    estimated_time: int  # segundos

class SecurityManager:
    """Gestor de segurança e encriptação"""
    
    def __init__(self, master_key_path: str = "/etc/alhica/master.key"):
        self.master_key_path = master_key_path
        self._ensure_master_key()
        self.cipher = self._load_cipher()
    
    def _ensure_master_key(self):
        """Garantir que existe chave mestra"""
        os.makedirs(os.path.dirname(self.master_key_path), exist_ok=True)
        
        if not os.path.exists(self.master_key_path):
            key = Fernet.generate_key()
            with open(self.master_key_path, 'wb') as f:
                f.write(key)
            os.chmod(self.master_key_path, 0o600)
            logger.info("Master key generated")
    
    def _load_cipher(self) -> Fernet:
        """Carregar cipher de encriptação"""
        with open(self.master_key_path, 'rb') as f:
            key = f.read()
        return Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encriptar dados"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Desencriptar dados"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
        """Hash de password com salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        pwdhash = hashlib.pbkdf2_hmac('sha256', 
                                     password.encode('utf-8'), 
                                     salt.encode('utf-8'), 
                                     100000)
        return pwdhash.hex(), salt
    
    def verify_password(self, password: str, hash_value: str, salt: str) -> bool:
        """Verificar password"""
        pwdhash, _ = self.hash_password(password, salt)
        return pwdhash == hash_value

class DatabaseManager:
    """Gestor de base de dados"""
    
    def __init__(self, db_path: str = "/var/lib/alhica-ai/alhica.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Inicializar base de dados"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    email TEXT,
                    role TEXT DEFAULT 'user',
                    mfa_secret TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    active BOOLEAN DEFAULT 1
                );
                
                CREATE TABLE IF NOT EXISTS servers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hostname TEXT NOT NULL,
                    port INTEGER DEFAULT 22,
                    username TEXT NOT NULL,
                    password_encrypted TEXT,
                    private_key_encrypted TEXT,
                    description TEXT,
                    tags TEXT,
                    status TEXT DEFAULT 'unknown',
                    last_seen TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by INTEGER,
                    FOREIGN KEY (created_by) REFERENCES users (id)
                );
                
                CREATE TABLE IF NOT EXISTS command_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    server_id INTEGER NOT NULL,
                    command TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    stdout TEXT,
                    stderr TEXT,
                    exit_code INTEGER,
                    execution_time REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (server_id) REFERENCES servers (id)
                );
                
                CREATE TABLE IF NOT EXISTS ai_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    confidence REAL,
                    risk_level TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                );
                
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    resource_type TEXT,
                    resource_id TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_command_history_timestamp ON command_history(timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_ai_conversations_session ON ai_conversations(session_id);
            """)
        
        # Criar utilizador admin padrão
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Criar utilizador admin padrão"""
        security = SecurityManager()
        password_hash, salt = security.hash_password("admin123")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO users (username, password_hash, salt, email, role)
                VALUES (?, ?, ?, ?, ?)
            """, ("admin", password_hash, salt, "admin@alhica.ai", "admin"))
    
    def get_connection(self):
        """Obter conexão com base de dados"""
        return sqlite3.connect(self.db_path)
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Executar query e retornar resultados"""
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Executar update/insert e retornar ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.lastrowid

class SSHManager:
    """Gestor de conexões SSH"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        self.connections = {}  # Cache de conexões ativas
        self.connection_lock = threading.Lock()
    
    def _get_connection_key(self, hostname: str, port: int, username: str) -> str:
        """Gerar chave única para conexão"""
        return f"{username}@{hostname}:{port}"
    
    def _create_ssh_client(self, server_config: ServerConfig) -> SSHClient:
        """Criar cliente SSH"""
        client = SSHClient()
        client.set_missing_host_key_policy(AutoAddPolicy())
        
        try:
            if server_config.private_key_path:
                # Usar chave privada
                client.connect(
                    hostname=server_config.hostname,
                    port=server_config.port,
                    username=server_config.username,
                    key_filename=server_config.private_key_path,
                    timeout=30
                )
            elif server_config.password:
                # Usar password
                client.connect(
                    hostname=server_config.hostname,
                    port=server_config.port,
                    username=server_config.username,
                    password=server_config.password,
                    timeout=30
                )
            else:
                raise ValueError("Nem password nem chave privada fornecidos")
            
            return client
            
        except Exception as e:
            client.close()
            raise e
    
    def test_connection(self, server_config: ServerConfig) -> bool:
        """Testar conexão SSH"""
        try:
            client = self._create_ssh_client(server_config)
            client.close()
            return True
        except Exception as e:
            logger.error(f"Falha ao testar conexão {server_config.hostname}: {e}")
            return False
    
    def execute_command(self, server_config: ServerConfig, command: str, 
                       timeout: int = 300) -> CommandExecution:
        """Executar comando via SSH"""
        start_time = time.time()
        
        try:
            client = self._create_ssh_client(server_config)
            
            # Executar comando
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            
            # Ler resultados
            stdout_data = stdout.read().decode('utf-8', errors='ignore')
            stderr_data = stderr.read().decode('utf-8', errors='ignore')
            exit_code = stdout.channel.recv_exit_status()
            
            client.close()
            
            execution_time = time.time() - start_time
            
            return CommandExecution(
                command=command,
                server=server_config.hostname,
                success=(exit_code == 0),
                stdout=stdout_data,
                stderr=stderr_data,
                exit_code=exit_code,
                execution_time=execution_time,
                timestamp=datetime.now(),
                user_id="",  # Será preenchido pelo caller
                session_id=""  # Será preenchido pelo caller
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Erro ao executar comando '{command}' em {server_config.hostname}: {e}")
            
            return CommandExecution(
                command=command,
                server=server_config.hostname,
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time=execution_time,
                timestamp=datetime.now(),
                user_id="",
                session_id=""
            )

class AIModelManager:
    """Gestor dos modelos de IA"""
    
    def __init__(self):
        self.models = {
            'qwen': {
                'name': 'Qwen 3 25B',
                'endpoint': 'http://localhost:5001/v1/chat/completions',
                'speciality': 'general',
                'description': 'Modelo generalista para compreensão de linguagem natural'
            },
            'deepseek': {
                'name': 'DeepSeek-Coder',
                'endpoint': 'http://localhost:5002/v1/chat/completions',
                'speciality': 'coding',
                'description': 'Especialista em geração de código e scripts'
            },
            'wizardcoder': {
                'name': 'WizardCoder',
                'endpoint': 'http://localhost:5003/v1/chat/completions',
                'speciality': 'automation',
                'description': 'Mestre em automação e workflows complexos'
            }
        }
        self.model_health = {}
        self._check_models_health()
    
    def _check_models_health(self):
        """Verificar saúde dos modelos"""
        for model_id, config in self.models.items():
            try:
                response = requests.get(f"{config['endpoint']}/health", timeout=5)
                self.model_health[model_id] = response.status_code == 200
            except:
                self.model_health[model_id] = False
    
    def select_best_model(self, user_input: str, context: Dict = None) -> str:
        """Selecionar melhor modelo baseado na entrada"""
        # Palavras-chave para cada modelo
        coding_keywords = ['instalar', 'configurar', 'script', 'código', 'programar', 'deploy']
        automation_keywords = ['automatizar', 'workflow', 'pipeline', 'orquestrar', 'batch']
        
        user_lower = user_input.lower()
        
        # Verificar se é tarefa de automação
        if any(keyword in user_lower for keyword in automation_keywords):
            if self.model_health.get('wizardcoder', False):
                return 'wizardcoder'
        
        # Verificar se é tarefa de código
        if any(keyword in user_lower for keyword in coding_keywords):
            if self.model_health.get('deepseek', False):
                return 'deepseek'
        
        # Padrão: usar Qwen para compreensão geral
        if self.model_health.get('qwen', False):
            return 'qwen'
        
        # Fallback: primeiro modelo disponível
        for model_id, healthy in self.model_health.items():
            if healthy:
                return model_id
        
        raise Exception("Nenhum modelo de IA disponível")
    
    def query_model(self, model_id: str, prompt: str, context: Dict = None) -> AIResponse:
        """Consultar modelo de IA"""
        if not self.model_health.get(model_id, False):
            raise Exception(f"Modelo {model_id} não está disponível")
        
        model_config = self.models[model_id]
        
        # Preparar prompt com contexto
        system_prompt = self._build_system_prompt(model_id, context)
        
        try:
            response = requests.post(
                model_config['endpoint'],
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2048
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                ai_response = data['choices'][0]['message']['content']
                
                # Analisar resposta para extrair plano de execução
                execution_plan = self._parse_execution_plan(ai_response)
                risk_level = self._assess_risk_level(execution_plan)
                
                return AIResponse(
                    model_used=model_id,
                    response=ai_response,
                    confidence=0.85,  # Placeholder
                    execution_plan=execution_plan,
                    risk_level=risk_level,
                    requires_confirmation=(risk_level in ['high', 'critical']),
                    estimated_time=self._estimate_execution_time(execution_plan)
                )
            else:
                raise Exception(f"Erro na API do modelo: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Erro ao consultar modelo {model_id}: {e}")
            raise e
    
    def _build_system_prompt(self, model_id: str, context: Dict = None) -> str:
        """Construir prompt do sistema"""
        base_prompt = """Você é Alhica AI, a primeira plataforma do mundo com IA conversacional + execução SSH automática.

Suas capacidades:
- Compreender linguagem natural
- Gerar comandos SSH seguros
- Executar operações em servidores remotos
- Monitorizar e fazer rollback automático
- Aprender com cada interação

Diretrizes:
1. Sempre priorize segurança
2. Explique o que vai fazer antes de executar
3. Use comandos seguros e testados
4. Implemente verificações de erro
5. Documente todas as ações

Responda de forma clara e estruturada."""

        if model_id == 'deepseek':
            base_prompt += "\n\nEspecialização: Geração de código e scripts. Foque em soluções técnicas precisas."
        elif model_id == 'wizardcoder':
            base_prompt += "\n\nEspecialização: Automação e workflows. Foque em orquestração e otimização."
        
        if context:
            base_prompt += f"\n\nContexto atual: {json.dumps(context, indent=2)}"
        
        return base_prompt
    
    def _parse_execution_plan(self, ai_response: str) -> List[Dict]:
        """Extrair plano de execução da resposta da IA"""
        # Implementação simplificada - pode ser melhorada com NLP
        plan = []
        
        # Procurar por comandos na resposta
        lines = ai_response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('```bash') or line.startswith('```sh'):
                continue
            if line.startswith('```'):
                continue
            if line.startswith('sudo ') or line.startswith('apt ') or line.startswith('systemctl '):
                plan.append({
                    'type': 'command',
                    'command': line,
                    'description': f"Executar: {line}"
                })
        
        return plan
    
    def _assess_risk_level(self, execution_plan: List[Dict]) -> str:
        """Avaliar nível de risco do plano"""
        high_risk_commands = ['rm -rf', 'dd if=', 'mkfs', 'fdisk', 'shutdown', 'reboot']
        medium_risk_commands = ['sudo', 'chmod 777', 'chown', 'systemctl stop']
        
        for step in execution_plan:
            if step.get('type') == 'command':
                command = step.get('command', '').lower()
                
                if any(risky in command for risky in high_risk_commands):
                    return 'critical'
                if any(risky in command for risky in medium_risk_commands):
                    return 'high'
        
        return 'low'
    
    def _estimate_execution_time(self, execution_plan: List[Dict]) -> int:
        """Estimar tempo de execução em segundos"""
        base_time = 5  # 5 segundos base
        return base_time + (len(execution_plan) * 10)

class AlhicaAICore:
    """Classe principal da plataforma Alhica AI"""
    
    def __init__(self):
        self.security = SecurityManager()
        self.database = DatabaseManager()
        self.ssh_manager = SSHManager(self.security)
        self.ai_manager = AIModelManager()
        self.active_sessions = {}
        
        logger.info("Alhica AI Core inicializado")
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Autenticar utilizador"""
        users = self.database.execute_query(
            "SELECT * FROM users WHERE username = ? AND active = 1",
            (username,)
        )
        
        if not users:
            return None
        
        user = users[0]
        if self.security.verify_password(password, user['password_hash'], user['salt']):
            # Atualizar último login
            self.database.execute_update(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                (user['id'],)
            )
            
            return {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'role': user['role']
            }
        
        return None
    
    def add_server(self, user_id: int, server_config: ServerConfig) -> int:
        """Adicionar servidor"""
        # Encriptar credenciais
        password_encrypted = None
        if server_config.password:
            password_encrypted = self.security.encrypt(server_config.password)
        
        # Testar conexão
        if not self.ssh_manager.test_connection(server_config):
            raise Exception("Falha ao conectar ao servidor")
        
        # Salvar na base de dados
        server_id = self.database.execute_update("""
            INSERT INTO servers (hostname, port, username, password_encrypted, 
                               description, tags, status, last_seen, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            server_config.hostname,
            server_config.port,
            server_config.username,
            password_encrypted,
            server_config.description,
            json.dumps(server_config.tags),
            'online',
            datetime.now(),
            user_id
        ))
        
        # Log de auditoria
        self._log_audit(user_id, 'server_added', 'server', str(server_id), 
                       f"Servidor {server_config.hostname} adicionado")
        
        return server_id
    
    def get_servers(self, user_id: int) -> List[Dict]:
        """Obter servidores do utilizador"""
        return self.database.execute_query("""
            SELECT id, hostname, port, username, description, tags, status, last_seen
            FROM servers 
            WHERE created_by = ?
            ORDER BY hostname
        """, (user_id,))
    
    def process_chat_message(self, user_id: int, session_id: str, 
                           message: str, context: Dict = None) -> Dict:
        """Processar mensagem do chat"""
        try:
            # Selecionar melhor modelo
            model_id = self.ai_manager.select_best_model(message, context)
            
            # Consultar IA
            ai_response = self.ai_manager.query_model(model_id, message, context)
            
            # Salvar conversa
            self.database.execute_update("""
                INSERT INTO ai_conversations (user_id, session_id, user_message, 
                                            ai_response, model_used, confidence, risk_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, session_id, message, ai_response.response,
                ai_response.model_used, ai_response.confidence, ai_response.risk_level
            ))
            
            return {
                'success': True,
                'response': ai_response.response,
                'model_used': ai_response.model_used,
                'execution_plan': ai_response.execution_plan,
                'risk_level': ai_response.risk_level,
                'requires_confirmation': ai_response.requires_confirmation,
                'estimated_time': ai_response.estimated_time
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar mensagem: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': "Desculpe, ocorreu um erro ao processar a sua mensagem."
            }
    
    def execute_plan(self, user_id: int, session_id: str, server_id: int, 
                    execution_plan: List[Dict]) -> List[CommandExecution]:
        """Executar plano de comandos"""
        # Obter configuração do servidor
        servers = self.database.execute_query(
            "SELECT * FROM servers WHERE id = ? AND created_by = ?",
            (server_id, user_id)
        )
        
        if not servers:
            raise Exception("Servidor não encontrado")
        
        server_data = servers[0]
        
        # Desencriptar password se existir
        password = None
        if server_data['password_encrypted']:
            password = self.security.decrypt(server_data['password_encrypted'])
        
        server_config = ServerConfig(
            hostname=server_data['hostname'],
            port=server_data['port'],
            username=server_data['username'],
            password=password
        )
        
        results = []
        
        for step in execution_plan:
            if step.get('type') == 'command':
                command = step.get('command')
                
                # Executar comando
                result = self.ssh_manager.execute_command(server_config, command)
                result.user_id = str(user_id)
                result.session_id = session_id
                
                # Salvar histórico
                self.database.execute_update("""
                    INSERT INTO command_history (user_id, server_id, command, success,
                                               stdout, stderr, exit_code, execution_time, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, server_id, result.command, result.success,
                    result.stdout, result.stderr, result.exit_code,
                    result.execution_time, session_id
                ))
                
                results.append(result)
                
                # Se comando falhou, parar execução
                if not result.success:
                    logger.warning(f"Comando falhou, parando execução: {command}")
                    break
        
        return results
    
    def _log_audit(self, user_id: int, action: str, resource_type: str = None,
                  resource_id: str = None, details: str = None,
                  ip_address: str = None, user_agent: str = None):
        """Registar log de auditoria"""
        self.database.execute_update("""
            INSERT INTO audit_log (user_id, action, resource_type, resource_id,
                                 details, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, action, resource_type, resource_id, details, ip_address, user_agent))
    
    def get_system_stats(self) -> Dict:
        """Obter estatísticas do sistema"""
        stats = {}
        
        # Contadores básicos
        stats['total_users'] = self.database.execute_query(
            "SELECT COUNT(*) as count FROM users WHERE active = 1"
        )[0]['count']
        
        stats['total_servers'] = self.database.execute_query(
            "SELECT COUNT(*) as count FROM servers"
        )[0]['count']
        
        stats['commands_today'] = self.database.execute_query(
            "SELECT COUNT(*) as count FROM command_history WHERE DATE(timestamp) = DATE('now')"
        )[0]['count']
        
        stats['success_rate'] = self.database.execute_query("""
            SELECT 
                ROUND(
                    (SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 
                    2
                ) as rate
            FROM command_history 
            WHERE timestamp >= datetime('now', '-7 days')
        """)[0]['rate'] or 0
        
        # Estado dos modelos IA
        stats['ai_models'] = self.ai_manager.model_health
        
        return stats

if __name__ == "__main__":
    # Teste básico
    core = AlhicaAICore()
    print("Alhica AI Core inicializado com sucesso!")
    
    # Testar autenticação
    user = core.authenticate_user("admin", "admin123")
    if user:
        print(f"Utilizador autenticado: {user['username']}")
        
        # Obter estatísticas
        stats = core.get_system_stats()
        print(f"Estatísticas do sistema: {stats}")
    else:
        print("Falha na autenticação")

