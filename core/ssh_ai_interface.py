#!/usr/bin/env python3
"""
Manus AI SSH Intelligent Interface
==================================

Interface revolucionária que combina IA com execução automática SSH.
Primeira implementação no mundo de IA que interpreta linguagem natural
e executa comandos automaticamente em servidores remotos.

Autor: Manus AI Team
Versão: 1.0.0
Data: 2024

Funcionalidades:
- Interpretação de linguagem natural para comandos SSH
- Execução automática baseada em intenções
- Integração com WizardCoder, DeepSeek e Qwen
- Geração inteligente de scripts
- Validação automática de comandos
- Sugestões contextuais
- Aprendizagem contínua
"""

import re
import json
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import requests
from datetime import datetime
import sqlite3
import os
from ssh_automation_core import SSHAutomationCore, CommandExecution, CommandStatus
from ssh_credential_manager_web import app, require_auth, user_manager
from flask import request, jsonify
import openai

# Configurar logging
logger = logging.getLogger(__name__)

class CommandIntent(Enum):
    """Tipos de intenção de comando"""
    INSTALL_SOFTWARE = "install_software"
    UPDATE_SYSTEM = "update_system"
    CONFIGURE_SERVICE = "configure_service"
    MONITOR_SYSTEM = "monitor_system"
    MANAGE_FILES = "manage_files"
    NETWORK_OPERATIONS = "network_operations"
    SECURITY_OPERATIONS = "security_operations"
    DATABASE_OPERATIONS = "database_operations"
    DEVELOPMENT_SETUP = "development_setup"
    BACKUP_RESTORE = "backup_restore"
    USER_MANAGEMENT = "user_management"
    PROCESS_MANAGEMENT = "process_management"
    CUSTOM_SCRIPT = "custom_script"
    UNKNOWN = "unknown"

@dataclass
class CommandAnalysis:
    """Análise de comando"""
    intent: CommandIntent
    confidence: float
    target_servers: List[str]
    software_packages: List[str]
    parameters: Dict[str, Any]
    risk_level: str  # low, medium, high, critical
    requires_confirmation: bool
    estimated_duration: int  # segundos
    rollback_possible: bool
    rollback_commands: List[str]
    dependencies: List[str]
    description: str

@dataclass
class AIResponse:
    """Resposta da IA"""
    commands: List[str]
    explanation: str
    warnings: List[str]
    suggestions: List[str]
    confidence: float
    model_used: str

class CommandClassifier:
    """Classificador de comandos usando IA"""
    
    def __init__(self, ssh_automation: SSHAutomationCore):
        self.ssh_automation = ssh_automation
        self.model_urls = {
            'qwen': 'http://localhost:5001',
            'deepseek': 'http://localhost:5003',
            'wizardcoder': 'http://localhost:5000'
        }
        
        # Padrões de comando conhecidos
        self.command_patterns = {
            CommandIntent.INSTALL_SOFTWARE: [
                r'install\s+(.+)',
                r'instalar\s+(.+)',
                r'setup\s+(.+)',
                r'configurar\s+(.+)',
                r'apt\s+install\s+(.+)',
                r'yum\s+install\s+(.+)',
                r'pip\s+install\s+(.+)',
                r'npm\s+install\s+(.+)'
            ],
            CommandIntent.UPDATE_SYSTEM: [
                r'update\s+system',
                r'atualizar\s+sistema',
                r'upgrade\s+all',
                r'apt\s+update',
                r'yum\s+update'
            ],
            CommandIntent.MONITOR_SYSTEM: [
                r'check\s+status',
                r'verificar\s+estado',
                r'monitor\s+(.+)',
                r'show\s+(.+)',
                r'list\s+(.+)',
                r'ps\s+aux',
                r'top',
                r'htop',
                r'df\s+-h',
                r'free\s+-h'
            ],
            CommandIntent.MANAGE_FILES: [
                r'copy\s+(.+)',
                r'move\s+(.+)',
                r'delete\s+(.+)',
                r'backup\s+(.+)',
                r'cp\s+(.+)',
                r'mv\s+(.+)',
                r'rm\s+(.+)',
                r'mkdir\s+(.+)',
                r'chmod\s+(.+)',
                r'chown\s+(.+)'
            ],
            CommandIntent.CONFIGURE_SERVICE: [
                r'start\s+(.+)',
                r'stop\s+(.+)',
                r'restart\s+(.+)',
                r'enable\s+(.+)',
                r'disable\s+(.+)',
                r'systemctl\s+(.+)',
                r'service\s+(.+)'
            ]
        }
        
        # Base de conhecimento de software
        self.software_knowledge = {
            'docker': {
                'install_commands': ['curl -fsSL https://get.docker.com | sh'],
                'dependencies': ['curl'],
                'post_install': ['systemctl start docker', 'systemctl enable docker'],
                'risk_level': 'medium',
                'rollback': ['systemctl stop docker', 'apt remove docker.io -y']
            },
            'nginx': {
                'install_commands': ['apt update', 'apt install nginx -y'],
                'dependencies': [],
                'post_install': ['systemctl start nginx', 'systemctl enable nginx'],
                'risk_level': 'low',
                'rollback': ['systemctl stop nginx', 'apt remove nginx -y']
            },
            'mysql': {
                'install_commands': ['apt update', 'apt install mysql-server -y'],
                'dependencies': [],
                'post_install': ['systemctl start mysql', 'systemctl enable mysql', 'mysql_secure_installation'],
                'risk_level': 'high',
                'rollback': ['systemctl stop mysql', 'apt remove mysql-server -y']
            },
            'postgresql': {
                'install_commands': ['apt update', 'apt install postgresql postgresql-contrib -y'],
                'dependencies': [],
                'post_install': ['systemctl start postgresql', 'systemctl enable postgresql'],
                'risk_level': 'medium',
                'rollback': ['systemctl stop postgresql', 'apt remove postgresql -y']
            },
            'nodejs': {
                'install_commands': ['curl -fsSL https://deb.nodesource.com/setup_18.x | bash -', 'apt install nodejs -y'],
                'dependencies': ['curl'],
                'post_install': [],
                'risk_level': 'low',
                'rollback': ['apt remove nodejs -y']
            },
            'python3': {
                'install_commands': ['apt update', 'apt install python3 python3-pip -y'],
                'dependencies': [],
                'post_install': [],
                'risk_level': 'low',
                'rollback': ['apt remove python3 python3-pip -y']
            }
        }
        
    def classify_intent(self, user_input: str) -> CommandIntent:
        """Classificar intenção do comando"""
        user_input_lower = user_input.lower()
        
        for intent, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    return intent
                    
        return CommandIntent.UNKNOWN
        
    def extract_software_packages(self, user_input: str) -> List[str]:
        """Extrair pacotes de software mencionados"""
        packages = []
        user_input_lower = user_input.lower()
        
        for software in self.software_knowledge.keys():
            if software in user_input_lower:
                packages.append(software)
                
        # Padrões adicionais para extração
        install_patterns = [
            r'install\s+([a-zA-Z0-9\-_]+)',
            r'instalar\s+([a-zA-Z0-9\-_]+)',
            r'apt\s+install\s+([a-zA-Z0-9\-_\s]+)',
            r'pip\s+install\s+([a-zA-Z0-9\-_\s]+)'
        ]
        
        for pattern in install_patterns:
            matches = re.findall(pattern, user_input_lower)
            for match in matches:
                if isinstance(match, str):
                    packages.extend(match.split())
                    
        return list(set(packages))
        
    def assess_risk_level(self, intent: CommandIntent, packages: List[str], commands: List[str]) -> str:
        """Avaliar nível de risco do comando"""
        high_risk_patterns = [
            r'rm\s+-rf\s+/',
            r'dd\s+if=',
            r'mkfs\.',
            r'fdisk',
            r'parted',
            r'shutdown',
            r'reboot',
            r'halt',
            r'init\s+0',
            r'init\s+6'
        ]
        
        medium_risk_patterns = [
            r'systemctl\s+stop',
            r'service\s+\w+\s+stop',
            r'rm\s+-rf',
            r'chmod\s+777',
            r'chown\s+root'
        ]
        
        # Verificar padrões de alto risco
        for command in commands:
            for pattern in high_risk_patterns:
                if re.search(pattern, command.lower()):
                    return 'critical'
                    
        # Verificar padrões de médio risco
        for command in commands:
            for pattern in medium_risk_patterns:
                if re.search(pattern, command.lower()):
                    return 'high'
                    
        # Verificar risco baseado em software
        for package in packages:
            if package in self.software_knowledge:
                pkg_risk = self.software_knowledge[package]['risk_level']
                if pkg_risk in ['high', 'critical']:
                    return pkg_risk
                elif pkg_risk == 'medium':
                    return 'medium'
                    
        # Verificar risco baseado em intenção
        if intent in [CommandIntent.SECURITY_OPERATIONS, CommandIntent.USER_MANAGEMENT]:
            return 'high'
        elif intent in [CommandIntent.CONFIGURE_SERVICE, CommandIntent.DATABASE_OPERATIONS]:
            return 'medium'
        else:
            return 'low'
            
    def generate_rollback_commands(self, intent: CommandIntent, packages: List[str], commands: List[str]) -> List[str]:
        """Gerar comandos de rollback"""
        rollback_commands = []
        
        # Rollback baseado em software
        for package in packages:
            if package in self.software_knowledge:
                rollback_commands.extend(self.software_knowledge[package]['rollback'])
                
        # Rollback baseado em comandos específicos
        for command in commands:
            if 'systemctl start' in command:
                service = command.split()[-1]
                rollback_commands.append(f'systemctl stop {service}')
            elif 'systemctl enable' in command:
                service = command.split()[-1]
                rollback_commands.append(f'systemctl disable {service}')
            elif 'apt install' in command:
                packages_to_remove = command.replace('apt install', '').replace('-y', '').strip()
                rollback_commands.append(f'apt remove {packages_to_remove} -y')
                
        return rollback_commands

class AICommandGenerator:
    """Gerador de comandos usando IA"""
    
    def __init__(self, ssh_automation: SSHAutomationCore):
        self.ssh_automation = ssh_automation
        self.model_urls = {
            'qwen': 'http://localhost:5001',
            'deepseek': 'http://localhost:5003',
            'wizardcoder': 'http://localhost:5000'
        }
        
    def select_best_model(self, intent: CommandIntent, complexity: str) -> str:
        """Selecionar melhor modelo para a tarefa"""
        if intent in [CommandIntent.INSTALL_SOFTWARE, CommandIntent.DEVELOPMENT_SETUP, CommandIntent.CUSTOM_SCRIPT]:
            return 'wizardcoder'
        elif intent in [CommandIntent.CONFIGURE_SERVICE, CommandIntent.DATABASE_OPERATIONS, CommandIntent.SECURITY_OPERATIONS]:
            return 'deepseek'
        else:
            return 'qwen'
            
    def generate_commands(self, user_input: str, analysis: CommandAnalysis, target_os: str = 'ubuntu') -> AIResponse:
        """Gerar comandos usando IA"""
        model = self.select_best_model(analysis.intent, 'medium')
        
        # Construir prompt contextual
        prompt = self._build_prompt(user_input, analysis, target_os)
        
        try:
            # Chamar modelo de IA
            response = requests.post(
                f"{self.model_urls[model]}/generate",
                json={
                    'prompt': prompt,
                    'max_tokens': 2048,
                    'temperature': 0.1
                },
                timeout=60
            )
            
            if response.ok:
                ai_response = response.json()
                return self._parse_ai_response(ai_response['response'], model)
            else:
                logger.error(f"Erro na chamada do modelo {model}: {response.status_code}")
                return self._fallback_response(analysis)
                
        except Exception as e:
            logger.error(f"Erro ao gerar comandos com IA: {e}")
            return self._fallback_response(analysis)
            
    def _build_prompt(self, user_input: str, analysis: CommandAnalysis, target_os: str) -> str:
        """Construir prompt para IA"""
        prompt = f"""
Você é um especialista em administração de sistemas Linux com foco em {target_os}.
Um utilizador solicitou: "{user_input}"

Análise da solicitação:
- Intenção: {analysis.intent.value}
- Pacotes identificados: {', '.join(analysis.software_packages)}
- Nível de risco: {analysis.risk_level}
- Sistema operacional: {target_os}

Instruções:
1. Gere uma sequência de comandos bash para executar a solicitação
2. Inclua verificações de erro quando apropriado
3. Use boas práticas de segurança
4. Forneça comandos de rollback se necessário
5. Explique cada passo importante

Formato da resposta:
COMANDOS:
[lista de comandos bash, um por linha]

EXPLICAÇÃO:
[explicação detalhada do que cada comando faz]

AVISOS:
[avisos importantes sobre riscos ou considerações]

SUGESTÕES:
[sugestões de melhorias ou alternativas]

ROLLBACK:
[comandos para desfazer as alterações, se aplicável]
"""
        return prompt
        
    def _parse_ai_response(self, response_text: str, model_used: str) -> AIResponse:
        """Analisar resposta da IA"""
        commands = []
        explanation = ""
        warnings = []
        suggestions = []
        rollback_commands = []
        
        sections = {
            'COMANDOS:': 'commands',
            'EXPLICAÇÃO:': 'explanation',
            'AVISOS:': 'warnings',
            'SUGESTÕES:': 'suggestions',
            'ROLLBACK:': 'rollback'
        }
        
        current_section = None
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Verificar se é início de nova seção
            for section_marker, section_name in sections.items():
                if line.startswith(section_marker):
                    current_section = section_name
                    break
            else:
                # Processar conteúdo da seção atual
                if current_section == 'commands' and line and not line.startswith('#'):
                    # Limpar comando
                    clean_command = line.replace('```bash', '').replace('```', '').strip()
                    if clean_command and not clean_command.startswith('#'):
                        commands.append(clean_command)
                elif current_section == 'explanation' and line:
                    explanation += line + " "
                elif current_section == 'warnings' and line:
                    if line.startswith('-') or line.startswith('•'):
                        warnings.append(line[1:].strip())
                    elif line:
                        warnings.append(line)
                elif current_section == 'suggestions' and line:
                    if line.startswith('-') or line.startswith('•'):
                        suggestions.append(line[1:].strip())
                    elif line:
                        suggestions.append(line)
                elif current_section == 'rollback' and line and not line.startswith('#'):
                    clean_command = line.replace('```bash', '').replace('```', '').strip()
                    if clean_command:
                        rollback_commands.append(clean_command)
                        
        return AIResponse(
            commands=commands,
            explanation=explanation.strip(),
            warnings=warnings,
            suggestions=suggestions,
            confidence=0.8,  # Calcular baseado na qualidade da resposta
            model_used=model_used
        )
        
    def _fallback_response(self, analysis: CommandAnalysis) -> AIResponse:
        """Resposta de fallback quando IA falha"""
        commands = []
        
        # Gerar comandos básicos baseado na análise
        if analysis.intent == CommandIntent.INSTALL_SOFTWARE:
            for package in analysis.software_packages:
                if package in ['docker', 'nginx', 'mysql', 'postgresql']:
                    commands.extend([
                        'apt update',
                        f'apt install {package} -y',
                        f'systemctl start {package}',
                        f'systemctl enable {package}'
                    ])
                    
        return AIResponse(
            commands=commands,
            explanation="Comandos gerados automaticamente (IA indisponível)",
            warnings=["IA indisponível - usando comandos básicos"],
            suggestions=["Verifique se os modelos de IA estão funcionando"],
            confidence=0.3,
            model_used="fallback"
        )

class IntelligentSSHInterface:
    """Interface inteligente principal"""
    
    def __init__(self, ssh_automation: SSHAutomationCore):
        self.ssh_automation = ssh_automation
        self.classifier = CommandClassifier(ssh_automation)
        self.ai_generator = AICommandGenerator(ssh_automation)
        self.conversation_history = {}
        self._setup_database()
        
    def _setup_database(self):
        """Configurar base de dados para histórico de conversas"""
        db_path = "/etc/manus/conversations.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    commands_generated TEXT,
                    executed BOOLEAN DEFAULT FALSE,
                    execution_results TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    commands TEXT NOT NULL,
                    success_rate REAL DEFAULT 0.0,
                    user_feedback TEXT,
                    improvement_suggestions TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
    def process_natural_language_request(self, user_input: str, user_id: int, 
                                       session_id: str, target_servers: List[str] = None) -> Dict[str, Any]:
        """Processar solicitação em linguagem natural"""
        try:
            # 1. Classificar intenção
            intent = self.classifier.classify_intent(user_input)
            
            # 2. Extrair informações
            software_packages = self.classifier.extract_software_packages(user_input)
            
            # 3. Analisar servidores alvo
            if not target_servers:
                target_servers = self._extract_target_servers(user_input)
                
            # 4. Criar análise completa
            analysis = CommandAnalysis(
                intent=intent,
                confidence=0.8,
                target_servers=target_servers,
                software_packages=software_packages,
                parameters={},
                risk_level='low',
                requires_confirmation=False,
                estimated_duration=60,
                rollback_possible=True,
                rollback_commands=[],
                dependencies=[],
                description=user_input
            )
            
            # 5. Gerar comandos com IA
            ai_response = self.ai_generator.generate_commands(user_input, analysis)
            
            # 6. Avaliar risco
            analysis.risk_level = self.classifier.assess_risk_level(
                intent, software_packages, ai_response.commands
            )
            analysis.requires_confirmation = analysis.risk_level in ['high', 'critical']
            analysis.rollback_commands = self.classifier.generate_rollback_commands(
                intent, software_packages, ai_response.commands
            )
            
            # 7. Salvar na base de dados
            self._save_conversation(user_id, session_id, user_input, analysis, ai_response)
            
            return {
                'success': True,
                'intent': intent.value,
                'confidence': ai_response.confidence,
                'commands': ai_response.commands,
                'explanation': ai_response.explanation,
                'warnings': ai_response.warnings,
                'suggestions': ai_response.suggestions,
                'risk_level': analysis.risk_level,
                'requires_confirmation': analysis.requires_confirmation,
                'target_servers': target_servers,
                'rollback_commands': analysis.rollback_commands,
                'estimated_duration': analysis.estimated_duration,
                'model_used': ai_response.model_used
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar solicitação: {e}")
            return {
                'success': False,
                'error': str(e),
                'suggestions': ['Tente reformular a solicitação', 'Verifique se os modelos de IA estão funcionando']
            }
            
    def execute_ai_generated_commands(self, commands: List[str], target_servers: List[str],
                                    user_id: int, session_id: str, 
                                    rollback_commands: List[str] = None) -> Dict[str, Any]:
        """Executar comandos gerados pela IA"""
        try:
            executions = []
            
            for hostname in target_servers:
                for command in commands:
                    # Executar comando com rollback se disponível
                    if rollback_commands:
                        rollback_cmd = rollback_commands[0] if rollback_commands else None
                        execution = self.ssh_automation.execute_with_rollback(
                            hostname, command, rollback_cmd
                        )
                    else:
                        execution = self.ssh_automation.execute_command(hostname, command)
                        
                    executions.append(execution)
                    
                    # Se comando falhou e é crítico, parar execução
                    if execution.status == CommandStatus.FAILED and 'critical' in command.lower():
                        break
                        
            # Analisar resultados
            success_count = sum(1 for e in executions if e.status == CommandStatus.SUCCESS)
            total_count = len(executions)
            success_rate = success_count / total_count if total_count > 0 else 0
            
            # Salvar resultados para aprendizagem
            self._save_execution_results(user_id, session_id, executions, success_rate)
            
            return {
                'success': True,
                'executions': [self._execution_to_dict(e) for e in executions],
                'success_rate': success_rate,
                'total_executions': total_count,
                'successful_executions': success_count
            }
            
        except Exception as e:
            logger.error(f"Erro ao executar comandos: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def _extract_target_servers(self, user_input: str) -> List[str]:
        """Extrair servidores alvo da entrada do utilizador"""
        # Padrões para identificar servidores
        server_patterns = [
            r'on\s+server\s+([a-zA-Z0-9\.\-_]+)',
            r'no\s+servidor\s+([a-zA-Z0-9\.\-_]+)',
            r'em\s+([a-zA-Z0-9\.\-_]+)',
            r'@([a-zA-Z0-9\.\-_]+)',
            r'servidor\s+([a-zA-Z0-9\.\-_]+)'
        ]
        
        servers = []
        for pattern in server_patterns:
            matches = re.findall(pattern, user_input.lower())
            servers.extend(matches)
            
        # Se não encontrou servidores específicos, usar todos os disponíveis
        if not servers:
            all_servers = self.ssh_automation.credential_manager.list_servers()
            if len(all_servers) == 1:
                servers = all_servers
            # Se múltiplos servidores, deixar para o utilizador escolher
            
        return list(set(servers))
        
    def _save_conversation(self, user_id: int, session_id: str, user_input: str,
                          analysis: CommandAnalysis, ai_response: AIResponse):
        """Salvar conversa na base de dados"""
        try:
            db_path = "/etc/manus/conversations.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT INTO conversations 
                    (user_id, session_id, user_input, intent, commands_generated)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    user_id,
                    session_id,
                    user_input,
                    analysis.intent.value,
                    json.dumps(ai_response.commands)
                ))
        except Exception as e:
            logger.error(f"Erro ao salvar conversa: {e}")
            
    def _save_execution_results(self, user_id: int, session_id: str, 
                              executions: List[CommandExecution], success_rate: float):
        """Salvar resultados de execução para aprendizagem"""
        try:
            db_path = "/etc/manus/conversations.db"
            with sqlite3.connect(db_path) as conn:
                # Atualizar conversa com resultados
                execution_data = [self._execution_to_dict(e) for e in executions]
                conn.execute("""
                    UPDATE conversations 
                    SET executed = TRUE, execution_results = ?
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (json.dumps(execution_data), user_id, session_id))
                
                # Salvar para aprendizagem
                conn.execute("""
                    INSERT INTO ai_learning 
                    (user_input, intent, commands, success_rate)
                    SELECT user_input, intent, commands_generated, ?
                    FROM conversations 
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (success_rate, user_id, session_id))
                
        except Exception as e:
            logger.error(f"Erro ao salvar resultados: {e}")
            
    def _execution_to_dict(self, execution: CommandExecution) -> Dict[str, Any]:
        """Converter execução para dicionário"""
        return {
            'id': execution.id,
            'server': execution.server,
            'command': execution.command,
            'status': execution.status.value,
            'start_time': execution.start_time.isoformat(),
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'stdout': execution.stdout,
            'stderr': execution.stderr,
            'exit_code': execution.exit_code,
            'execution_time': execution.execution_time
        }
        
    def get_conversation_history(self, user_id: int, session_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Obter histórico de conversas"""
        try:
            db_path = "/etc/manus/conversations.db"
            with sqlite3.connect(db_path) as conn:
                if session_id:
                    cursor = conn.execute("""
                        SELECT user_input, intent, commands_generated, executed, execution_results, timestamp
                        FROM conversations 
                        WHERE user_id = ? AND session_id = ?
                        ORDER BY timestamp DESC LIMIT ?
                    """, (user_id, session_id, limit))
                else:
                    cursor = conn.execute("""
                        SELECT user_input, intent, commands_generated, executed, execution_results, timestamp
                        FROM conversations 
                        WHERE user_id = ?
                        ORDER BY timestamp DESC LIMIT ?
                    """, (user_id, limit))
                    
                conversations = []
                for row in cursor.fetchall():
                    conversations.append({
                        'user_input': row[0],
                        'intent': row[1],
                        'commands': json.loads(row[2]) if row[2] else [],
                        'executed': bool(row[3]),
                        'results': json.loads(row[4]) if row[4] else None,
                        'timestamp': row[5]
                    })
                    
                return conversations
                
        except Exception as e:
            logger.error(f"Erro ao obter histórico: {e}")
            return []

# Inicializar interface inteligente
ssh_automation = SSHAutomationCore("manus_master_2024")
intelligent_interface = IntelligentSSHInterface(ssh_automation)

# Endpoints da API para interface inteligente
@app.route('/api/ai/process', methods=['POST'])
@require_auth
def process_ai_request():
    """Processar solicitação em linguagem natural"""
    try:
        data = request.json
        user_input = data.get('input', '')
        target_servers = data.get('servers', [])
        session_id = data.get('session_id', 'default')
        
        if not user_input:
            return jsonify({'error': 'Input é obrigatório'}), 400
            
        # Processar com IA
        result = intelligent_interface.process_natural_language_request(
            user_input, request.user['user_id'], session_id, target_servers
        )
        
        # Log da ação
        user_manager.log_action(
            request.user['user_id'], 'ai_request', None,
            f"Solicitação IA: {user_input}",
            request.remote_addr, request.user_agent.string
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro ao processar solicitação IA: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/execute', methods=['POST'])
@require_auth
def execute_ai_commands():
    """Executar comandos gerados pela IA"""
    try:
        data = request.json
        commands = data.get('commands', [])
        target_servers = data.get('servers', [])
        rollback_commands = data.get('rollback_commands', [])
        session_id = data.get('session_id', 'default')
        
        if not commands or not target_servers:
            return jsonify({'error': 'Comandos e servidores são obrigatórios'}), 400
            
        # Executar comandos
        result = intelligent_interface.execute_ai_generated_commands(
            commands, target_servers, request.user['user_id'], session_id, rollback_commands
        )
        
        # Log da execução
        user_manager.log_action(
            request.user['user_id'], 'ai_execute', ','.join(target_servers),
            f"Execução IA: {len(commands)} comandos",
            request.remote_addr, request.user_agent.string
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro ao executar comandos IA: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/conversation', methods=['GET'])
@require_auth
def get_ai_conversation():
    """Obter histórico de conversas com IA"""
    try:
        session_id = request.args.get('session_id')
        limit = request.args.get('limit', 50, type=int)
        
        conversations = intelligent_interface.get_conversation_history(
            request.user['user_id'], session_id, limit
        )
        
        return jsonify({'conversations': conversations})
        
    except Exception as e:
        logger.error(f"Erro ao obter conversas: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/suggestions', methods=['POST'])
@require_auth
def get_ai_suggestions():
    """Obter sugestões da IA baseadas no contexto"""
    try:
        data = request.json
        context = data.get('context', '')
        server_info = data.get('server_info', {})
        
        # Gerar sugestões baseadas no contexto
        suggestions = []
        
        if 'install' in context.lower():
            suggestions.extend([
                "Instalar Docker no servidor",
                "Configurar Nginx como proxy reverso",
                "Instalar e configurar MySQL",
                "Configurar ambiente Python com pip"
            ])
        elif 'monitor' in context.lower():
            suggestions.extend([
                "Verificar uso de CPU e memória",
                "Listar processos em execução",
                "Verificar espaço em disco",
                "Verificar status dos serviços"
            ])
        elif 'backup' in context.lower():
            suggestions.extend([
                "Fazer backup da base de dados",
                "Criar backup dos arquivos de configuração",
                "Sincronizar arquivos com servidor remoto",
                "Configurar backup automático"
            ])
        else:
            suggestions.extend([
                "Atualizar sistema operacional",
                "Verificar logs do sistema",
                "Listar serviços ativos",
                "Verificar conectividade de rede"
            ])
            
        return jsonify({'suggestions': suggestions})
        
    except Exception as e:
        logger.error(f"Erro ao obter sugestões: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🤖 Iniciando Manus AI SSH Intelligent Interface")
    print("Interface inteligente carregada e pronta para uso!")
    print("Endpoints disponíveis:")
    print("- POST /api/ai/process - Processar linguagem natural")
    print("- POST /api/ai/execute - Executar comandos gerados")
    print("- GET /api/ai/conversation - Histórico de conversas")
    print("- POST /api/ai/suggestions - Obter sugestões")

