#!/usr/bin/env python3
"""
Alhica AI - Sistema Avançado de Segurança e Gestão de Credenciais
A Primeira Plataforma do Mundo com IA Conversacional + SSH Automático

Copyright (c) 2024 Alhica AI Team
"""

import os
import json
import hashlib
import secrets
import base64
import time
import hmac
import qrcode
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
import threading
import sqlite3
from pathlib import Path

# Dependências de segurança
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import pyotp
import jwt

# Dependências para auditoria
import geoip2.database
import user_agents

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('alhica_ai_security')

@dataclass
class SecurityEvent:
    """Evento de segurança"""
    event_type: str  # login, logout, command_execution, credential_access, etc.
    user_id: Optional[int]
    ip_address: str
    user_agent: str
    location: Optional[str]
    risk_score: float  # 0.0 a 1.0
    details: Dict[str, Any]
    timestamp: datetime
    session_id: Optional[str] = None
    
@dataclass
class AccessAttempt:
    """Tentativa de acesso"""
    username: str
    ip_address: str
    success: bool
    timestamp: datetime
    failure_reason: Optional[str] = None
    user_agent: Optional[str] = None

@dataclass
class MFAConfig:
    """Configuração de autenticação multi-fator"""
    user_id: int
    secret: str
    backup_codes: List[str]
    enabled: bool
    last_used: Optional[datetime] = None

class AdvancedEncryption:
    """Sistema avançado de encriptação"""
    
    def __init__(self, master_key_path: str = "/etc/alhica/security/master.key"):
        self.master_key_path = master_key_path
        self.key_rotation_interval = timedelta(days=30)
        self._ensure_key_infrastructure()
        
    def _ensure_key_infrastructure(self):
        """Garantir infraestrutura de chaves"""
        os.makedirs(os.path.dirname(self.master_key_path), exist_ok=True)
        
        # Criar chave mestra se não existir
        if not os.path.exists(self.master_key_path):
            self._generate_master_key()
        
        # Verificar se precisa rotacionar chaves
        self._check_key_rotation()
    
    def _generate_master_key(self):
        """Gerar chave mestra"""
        # Gerar chave RSA para encriptação de chaves
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )
        
        # Serializar chave privada
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Salvar chave mestra
        with open(self.master_key_path, 'wb') as f:
            f.write(private_pem)
        
        os.chmod(self.master_key_path, 0o600)
        
        # Gerar chave pública
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        public_key_path = self.master_key_path.replace('.key', '.pub')
        with open(public_key_path, 'wb') as f:
            f.write(public_pem)
        
        logger.info("Master key pair generated")
    
    def _load_private_key(self):
        """Carregar chave privada"""
        with open(self.master_key_path, 'rb') as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )
        return private_key
    
    def _check_key_rotation(self):
        """Verificar se precisa rotacionar chaves"""
        if os.path.exists(self.master_key_path):
            key_age = datetime.now() - datetime.fromtimestamp(
                os.path.getmtime(self.master_key_path)
            )
            
            if key_age > self.key_rotation_interval:
                logger.warning("Master key rotation needed")
                # Implementar rotação automática se necessário
    
    def encrypt_data(self, data: str, context: str = "default") -> str:
        """Encriptar dados com contexto"""
        # Gerar chave simétrica única
        symmetric_key = Fernet.generate_key()
        fernet = Fernet(symmetric_key)
        
        # Encriptar dados
        encrypted_data = fernet.encrypt(data.encode())
        
        # Encriptar chave simétrica com chave mestra
        private_key = self._load_private_key()
        public_key = private_key.public_key()
        
        encrypted_key = public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Combinar dados encriptados
        result = {
            'encrypted_data': base64.b64encode(encrypted_data).decode(),
            'encrypted_key': base64.b64encode(encrypted_key).decode(),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        return base64.b64encode(json.dumps(result).encode()).decode()
    
    def decrypt_data(self, encrypted_payload: str) -> str:
        """Desencriptar dados"""
        try:
            # Decodificar payload
            payload = json.loads(base64.b64decode(encrypted_payload).decode())
            
            # Desencriptar chave simétrica
            private_key = self._load_private_key()
            encrypted_key = base64.b64decode(payload['encrypted_key'])
            
            symmetric_key = private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Desencriptar dados
            fernet = Fernet(symmetric_key)
            encrypted_data = base64.b64decode(payload['encrypted_data'])
            decrypted_data = fernet.decrypt(encrypted_data)
            
            return decrypted_data.decode()
            
        except Exception as e:
            logger.error(f"Erro ao desencriptar dados: {e}")
            raise ValueError("Falha na desencriptação")

class MFAManager:
    """Gestor de autenticação multi-fator"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_mfa_tables()
    
    def _init_mfa_tables(self):
        """Inicializar tabelas MFA"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mfa_config (
                    user_id INTEGER PRIMARY KEY,
                    secret TEXT NOT NULL,
                    backup_codes TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT 0,
                    last_used TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mfa_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    code_used TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
    
    def setup_mfa(self, user_id: int, app_name: str = "Alhica AI") -> Dict[str, Any]:
        """Configurar MFA para utilizador"""
        # Gerar secret
        secret = pyotp.random_base32()
        
        # Gerar códigos de backup
        backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
        
        # Criar URI para QR code
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=f"user_{user_id}",
            issuer_name=app_name
        )
        
        # Gerar QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Converter para base64
        img_buffer = io.BytesIO()
        qr_img.save(img_buffer, format='PNG')
        qr_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Salvar configuração (ainda não ativada)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO mfa_config (user_id, secret, backup_codes, enabled)
                VALUES (?, ?, ?, 0)
            """, (user_id, secret, json.dumps(backup_codes)))
        
        return {
            'secret': secret,
            'backup_codes': backup_codes,
            'qr_code': qr_base64,
            'manual_entry_key': secret
        }
    
    def verify_and_enable_mfa(self, user_id: int, verification_code: str) -> bool:
        """Verificar código e ativar MFA"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT secret FROM mfa_config WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return False
            
            secret = result[0]
            totp = pyotp.TOTP(secret)
            
            if totp.verify(verification_code):
                # Ativar MFA
                conn.execute("""
                    UPDATE mfa_config 
                    SET enabled = 1, last_used = CURRENT_TIMESTAMP 
                    WHERE user_id = ?
                """, (user_id,))
                
                # Registar tentativa bem-sucedida
                conn.execute("""
                    INSERT INTO mfa_attempts (user_id, code_used, success)
                    VALUES (?, ?, 1)
                """, (user_id, verification_code))
                
                return True
            
            # Registar tentativa falhada
            conn.execute("""
                INSERT INTO mfa_attempts (user_id, code_used, success)
                VALUES (?, ?, 0)
            """, (user_id, verification_code))
            
            return False
    
    def verify_mfa_code(self, user_id: int, code: str, ip_address: str = None, 
                       user_agent: str = None) -> bool:
        """Verificar código MFA"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT secret, backup_codes, enabled 
                FROM mfa_config 
                WHERE user_id = ? AND enabled = 1
            """, (user_id,))
            result = cursor.fetchone()
            
            if not result:
                return False
            
            secret, backup_codes_json, enabled = result
            backup_codes = json.loads(backup_codes_json)
            
            success = False
            
            # Verificar código TOTP
            totp = pyotp.TOTP(secret)
            if totp.verify(code):
                success = True
                # Atualizar último uso
                conn.execute("""
                    UPDATE mfa_config 
                    SET last_used = CURRENT_TIMESTAMP 
                    WHERE user_id = ?
                """, (user_id,))
            
            # Verificar código de backup
            elif code.upper() in backup_codes:
                success = True
                # Remover código de backup usado
                backup_codes.remove(code.upper())
                conn.execute("""
                    UPDATE mfa_config 
                    SET backup_codes = ?, last_used = CURRENT_TIMESTAMP 
                    WHERE user_id = ?
                """, (json.dumps(backup_codes), user_id))
            
            # Registar tentativa
            conn.execute("""
                INSERT INTO mfa_attempts (user_id, code_used, success, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, code, success, ip_address, user_agent))
            
            return success
    
    def is_mfa_enabled(self, user_id: int) -> bool:
        """Verificar se MFA está ativado"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT enabled FROM mfa_config WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            return result and result[0]
    
    def disable_mfa(self, user_id: int) -> bool:
        """Desativar MFA"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE mfa_config SET enabled = 0 WHERE user_id = ?",
                (user_id,)
            )
            return True
    
    def regenerate_backup_codes(self, user_id: int) -> List[str]:
        """Regenerar códigos de backup"""
        backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE mfa_config 
                SET backup_codes = ? 
                WHERE user_id = ?
            """, (json.dumps(backup_codes), user_id))
        
        return backup_codes

class SecurityAnalyzer:
    """Analisador de segurança e deteção de anomalias"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.risk_thresholds = {
            'failed_login_attempts': 5,
            'suspicious_location_change': 1000,  # km
            'unusual_time_access': 2,  # horas fora do padrão
            'command_risk_score': 0.7
        }
        self._init_security_tables()
    
    def _init_security_tables(self):
        """Inicializar tabelas de segurança"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    user_id INTEGER,
                    ip_address TEXT,
                    user_agent TEXT,
                    location TEXT,
                    risk_score REAL,
                    details TEXT,
                    session_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                );
                
                CREATE TABLE IF NOT EXISTS access_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    failure_reason TEXT,
                    user_agent TEXT,
                    location TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS blocked_ips (
                    ip_address TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    blocked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    block_count INTEGER DEFAULT 1
                );
                
                CREATE TABLE IF NOT EXISTS user_behavior_patterns (
                    user_id INTEGER PRIMARY KEY,
                    typical_login_hours TEXT,
                    typical_locations TEXT,
                    typical_commands TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_access_attempts_ip ON access_attempts(ip_address);
                CREATE INDEX IF NOT EXISTS idx_access_attempts_timestamp ON access_attempts(timestamp);
            """)
    
    def analyze_login_attempt(self, username: str, ip_address: str, 
                            user_agent: str, success: bool,
                            failure_reason: str = None) -> float:
        """Analisar tentativa de login e calcular risco"""
        risk_score = 0.0
        
        # Verificar tentativas falhadas recentes
        recent_failures = self._get_recent_failed_attempts(ip_address, hours=1)
        if recent_failures >= self.risk_thresholds['failed_login_attempts']:
            risk_score += 0.8
            self._block_ip(ip_address, "Múltiplas tentativas falhadas", hours=24)
        
        # Analisar localização
        location = self._get_location_from_ip(ip_address)
        if location:
            location_risk = self._analyze_location_risk(username, location)
            risk_score += location_risk
        
        # Analisar user agent
        ua_risk = self._analyze_user_agent_risk(username, user_agent)
        risk_score += ua_risk
        
        # Analisar horário
        time_risk = self._analyze_time_risk(username)
        risk_score += time_risk
        
        # Registar tentativa
        self._record_access_attempt(
            username, ip_address, success, failure_reason, user_agent, location
        )
        
        # Registar evento de segurança
        if risk_score > 0.5:
            self._record_security_event(
                'suspicious_login',
                None,  # user_id será preenchido se login for bem-sucedido
                ip_address,
                user_agent,
                location,
                risk_score,
                {
                    'username': username,
                    'recent_failures': recent_failures,
                    'success': success,
                    'failure_reason': failure_reason
                }
            )
        
        return min(risk_score, 1.0)
    
    def analyze_command_execution(self, user_id: int, command: str, 
                                server_hostname: str, ip_address: str) -> float:
        """Analisar execução de comando"""
        risk_score = 0.0
        
        # Comandos de alto risco
        high_risk_commands = [
            'rm -rf', 'dd if=', 'mkfs', 'fdisk', 'shutdown', 'reboot',
            'passwd', 'userdel', 'chmod 777', 'chown -R'
        ]
        
        medium_risk_commands = [
            'sudo', 'su -', 'systemctl stop', 'service stop',
            'iptables', 'ufw', 'firewall-cmd'
        ]
        
        command_lower = command.lower()
        
        # Verificar comandos perigosos
        for risky_cmd in high_risk_commands:
            if risky_cmd in command_lower:
                risk_score += 0.8
                break
        
        for risky_cmd in medium_risk_commands:
            if risky_cmd in command_lower:
                risk_score += 0.4
                break
        
        # Analisar padrões de comportamento
        behavior_risk = self._analyze_command_behavior(user_id, command)
        risk_score += behavior_risk
        
        # Registar evento se risco alto
        if risk_score > self.risk_thresholds['command_risk_score']:
            self._record_security_event(
                'high_risk_command',
                user_id,
                ip_address,
                None,
                None,
                risk_score,
                {
                    'command': command,
                    'server': server_hostname,
                    'risk_factors': self._identify_risk_factors(command)
                }
            )
        
        return min(risk_score, 1.0)
    
    def _get_recent_failed_attempts(self, ip_address: str, hours: int = 1) -> int:
        """Obter tentativas falhadas recentes"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM access_attempts 
                WHERE ip_address = ? AND success = 0 
                AND timestamp > datetime('now', '-{} hours')
            """.format(hours), (ip_address,))
            return cursor.fetchone()[0]
    
    def _get_location_from_ip(self, ip_address: str) -> Optional[str]:
        """Obter localização a partir do IP"""
        try:
            # Implementação simplificada - em produção usar GeoIP2
            if ip_address.startswith('192.168.') or ip_address.startswith('10.'):
                return "Local Network"
            return "Unknown"
        except:
            return None
    
    def _analyze_location_risk(self, username: str, location: str) -> float:
        """Analisar risco baseado na localização"""
        # Implementação simplificada
        # Em produção, comparar com localizações típicas do utilizador
        return 0.0
    
    def _analyze_user_agent_risk(self, username: str, user_agent: str) -> float:
        """Analisar risco baseado no user agent"""
        if not user_agent:
            return 0.3
        
        # Verificar se é um user agent suspeito
        suspicious_agents = ['bot', 'crawler', 'scanner', 'automated']
        if any(agent in user_agent.lower() for agent in suspicious_agents):
            return 0.6
        
        return 0.0
    
    def _analyze_time_risk(self, username: str) -> float:
        """Analisar risco baseado no horário"""
        current_hour = datetime.now().hour
        
        # Horários suspeitos (madrugada)
        if current_hour < 6 or current_hour > 23:
            return 0.2
        
        return 0.0
    
    def _analyze_command_behavior(self, user_id: int, command: str) -> float:
        """Analisar comportamento de comandos"""
        # Implementação simplificada
        # Em produção, analisar padrões históricos do utilizador
        return 0.0
    
    def _identify_risk_factors(self, command: str) -> List[str]:
        """Identificar fatores de risco no comando"""
        factors = []
        
        if 'rm -rf' in command:
            factors.append('Comando destrutivo de arquivos')
        if 'sudo' in command:
            factors.append('Execução com privilégios elevados')
        if any(word in command for word in ['passwd', 'user']):
            factors.append('Manipulação de utilizadores')
        
        return factors
    
    def _block_ip(self, ip_address: str, reason: str, hours: int = 24):
        """Bloquear IP"""
        expires_at = datetime.now() + timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO blocked_ips (ip_address, reason, expires_at, block_count)
                VALUES (?, ?, ?, COALESCE((SELECT block_count FROM blocked_ips WHERE ip_address = ?) + 1, 1))
            """, (ip_address, reason, expires_at, ip_address))
        
        logger.warning(f"IP {ip_address} blocked: {reason}")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Verificar se IP está bloqueado"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 1 FROM blocked_ips 
                WHERE ip_address = ? AND (expires_at IS NULL OR expires_at > datetime('now'))
            """, (ip_address,))
            return cursor.fetchone() is not None
    
    def _record_access_attempt(self, username: str, ip_address: str, success: bool,
                             failure_reason: str, user_agent: str, location: str):
        """Registar tentativa de acesso"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO access_attempts (username, ip_address, success, failure_reason, user_agent, location)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (username, ip_address, success, failure_reason, user_agent, location))
    
    def _record_security_event(self, event_type: str, user_id: Optional[int],
                             ip_address: str, user_agent: str, location: str,
                             risk_score: float, details: Dict[str, Any],
                             session_id: str = None):
        """Registar evento de segurança"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO security_events (event_type, user_id, ip_address, user_agent, 
                                           location, risk_score, details, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (event_type, user_id, ip_address, user_agent, location, 
                  risk_score, json.dumps(details), session_id))
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Obter dados para dashboard de segurança"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Eventos recentes
            cursor.execute("""
                SELECT * FROM security_events 
                WHERE timestamp > datetime('now', '-24 hours')
                ORDER BY timestamp DESC LIMIT 50
            """)
            recent_events = [dict(row) for row in cursor.fetchall()]
            
            # IPs bloqueados
            cursor.execute("""
                SELECT * FROM blocked_ips 
                WHERE expires_at IS NULL OR expires_at > datetime('now')
                ORDER BY blocked_at DESC
            """)
            blocked_ips = [dict(row) for row in cursor.fetchall()]
            
            # Estatísticas
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_events,
                    SUM(CASE WHEN risk_score > 0.7 THEN 1 ELSE 0 END) as high_risk_events,
                    COUNT(DISTINCT ip_address) as unique_ips
                FROM security_events 
                WHERE timestamp > datetime('now', '-24 hours')
            """)
            stats = dict(cursor.fetchone())
            
            return {
                'recent_events': recent_events,
                'blocked_ips': blocked_ips,
                'stats': stats
            }

class SessionManager:
    """Gestor de sessões seguras"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.active_sessions = {}
        self.session_timeout = timedelta(hours=8)
    
    def create_session(self, user_id: int, ip_address: str, user_agent: str) -> str:
        """Criar sessão segura"""
        session_id = secrets.token_urlsafe(32)
        
        session_data = {
            'user_id': user_id,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'csrf_token': secrets.token_urlsafe(32)
        }
        
        self.active_sessions[session_id] = session_data
        
        # Gerar JWT token
        jwt_payload = {
            'session_id': session_id,
            'user_id': user_id,
            'exp': datetime.now() + self.session_timeout,
            'iat': datetime.now()
        }
        
        jwt_token = jwt.encode(jwt_payload, self.secret_key, algorithm='HS256')
        
        return jwt_token
    
    def validate_session(self, jwt_token: str, ip_address: str = None) -> Optional[Dict]:
        """Validar sessão"""
        try:
            payload = jwt.decode(jwt_token, self.secret_key, algorithms=['HS256'])
            session_id = payload['session_id']
            
            if session_id not in self.active_sessions:
                return None
            
            session_data = self.active_sessions[session_id]
            
            # Verificar timeout
            if datetime.now() - session_data['last_activity'] > self.session_timeout:
                del self.active_sessions[session_id]
                return None
            
            # Verificar IP (opcional)
            if ip_address and session_data['ip_address'] != ip_address:
                logger.warning(f"IP mismatch for session {session_id}")
                # Pode escolher invalidar ou apenas alertar
            
            # Atualizar última atividade
            session_data['last_activity'] = datetime.now()
            
            return session_data
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def invalidate_session(self, jwt_token: str):
        """Invalidar sessão"""
        try:
            payload = jwt.decode(jwt_token, self.secret_key, algorithms=['HS256'])
            session_id = payload['session_id']
            
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                
        except jwt.InvalidTokenError:
            pass
    
    def cleanup_expired_sessions(self):
        """Limpar sessões expiradas"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in self.active_sessions.items():
            if current_time - session_data['last_activity'] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

class AlhicaSecurityManager:
    """Gestor principal de segurança da Alhica AI"""
    
    def __init__(self, db_path: str = "/var/lib/alhica-ai/alhica.db"):
        self.db_path = db_path
        self.encryption = AdvancedEncryption()
        self.mfa = MFAManager(db_path)
        self.analyzer = SecurityAnalyzer(db_path)
        self.session_manager = SessionManager(secrets.token_hex(32))
        
        # Thread para limpeza periódica
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Iniciar thread de limpeza periódica"""
        def cleanup_worker():
            while True:
                try:
                    self.session_manager.cleanup_expired_sessions()
                    time.sleep(300)  # 5 minutos
                except Exception as e:
                    logger.error(f"Erro na limpeza periódica: {e}")
                    time.sleep(60)
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def authenticate_user(self, username: str, password: str, mfa_code: str = None,
                         ip_address: str = None, user_agent: str = None) -> Optional[str]:
        """Autenticar utilizador com análise de segurança"""
        
        # Verificar se IP está bloqueado
        if ip_address and self.analyzer.is_ip_blocked(ip_address):
            logger.warning(f"Blocked IP attempted login: {ip_address}")
            return None
        
        # Analisar tentativa de login
        risk_score = self.analyzer.analyze_login_attempt(
            username, ip_address or '127.0.0.1', user_agent or '', False
        )
        
        # Se risco muito alto, bloquear
        if risk_score > 0.9:
            logger.warning(f"High risk login attempt blocked: {username} from {ip_address}")
            return None
        
        # Verificar credenciais básicas (implementar integração com core)
        # user = core.authenticate_user(username, password)
        # if not user:
        #     return None
        
        # Simulação para desenvolvimento
        if username == "admin" and password == "admin123":
            user_id = 1
        else:
            return None
        
        # Verificar MFA se ativado
        if self.mfa.is_mfa_enabled(user_id):
            if not mfa_code:
                raise ValueError("MFA code required")
            
            if not self.mfa.verify_mfa_code(user_id, mfa_code, ip_address, user_agent):
                return None
        
        # Criar sessão segura
        jwt_token = self.session_manager.create_session(
            user_id, ip_address or '127.0.0.1', user_agent or ''
        )
        
        # Registar login bem-sucedido
        self.analyzer.analyze_login_attempt(
            username, ip_address or '127.0.0.1', user_agent or '', True
        )
        
        return jwt_token
    
    def validate_request(self, jwt_token: str, ip_address: str = None) -> Optional[Dict]:
        """Validar pedido com token JWT"""
        return self.session_manager.validate_session(jwt_token, ip_address)
    
    def logout_user(self, jwt_token: str):
        """Fazer logout do utilizador"""
        self.session_manager.invalidate_session(jwt_token)
    
    def setup_user_mfa(self, user_id: int) -> Dict[str, Any]:
        """Configurar MFA para utilizador"""
        return self.mfa.setup_mfa(user_id)
    
    def enable_user_mfa(self, user_id: int, verification_code: str) -> bool:
        """Ativar MFA para utilizador"""
        return self.mfa.verify_and_enable_mfa(user_id, verification_code)
    
    def analyze_command_security(self, user_id: int, command: str, 
                               server_hostname: str, ip_address: str) -> float:
        """Analisar segurança de comando"""
        return self.analyzer.analyze_command_execution(
            user_id, command, server_hostname, ip_address
        )
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Obter dashboard de segurança"""
        return self.analyzer.get_security_dashboard()
    
    def encrypt_sensitive_data(self, data: str, context: str = "default") -> str:
        """Encriptar dados sensíveis"""
        return self.encryption.encrypt_data(data, context)
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Desencriptar dados sensíveis"""
        return self.encryption.decrypt_data(encrypted_data)

if __name__ == "__main__":
    # Teste do sistema de segurança
    security = AlhicaSecurityManager()
    
    print("🔐 Alhica AI Security Manager inicializado!")
    
    # Testar autenticação
    try:
        token = security.authenticate_user(
            "admin", "admin123", 
            ip_address="192.168.1.100", 
            user_agent="Mozilla/5.0 Test"
        )
        
        if token:
            print("✅ Autenticação bem-sucedida")
            
            # Validar sessão
            session = security.validate_request(token, "192.168.1.100")
            if session:
                print(f"✅ Sessão válida para utilizador {session['user_id']}")
            
        else:
            print("❌ Falha na autenticação")
            
    except ValueError as e:
        print(f"⚠️ {e}")
    
    # Testar encriptação
    test_data = "password123"
    encrypted = security.encrypt_sensitive_data(test_data, "test")
    decrypted = security.decrypt_sensitive_data(encrypted)
    
    print(f"🔒 Encriptação: {test_data} -> {encrypted[:50]}...")
    print(f"🔓 Desencriptação: {decrypted}")
    print(f"✅ Encriptação funcional: {test_data == decrypted}")
    
    # Dashboard de segurança
    dashboard = security.get_security_dashboard()
    print(f"📊 Eventos de segurança: {dashboard['stats']['total_events']}")
    print(f"🚨 Eventos de alto risco: {dashboard['stats']['high_risk_events']}")
    print(f"🔒 IPs bloqueados: {len(dashboard['blocked_ips'])}")

