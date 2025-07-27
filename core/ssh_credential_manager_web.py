#!/usr/bin/env python3
"""
Manus AI SSH Credential Manager Web Interface
============================================

Sistema revolucionário de gestão segura de credenciais SSH com interface web avançada.
Primeira implementação no mundo de gestão de credenciais SSH integrada com IA.

Autor: Manus AI Team
Versão: 1.0.0
Data: 2024

Funcionalidades:
- Interface web responsiva para gestão de credenciais
- Encriptação AES-256 de todas as credenciais
- Autenticação multi-fator
- Auditoria completa de acessos
- Importação/exportação segura
- Gestão de chaves SSH
- Rotação automática de credenciais
- Dashboard de monitorização
"""

from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
import os
import json
import hashlib
import secrets
import qrcode
import io
import base64
from datetime import datetime, timedelta
import pyotp
import sqlite3
from ssh_automation_core import CredentialManager, SSHCredentials, SSHAutomationCore
import logging
from functools import wraps
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)

# Configurações globais
MASTER_PASSWORD = "manus_master_2024"  # Em produção, usar variável de ambiente
JWT_SECRET = secrets.token_hex(32)
MFA_ISSUER = "Manus AI SSH Manager"

# Inicializar componentes
ssh_automation = SSHAutomationCore(MASTER_PASSWORD)
credential_manager = ssh_automation.credential_manager

class UserManager:
    """Gestão de utilizadores do sistema"""
    
    def __init__(self, db_path: str = "/etc/manus/users.db"):
        self.db_path = db_path
        self._setup_database()
        
    def _setup_database(self):
        """Configurar base de dados de utilizadores"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    mfa_secret TEXT,
                    mfa_enabled BOOLEAN DEFAULT FALSE,
                    role TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    resource TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Criar utilizador admin padrão se não existir
            cursor = conn.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
            if cursor.fetchone()[0] == 0:
                admin_password = generate_password_hash("admin123")
                conn.execute("""
                    INSERT INTO users (username, email, password_hash, role)
                    VALUES ('admin', 'admin@manus.ai', ?, 'admin')
                """, (admin_password,))
                
    def create_user(self, username: str, email: str, password: str, role: str = 'user') -> bool:
        """Criar novo utilizador"""
        try:
            password_hash = generate_password_hash(password)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO users (username, email, password_hash, role)
                    VALUES (?, ?, ?, ?)
                """, (username, email, password_hash, role))
            return True
        except sqlite3.IntegrityError:
            return False
            
    def authenticate_user(self, username: str, password: str) -> dict:
        """Autenticar utilizador"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, username, email, password_hash, mfa_enabled, mfa_secret, role, is_active
                FROM users WHERE username = ? AND is_active = TRUE
            """, (username,))
            
            user = cursor.fetchone()
            if not user:
                return None
                
            if not check_password_hash(user[3], password):
                return None
                
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'mfa_enabled': user[4],
                'mfa_secret': user[5],
                'role': user[6]
            }
            
    def verify_mfa(self, user_id: int, token: str) -> bool:
        """Verificar token MFA"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT mfa_secret FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()
            
            if not result or not result[0]:
                return False
                
            totp = pyotp.TOTP(result[0])
            return totp.verify(token)
            
    def setup_mfa(self, user_id: int) -> str:
        """Configurar MFA para utilizador"""
        secret = pyotp.random_base32()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE users SET mfa_secret = ?, mfa_enabled = TRUE WHERE id = ?
            """, (secret, user_id))
            
        return secret
        
    def log_action(self, user_id: int, action: str, resource: str = None, 
                   details: str = None, ip_address: str = None, user_agent: str = None):
        """Registar ação no log de auditoria"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO audit_log (user_id, action, resource, details, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, action, resource, details, ip_address, user_agent))

user_manager = UserManager()

def require_auth(f):
    """Decorator para exigir autenticação"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            token = session.get('token')
            
        if not token:
            return jsonify({'error': 'Token de autenticação necessário'}), 401
            
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            request.user = payload
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expirado'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Token inválido'}), 401
            
        return f(*args, **kwargs)
    return decorated_function

def require_admin(f):
    """Decorator para exigir privilégios de admin"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(request, 'user') or request.user.get('role') != 'admin':
            return jsonify({'error': 'Privilégios de administrador necessários'}), 403
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Página principal"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="pt">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manus AI SSH Manager</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card-hover { transition: transform 0.2s; }
        .card-hover:hover { transform: translateY(-5px); }
        .status-online { color: #28a745; }
        .status-offline { color: #dc3545; }
        .sidebar { min-height: 100vh; background: #343a40; }
        .main-content { background: #f8f9fa; min-height: 100vh; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-2 sidebar text-white p-0">
                <div class="p-3">
                    <h4><i class="fas fa-robot"></i> Manus AI</h4>
                    <small>SSH Manager</small>
                </div>
                <nav class="nav flex-column">
                    <a class="nav-link text-white" href="#dashboard" onclick="showSection('dashboard')">
                        <i class="fas fa-tachometer-alt"></i> Dashboard
                    </a>
                    <a class="nav-link text-white" href="#servers" onclick="showSection('servers')">
                        <i class="fas fa-server"></i> Servidores
                    </a>
                    <a class="nav-link text-white" href="#credentials" onclick="showSection('credentials')">
                        <i class="fas fa-key"></i> Credenciais
                    </a>
                    <a class="nav-link text-white" href="#executions" onclick="showSection('executions')">
                        <i class="fas fa-terminal"></i> Execuções
                    </a>
                    <a class="nav-link text-white" href="#monitoring" onclick="showSection('monitoring')">
                        <i class="fas fa-chart-line"></i> Monitorização
                    </a>
                    <a class="nav-link text-white" href="#settings" onclick="showSection('settings')">
                        <i class="fas fa-cog"></i> Configurações
                    </a>
                </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 main-content">
                <!-- Header -->
                <nav class="navbar navbar-expand-lg navbar-light bg-white shadow-sm">
                    <div class="container-fluid">
                        <span class="navbar-brand">SSH Automation Dashboard</span>
                        <div class="navbar-nav ms-auto">
                            <div class="nav-item dropdown">
                                <a class="nav-link dropdown-toggle" href="#" id="userDropdown" role="button" data-bs-toggle="dropdown">
                                    <i class="fas fa-user"></i> <span id="username">Admin</span>
                                </a>
                                <ul class="dropdown-menu">
                                    <li><a class="dropdown-item" href="#" onclick="showMFASetup()">Configurar MFA</a></li>
                                    <li><a class="dropdown-item" href="#" onclick="logout()">Sair</a></li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </nav>
                
                <!-- Dashboard Section -->
                <div id="dashboard" class="section p-4">
                    <h2>Dashboard</h2>
                    <div class="row">
                        <div class="col-md-3">
                            <div class="card card-hover text-center">
                                <div class="card-body">
                                    <i class="fas fa-server fa-2x text-primary"></i>
                                    <h4 id="total-servers">0</h4>
                                    <p>Servidores</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card card-hover text-center">
                                <div class="card-body">
                                    <i class="fas fa-check-circle fa-2x text-success"></i>
                                    <h4 id="online-servers">0</h4>
                                    <p>Online</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card card-hover text-center">
                                <div class="card-body">
                                    <i class="fas fa-terminal fa-2x text-info"></i>
                                    <h4 id="total-executions">0</h4>
                                    <p>Execuções</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card card-hover text-center">
                                <div class="card-body">
                                    <i class="fas fa-exclamation-triangle fa-2x text-warning"></i>
                                    <h4 id="failed-executions">0</h4>
                                    <p>Falhas</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row mt-4">
                        <div class="col-md-8">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Execuções Recentes</h5>
                                </div>
                                <div class="card-body">
                                    <div class="table-responsive">
                                        <table class="table table-striped" id="recent-executions">
                                            <thead>
                                                <tr>
                                                    <th>Servidor</th>
                                                    <th>Comando</th>
                                                    <th>Status</th>
                                                    <th>Data/Hora</th>
                                                </tr>
                                            </thead>
                                            <tbody></tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Status dos Servidores</h5>
                                </div>
                                <div class="card-body" id="server-status-list">
                                    <!-- Será preenchido dinamicamente -->
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Servers Section -->
                <div id="servers" class="section p-4" style="display: none;">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2>Gestão de Servidores</h2>
                        <button class="btn btn-primary" onclick="showAddServerModal()">
                            <i class="fas fa-plus"></i> Adicionar Servidor
                        </button>
                    </div>
                    
                    <div class="card">
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-striped" id="servers-table">
                                    <thead>
                                        <tr>
                                            <th>Hostname</th>
                                            <th>IP</th>
                                            <th>OS</th>
                                            <th>CPU</th>
                                            <th>RAM</th>
                                            <th>Status</th>
                                            <th>Última Conexão</th>
                                            <th>Ações</th>
                                        </tr>
                                    </thead>
                                    <tbody></tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Credentials Section -->
                <div id="credentials" class="section p-4" style="display: none;">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2>Gestão de Credenciais</h2>
                        <div>
                            <button class="btn btn-success" onclick="exportCredentials()">
                                <i class="fas fa-download"></i> Exportar
                            </button>
                            <button class="btn btn-warning" onclick="showImportModal()">
                                <i class="fas fa-upload"></i> Importar
                            </button>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-striped" id="credentials-table">
                                    <thead>
                                        <tr>
                                            <th>Hostname</th>
                                            <th>Username</th>
                                            <th>Tipo de Auth</th>
                                            <th>Porta</th>
                                            <th>Criado</th>
                                            <th>Ações</th>
                                        </tr>
                                    </thead>
                                    <tbody></tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Executions Section -->
                <div id="executions" class="section p-4" style="display: none;">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2>Execução de Comandos</h2>
                        <button class="btn btn-primary" onclick="showExecuteCommandModal()">
                            <i class="fas fa-terminal"></i> Executar Comando
                        </button>
                    </div>
                    
                    <div class="row mb-4">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Execução Rápida</h5>
                                </div>
                                <div class="card-body">
                                    <form id="quick-execute-form">
                                        <div class="mb-3">
                                            <label class="form-label">Servidor</label>
                                            <select class="form-select" id="quick-server" required>
                                                <option value="">Selecionar servidor...</option>
                                            </select>
                                        </div>
                                        <div class="mb-3">
                                            <label class="form-label">Comando</label>
                                            <input type="text" class="form-control" id="quick-command" 
                                                   placeholder="Ex: ls -la" required>
                                        </div>
                                        <button type="submit" class="btn btn-success">
                                            <i class="fas fa-play"></i> Executar
                                        </button>
                                    </form>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Resultado da Execução</h5>
                                </div>
                                <div class="card-body">
                                    <pre id="execution-result" class="bg-dark text-light p-3" style="height: 200px; overflow-y: auto;">
Aguardando execução...
                                    </pre>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <h5>Histórico de Execuções</h5>
                        </div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-striped" id="executions-table">
                                    <thead>
                                        <tr>
                                            <th>ID</th>
                                            <th>Servidor</th>
                                            <th>Comando</th>
                                            <th>Status</th>
                                            <th>Tempo (s)</th>
                                            <th>Data/Hora</th>
                                            <th>Ações</th>
                                        </tr>
                                    </thead>
                                    <tbody></tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Monitoring Section -->
                <div id="monitoring" class="section p-4" style="display: none;">
                    <h2>Monitorização em Tempo Real</h2>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Execuções por Hora</h5>
                                </div>
                                <div class="card-body">
                                    <canvas id="executions-chart"></canvas>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Status dos Servidores</h5>
                                </div>
                                <div class="card-body">
                                    <canvas id="servers-chart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row mt-4">
                        <div class="col-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Log de Auditoria</h5>
                                </div>
                                <div class="card-body">
                                    <div class="table-responsive">
                                        <table class="table table-striped" id="audit-log-table">
                                            <thead>
                                                <tr>
                                                    <th>Utilizador</th>
                                                    <th>Ação</th>
                                                    <th>Recurso</th>
                                                    <th>IP</th>
                                                    <th>Data/Hora</th>
                                                </tr>
                                            </thead>
                                            <tbody></tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Settings Section -->
                <div id="settings" class="section p-4" style="display: none;">
                    <h2>Configurações</h2>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Configurações de Segurança</h5>
                                </div>
                                <div class="card-body">
                                    <form id="security-settings-form">
                                        <div class="mb-3">
                                            <label class="form-label">Timeout de Sessão (minutos)</label>
                                            <input type="number" class="form-control" value="30" min="5" max="480">
                                        </div>
                                        <div class="mb-3">
                                            <label class="form-label">Timeout de Comando (segundos)</label>
                                            <input type="number" class="form-control" value="300" min="30" max="3600">
                                        </div>
                                        <div class="mb-3">
                                            <div class="form-check">
                                                <input class="form-check-input" type="checkbox" id="require-mfa" checked>
                                                <label class="form-check-label" for="require-mfa">
                                                    Exigir MFA para todos os utilizadores
                                                </label>
                                            </div>
                                        </div>
                                        <div class="mb-3">
                                            <div class="form-check">
                                                <input class="form-check-input" type="checkbox" id="audit-all" checked>
                                                <label class="form-check-label" for="audit-all">
                                                    Auditar todas as ações
                                                </label>
                                            </div>
                                        </div>
                                        <button type="submit" class="btn btn-primary">Guardar</button>
                                    </form>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Gestão de Utilizadores</h5>
                                </div>
                                <div class="card-body">
                                    <button class="btn btn-success mb-3" onclick="showAddUserModal()">
                                        <i class="fas fa-user-plus"></i> Adicionar Utilizador
                                    </button>
                                    <div class="table-responsive">
                                        <table class="table table-striped" id="users-table">
                                            <thead>
                                                <tr>
                                                    <th>Username</th>
                                                    <th>Email</th>
                                                    <th>Role</th>
                                                    <th>MFA</th>
                                                    <th>Ações</th>
                                                </tr>
                                            </thead>
                                            <tbody></tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Modals -->
    <!-- Add Server Modal -->
    <div class="modal fade" id="addServerModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Adicionar Servidor</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <form id="add-server-form">
                        <div class="mb-3">
                            <label class="form-label">Hostname/IP</label>
                            <input type="text" class="form-control" id="server-hostname" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Username</label>
                            <input type="text" class="form-control" id="server-username" value="root" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Porta</label>
                            <input type="number" class="form-control" id="server-port" value="22" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Tipo de Autenticação</label>
                            <select class="form-select" id="auth-type" onchange="toggleAuthFields()">
                                <option value="password">Senha</option>
                                <option value="key">Chave Privada</option>
                                <option value="key-file">Arquivo de Chave</option>
                            </select>
                        </div>
                        <div class="mb-3" id="password-field">
                            <label class="form-label">Senha</label>
                            <input type="password" class="form-control" id="server-password">
                        </div>
                        <div class="mb-3" id="key-field" style="display: none;">
                            <label class="form-label">Chave Privada</label>
                            <textarea class="form-control" id="server-private-key" rows="5"></textarea>
                        </div>
                        <div class="mb-3" id="key-file-field" style="display: none;">
                            <label class="form-label">Caminho da Chave</label>
                            <input type="text" class="form-control" id="server-key-path">
                        </div>
                        <div class="mb-3" id="passphrase-field" style="display: none;">
                            <label class="form-label">Passphrase (opcional)</label>
                            <input type="password" class="form-control" id="server-passphrase">
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancelar</button>
                    <button type="button" class="btn btn-primary" onclick="addServer()">Adicionar</button>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Execute Command Modal -->
    <div class="modal fade" id="executeCommandModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Executar Comando</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <form id="execute-command-form">
                        <div class="mb-3">
                            <label class="form-label">Servidores</label>
                            <select class="form-select" id="execute-servers" multiple size="5">
                                <!-- Será preenchido dinamicamente -->
                            </select>
                            <small class="form-text text-muted">Segure Ctrl para selecionar múltiplos servidores</small>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Comando</label>
                            <textarea class="form-control" id="execute-command" rows="3" required></textarea>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Comando de Rollback (opcional)</label>
                            <textarea class="form-control" id="rollback-command" rows="2"></textarea>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Timeout (segundos)</label>
                            <input type="number" class="form-control" id="execute-timeout" value="300" min="30" max="3600">
                        </div>
                        <div class="mb-3">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="parallel-execution" checked>
                                <label class="form-check-label" for="parallel-execution">
                                    Execução paralela
                                </label>
                            </div>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancelar</button>
                    <button type="button" class="btn btn-primary" onclick="executeCommand()">Executar</button>
                </div>
            </div>
        </div>
    </div>
    
    <!-- MFA Setup Modal -->
    <div class="modal fade" id="mfaSetupModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Configurar Autenticação Multi-Fator</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body text-center">
                    <p>Escaneie o código QR com o seu aplicativo de autenticação:</p>
                    <div id="qr-code"></div>
                    <p class="mt-3">Ou insira manualmente a chave:</p>
                    <code id="mfa-secret"></code>
                    <div class="mt-3">
                        <label class="form-label">Código de Verificação</label>
                        <input type="text" class="form-control" id="mfa-token" maxlength="6">
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancelar</button>
                    <button type="button" class="btn btn-primary" onclick="verifyMFASetup()">Verificar</button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Estado global da aplicação
        let currentUser = null;
        let authToken = null;
        let refreshInterval = null;
        
        // Inicialização
        document.addEventListener('DOMContentLoaded', function() {
            checkAuth();
            if (authToken) {
                loadDashboard();
                startAutoRefresh();
            }
        });
        
        // Gestão de autenticação
        function checkAuth() {
            authToken = localStorage.getItem('authToken');
            if (authToken) {
                // Verificar se o token ainda é válido
                fetch('/api/auth/verify', {
                    headers: { 'Authorization': 'Bearer ' + authToken }
                })
                .then(response => {
                    if (!response.ok) {
                        logout();
                        return;
                    }
                    return response.json();
                })
                .then(data => {
                    if (data) {
                        currentUser = data.user;
                        document.getElementById('username').textContent = currentUser.username;
                    }
                })
                .catch(() => logout());
            } else {
                showLoginModal();
            }
        }
        
        function logout() {
            localStorage.removeItem('authToken');
            authToken = null;
            currentUser = null;
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
            showLoginModal();
        }
        
        // Gestão de secções
        function showSection(sectionId) {
            // Esconder todas as secções
            document.querySelectorAll('.section').forEach(section => {
                section.style.display = 'none';
            });
            
            // Mostrar secção selecionada
            document.getElementById(sectionId).style.display = 'block';
            
            // Carregar dados da secção
            switch(sectionId) {
                case 'dashboard':
                    loadDashboard();
                    break;
                case 'servers':
                    loadServers();
                    break;
                case 'credentials':
                    loadCredentials();
                    break;
                case 'executions':
                    loadExecutions();
                    break;
                case 'monitoring':
                    loadMonitoring();
                    break;
                case 'settings':
                    loadSettings();
                    break;
            }
        }
        
        // Carregar dashboard
        function loadDashboard() {
            fetch('/api/dashboard/stats', {
                headers: { 'Authorization': 'Bearer ' + authToken }
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('total-servers').textContent = data.total_servers || 0;
                document.getElementById('online-servers').textContent = data.online_servers || 0;
                document.getElementById('total-executions').textContent = data.total_executions || 0;
                document.getElementById('failed-executions').textContent = data.failed_executions || 0;
            })
            .catch(console.error);
            
            // Carregar execuções recentes
            loadRecentExecutions();
            loadServerStatus();
        }
        
        function loadRecentExecutions() {
            fetch('/api/executions/recent', {
                headers: { 'Authorization': 'Bearer ' + authToken }
            })
            .then(response => response.json())
            .then(data => {
                const tbody = document.querySelector('#recent-executions tbody');
                tbody.innerHTML = '';
                
                data.executions.forEach(exec => {
                    const row = tbody.insertRow();
                    const statusClass = exec.status === 'success' ? 'text-success' : 'text-danger';
                    const commandShort = exec.command.length > 30 ? 
                        exec.command.substring(0, 30) + '...' : exec.command;
                    
                    row.innerHTML = `
                        <td>${exec.server}</td>
                        <td><code>${commandShort}</code></td>
                        <td><span class="${statusClass}">${exec.status}</span></td>
                        <td>${new Date(exec.start_time).toLocaleString()}</td>
                    `;
                });
            })
            .catch(console.error);
        }
        
        function loadServerStatus() {
            fetch('/api/servers/status', {
                headers: { 'Authorization': 'Bearer ' + authToken }
            })
            .then(response => response.json())
            .then(data => {
                const container = document.getElementById('server-status-list');
                container.innerHTML = '';
                
                Object.entries(data.servers).forEach(([hostname, status]) => {
                    const statusClass = status === 'online' ? 'status-online' : 'status-offline';
                    const icon = status === 'online' ? 'fa-check-circle' : 'fa-times-circle';
                    
                    const div = document.createElement('div');
                    div.className = 'd-flex justify-content-between align-items-center mb-2';
                    div.innerHTML = `
                        <span>${hostname}</span>
                        <i class="fas ${icon} ${statusClass}"></i>
                    `;
                    container.appendChild(div);
                });
            })
            .catch(console.error);
        }
        
        // Carregar servidores
        function loadServers() {
            fetch('/api/servers', {
                headers: { 'Authorization': 'Bearer ' + authToken }
            })
            .then(response => response.json())
            .then(data => {
                const tbody = document.querySelector('#servers-table tbody');
                tbody.innerHTML = '';
                
                data.servers.forEach(server => {
                    const row = tbody.insertRow();
                    const statusClass = server.status === 'online' ? 'text-success' : 'text-danger';
                    const lastSeen = server.last_seen ? 
                        new Date(server.last_seen).toLocaleString() : 'Nunca';
                    
                    row.innerHTML = `
                        <td>${server.hostname}</td>
                        <td>${server.ip}</td>
                        <td>${server.os_type} ${server.os_version}</td>
                        <td>${server.cpu_cores}</td>
                        <td>${server.memory_gb.toFixed(1)} GB</td>
                        <td><span class="${statusClass}">${server.status}</span></td>
                        <td>${lastSeen}</td>
                        <td>
                            <button class="btn btn-sm btn-primary" onclick="testConnection('${server.hostname}')">
                                <i class="fas fa-plug"></i>
                            </button>
                            <button class="btn btn-sm btn-danger" onclick="removeServer('${server.hostname}')">
                                <i class="fas fa-trash"></i>
                            </button>
                        </td>
                    `;
                });
            })
            .catch(console.error);
        }
        
        // Mostrar modal de adicionar servidor
        function showAddServerModal() {
            new bootstrap.Modal(document.getElementById('addServerModal')).show();
        }
        
        // Alternar campos de autenticação
        function toggleAuthFields() {
            const authType = document.getElementById('auth-type').value;
            const passwordField = document.getElementById('password-field');
            const keyField = document.getElementById('key-field');
            const keyFileField = document.getElementById('key-file-field');
            const passphraseField = document.getElementById('passphrase-field');
            
            // Esconder todos os campos
            passwordField.style.display = 'none';
            keyField.style.display = 'none';
            keyFileField.style.display = 'none';
            passphraseField.style.display = 'none';
            
            // Mostrar campos relevantes
            switch(authType) {
                case 'password':
                    passwordField.style.display = 'block';
                    break;
                case 'key':
                    keyField.style.display = 'block';
                    passphraseField.style.display = 'block';
                    break;
                case 'key-file':
                    keyFileField.style.display = 'block';
                    passphraseField.style.display = 'block';
                    break;
            }
        }
        
        // Adicionar servidor
        function addServer() {
            const formData = {
                hostname: document.getElementById('server-hostname').value,
                username: document.getElementById('server-username').value,
                port: parseInt(document.getElementById('server-port').value),
                auth_type: document.getElementById('auth-type').value,
                password: document.getElementById('server-password').value,
                private_key: document.getElementById('server-private-key').value,
                private_key_path: document.getElementById('server-key-path').value,
                passphrase: document.getElementById('server-passphrase').value
            };
            
            fetch('/api/servers/add', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + authToken
                },
                body: JSON.stringify(formData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    bootstrap.Modal.getInstance(document.getElementById('addServerModal')).hide();
                    loadServers();
                    showAlert('Servidor adicionado com sucesso!', 'success');
                } else {
                    showAlert('Erro ao adicionar servidor: ' + data.error, 'danger');
                }
            })
            .catch(error => {
                showAlert('Erro de conexão: ' + error.message, 'danger');
            });
        }
        
        // Testar conexão
        function testConnection(hostname) {
            fetch(`/api/servers/${hostname}/test`, {
                method: 'POST',
                headers: { 'Authorization': 'Bearer ' + authToken }
            })
            .then(response => response.json())
            .then(data => {
                const alertType = data.success ? 'success' : 'danger';
                const message = data.success ? 'Conexão bem-sucedida!' : 'Falha na conexão: ' + data.error;
                showAlert(message, alertType);
            })
            .catch(error => {
                showAlert('Erro ao testar conexão: ' + error.message, 'danger');
            });
        }
        
        // Executar comando rápido
        document.getElementById('quick-execute-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const server = document.getElementById('quick-server').value;
            const command = document.getElementById('quick-command').value;
            
            if (!server || !command) return;
            
            const resultElement = document.getElementById('execution-result');
            resultElement.textContent = 'Executando comando...';
            
            fetch('/api/executions/execute', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + authToken
                },
                body: JSON.stringify({
                    servers: [server],
                    command: command,
                    timeout: 300
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const execution = data.executions[0];
                    const result = `
Servidor: ${execution.server}
Comando: ${execution.command}
Status: ${execution.status}
Código de Saída: ${execution.exit_code}
Tempo de Execução: ${execution.execution_time}s

=== STDOUT ===
${execution.stdout}

=== STDERR ===
${execution.stderr}
                    `;
                    resultElement.textContent = result;
                } else {
                    resultElement.textContent = 'Erro: ' + data.error;
                }
            })
            .catch(error => {
                resultElement.textContent = 'Erro de conexão: ' + error.message;
            });
        });
        
        // Mostrar alerta
        function showAlert(message, type) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
            alertDiv.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            
            document.body.insertBefore(alertDiv, document.body.firstChild);
            
            // Remover automaticamente após 5 segundos
            setTimeout(() => {
                if (alertDiv.parentNode) {
                    alertDiv.remove();
                }
            }, 5000);
        }
        
        // Auto-refresh
        function startAutoRefresh() {
            refreshInterval = setInterval(() => {
                const currentSection = document.querySelector('.section:not([style*="display: none"])');
                if (currentSection) {
                    const sectionId = currentSection.id;
                    if (sectionId === 'dashboard') {
                        loadDashboard();
                    }
                }
            }, 30000); // Refresh a cada 30 segundos
        }
        
        // Carregar lista de servidores para selects
        function loadServerOptions() {
            fetch('/api/servers/list', {
                headers: { 'Authorization': 'Bearer ' + authToken }
            })
            .then(response => response.json())
            .then(data => {
                const selects = ['quick-server', 'execute-servers'];
                selects.forEach(selectId => {
                    const select = document.getElementById(selectId);
                    if (select) {
                        select.innerHTML = selectId === 'quick-server' ? 
                            '<option value="">Selecionar servidor...</option>' : '';
                        
                        data.servers.forEach(server => {
                            const option = document.createElement('option');
                            option.value = server;
                            option.textContent = server;
                            select.appendChild(option);
                        });
                    }
                });
            })
            .catch(console.error);
        }
        
        // Carregar opções de servidores quando necessário
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(loadServerOptions, 1000);
        });
    </script>
</body>
</html>
    """)

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Endpoint de login"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        mfa_token = data.get('mfa_token')
        
        if not username or not password:
            return jsonify({'error': 'Username e password são obrigatórios'}), 400
            
        # Autenticar utilizador
        user = user_manager.authenticate_user(username, password)
        if not user:
            user_manager.log_action(None, 'login_failed', username, 
                                  f"Tentativa de login falhada para {username}",
                                  request.remote_addr, request.user_agent.string)
            return jsonify({'error': 'Credenciais inválidas'}), 401
            
        # Verificar MFA se ativado
        if user['mfa_enabled']:
            if not mfa_token:
                return jsonify({'error': 'Token MFA necessário', 'mfa_required': True}), 401
                
            if not user_manager.verify_mfa(user['id'], mfa_token):
                user_manager.log_action(user['id'], 'mfa_failed', None, 
                                      "Token MFA inválido",
                                      request.remote_addr, request.user_agent.string)
                return jsonify({'error': 'Token MFA inválido'}), 401
                
        # Gerar token JWT
        payload = {
            'user_id': user['id'],
            'username': user['username'],
            'role': user['role'],
            'exp': datetime.utcnow() + timedelta(hours=8)
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
        
        # Log de login bem-sucedido
        user_manager.log_action(user['id'], 'login_success', None, 
                              "Login bem-sucedido",
                              request.remote_addr, request.user_agent.string)
        
        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'role': user['role']
            }
        })
        
    except Exception as e:
        logger.error(f"Erro no login: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

@app.route('/api/auth/verify', methods=['GET'])
@require_auth
def verify_token():
    """Verificar token de autenticação"""
    return jsonify({
        'valid': True,
        'user': request.user
    })

@app.route('/api/servers/add', methods=['POST'])
@require_auth
def add_server():
    """Adicionar novo servidor"""
    try:
        data = request.json
        hostname = data.get('hostname')
        username = data.get('username')
        port = data.get('port', 22)
        auth_type = data.get('auth_type', 'password')
        
        if not hostname or not username:
            return jsonify({'error': 'Hostname e username são obrigatórios'}), 400
            
        # Preparar credenciais baseado no tipo de autenticação
        credentials_data = {
            'hostname': hostname,
            'username': username,
            'port': port
        }
        
        if auth_type == 'password':
            credentials_data['password'] = data.get('password')
        elif auth_type == 'key':
            credentials_data['private_key'] = data.get('private_key')
            credentials_data['passphrase'] = data.get('passphrase')
        elif auth_type == 'key-file':
            credentials_data['private_key_path'] = data.get('private_key_path')
            credentials_data['passphrase'] = data.get('passphrase')
            
        # Adicionar servidor
        success = ssh_automation.add_server(**credentials_data)
        
        if success:
            user_manager.log_action(request.user['user_id'], 'server_added', hostname,
                                  f"Servidor {hostname} adicionado",
                                  request.remote_addr, request.user_agent.string)
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Falha ao adicionar servidor'}), 400
            
    except Exception as e:
        logger.error(f"Erro ao adicionar servidor: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/servers', methods=['GET'])
@require_auth
def list_servers():
    """Listar servidores"""
    try:
        servers = ssh_automation.list_servers()
        servers_data = []
        
        for server in servers:
            servers_data.append({
                'hostname': server.hostname,
                'ip': server.ip,
                'os_type': server.os_type,
                'os_version': server.os_version,
                'cpu_cores': server.cpu_cores,
                'memory_gb': server.memory_gb,
                'status': server.status.value,
                'last_seen': server.last_seen.isoformat() if server.last_seen else None
            })
            
        return jsonify({'servers': servers_data})
        
    except Exception as e:
        logger.error(f"Erro ao listar servidores: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/servers/<hostname>/test', methods=['POST'])
@require_auth
def test_server_connection(hostname):
    """Testar conexão com servidor"""
    try:
        status = ssh_automation.get_server_status(hostname)
        success = status.value == 'online'
        
        user_manager.log_action(request.user['user_id'], 'connection_test', hostname,
                              f"Teste de conexão: {status.value}",
                              request.remote_addr, request.user_agent.string)
        
        return jsonify({
            'success': success,
            'status': status.value,
            'error': None if success else f"Servidor {hostname} offline"
        })
        
    except Exception as e:
        logger.error(f"Erro ao testar conexão: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/executions/execute', methods=['POST'])
@require_auth
def execute_command():
    """Executar comando em servidores"""
    try:
        data = request.json
        servers = data.get('servers', [])
        command = data.get('command')
        timeout = data.get('timeout', 300)
        parallel = data.get('parallel', True)
        rollback_command = data.get('rollback_command')
        
        if not servers or not command:
            return jsonify({'error': 'Servidores e comando são obrigatórios'}), 400
            
        # Log da execução
        user_manager.log_action(request.user['user_id'], 'command_execute', 
                              ','.join(servers), f"Comando: {command}",
                              request.remote_addr, request.user_agent.string)
        
        if parallel and len(servers) > 1:
            # Execução paralela
            executions = ssh_automation.execute_on_multiple(servers, command, timeout)
        else:
            # Execução sequencial
            executions = []
            for hostname in servers:
                if rollback_command:
                    execution = ssh_automation.execute_with_rollback(hostname, command, rollback_command, timeout)
                else:
                    execution = ssh_automation.execute_command(hostname, command, timeout)
                executions.append(execution)
                
        # Converter execuções para JSON
        executions_data = []
        for execution in executions:
            executions_data.append({
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
            })
            
        return jsonify({
            'success': True,
            'executions': executions_data
        })
        
    except Exception as e:
        logger.error(f"Erro ao executar comando: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/executions/recent', methods=['GET'])
@require_auth
def get_recent_executions():
    """Obter execuções recentes"""
    try:
        limit = request.args.get('limit', 10, type=int)
        executions = ssh_automation.get_execution_history(limit=limit)
        
        executions_data = []
        for execution in executions:
            executions_data.append({
                'id': execution.id,
                'server': execution.server,
                'command': execution.command,
                'status': execution.status.value,
                'start_time': execution.start_time.isoformat(),
                'execution_time': execution.execution_time
            })
            
        return jsonify({'executions': executions_data})
        
    except Exception as e:
        logger.error(f"Erro ao obter execuções recentes: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/stats', methods=['GET'])
@require_auth
def get_dashboard_stats():
    """Obter estatísticas do dashboard"""
    try:
        servers = ssh_automation.list_servers()
        executions = ssh_automation.get_execution_history(limit=1000)
        
        total_servers = len(servers)
        online_servers = sum(1 for s in servers if s.status.value == 'online')
        total_executions = len(executions)
        failed_executions = sum(1 for e in executions if e.status.value == 'failed')
        
        return jsonify({
            'total_servers': total_servers,
            'online_servers': online_servers,
            'total_executions': total_executions,
            'failed_executions': failed_executions
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/servers/status', methods=['GET'])
@require_auth
def get_servers_status():
    """Obter status de todos os servidores"""
    try:
        status_map = ssh_automation.health_check_all()
        servers_status = {hostname: status.value for hostname, status in status_map.items()}
        
        return jsonify({'servers': servers_status})
        
    except Exception as e:
        logger.error(f"Erro ao obter status dos servidores: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/servers/list', methods=['GET'])
@require_auth
def get_server_list():
    """Obter lista simples de servidores"""
    try:
        servers = credential_manager.list_servers()
        return jsonify({'servers': servers})
        
    except Exception as e:
        logger.error(f"Erro ao obter lista de servidores: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mfa/setup', methods=['POST'])
@require_auth
def setup_mfa():
    """Configurar MFA para utilizador"""
    try:
        user_id = request.user['user_id']
        secret = user_manager.setup_mfa(user_id)
        
        # Gerar QR code
        username = request.user['username']
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=username,
            issuer_name=MFA_ISSUER
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'secret': secret,
            'qr_code': f"data:image/png;base64,{img_str}"
        })
        
    except Exception as e:
        logger.error(f"Erro ao configurar MFA: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Iniciando Manus AI SSH Manager Web Interface")
    app.run(host='0.0.0.0', port=5004, debug=False)

