#!/usr/bin/env python3
"""
Alhica AI - Interface Web e Chat Inteligente
A Primeira Plataforma do Mundo com IA Conversacional + SSH Automático

Copyright (c) 2024 Alhica AI Team
"""

import os
import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for, flash
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
import secrets

from alhica_ai_core import AlhicaAICore, ServerConfig

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('alhica_ai_web')

# Inicializar Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)

# Configurar CORS
CORS(app, origins=["*"])

# Inicializar SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Inicializar core
core = AlhicaAICore()

# Sessões ativas
active_sessions = {}

# Template HTML principal
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alhica AI - A Primeira Plataforma com IA + SSH Automático</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            color: #4a5568;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-align: center;
        }
        
        .header .subtitle {
            text-align: center;
            color: #718096;
            font-size: 1.2em;
            margin-bottom: 20px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        }
        
        .stat-card .icon {
            font-size: 2em;
            margin-bottom: 10px;
            color: #667eea;
        }
        
        .stat-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #2d3748;
        }
        
        .stat-card .label {
            color: #718096;
            margin-top: 5px;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 20px;
            margin-top: 20px;
        }
        
        .chat-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            height: 600px;
            display: flex;
            flex-direction: column;
        }
        
        .chat-header {
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        
        .chat-header h2 {
            color: #4a5568;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            background: #f7fafc;
            margin-bottom: 15px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px;
            border-radius: 10px;
            max-width: 80%;
        }
        
        .message.user {
            background: #667eea;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        
        .message.ai {
            background: #48bb78;
            color: white;
        }
        
        .message.system {
            background: #ed8936;
            color: white;
            text-align: center;
            max-width: 100%;
        }
        
        .message .timestamp {
            font-size: 0.8em;
            opacity: 0.8;
            margin-top: 5px;
        }
        
        .chat-input-container {
            display: flex;
            gap: 10px;
        }
        
        .chat-input {
            flex: 1;
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .chat-input:focus {
            border-color: #667eea;
        }
        
        .send-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
        }
        
        .send-btn:hover {
            background: #5a67d8;
        }
        
        .send-btn:disabled {
            background: #a0aec0;
            cursor: not-allowed;
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .panel h3 {
            color: #4a5568;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .server-list {
            max-height: 200px;
            overflow-y: auto;
        }
        
        .server-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            margin-bottom: 8px;
            background: #f7fafc;
        }
        
        .server-status {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .server-status.online {
            background: #48bb78;
        }
        
        .server-status.offline {
            background: #f56565;
        }
        
        .server-status.unknown {
            background: #ed8936;
        }
        
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }
        
        .btn:hover {
            background: #5a67d8;
        }
        
        .btn.secondary {
            background: #718096;
        }
        
        .btn.secondary:hover {
            background: #4a5568;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            color: #4a5568;
            font-weight: 500;
        }
        
        .form-group input, .form-group textarea, .form-group select {
            width: 100%;
            padding: 10px;
            border: 2px solid #e2e8f0;
            border-radius: 6px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .form-group input:focus, .form-group textarea:focus, .form-group select:focus {
            border-color: #667eea;
        }
        
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 1000;
        }
        
        .modal-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border-radius: 15px;
            padding: 30px;
            max-width: 500px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .modal-header h3 {
            color: #4a5568;
            margin: 0;
        }
        
        .close-btn {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #718096;
        }
        
        .close-btn:hover {
            color: #4a5568;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .alert {
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        
        .alert.success {
            background: #c6f6d5;
            color: #22543d;
            border: 1px solid #9ae6b4;
        }
        
        .alert.error {
            background: #fed7d7;
            color: #742a2a;
            border: 1px solid #fc8181;
        }
        
        .alert.warning {
            background: #fefcbf;
            color: #744210;
            border: 1px solid #f6e05e;
        }
        
        .execution-plan {
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .execution-step {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .execution-step:last-child {
            border-bottom: none;
        }
        
        .step-icon {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
        }
        
        .step-icon.pending {
            background: #ed8936;
        }
        
        .step-icon.running {
            background: #3182ce;
        }
        
        .step-icon.success {
            background: #48bb78;
        }
        
        .step-icon.error {
            background: #f56565;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1><i class="fas fa-robot"></i> Alhica AI</h1>
            <div class="subtitle">A Primeira Plataforma do Mundo com IA Conversacional + SSH Automático</div>
            
            <!-- Stats -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="icon"><i class="fas fa-users"></i></div>
                    <div class="value" id="stat-users">-</div>
                    <div class="label">Utilizadores</div>
                </div>
                <div class="stat-card">
                    <div class="icon"><i class="fas fa-server"></i></div>
                    <div class="value" id="stat-servers">-</div>
                    <div class="label">Servidores</div>
                </div>
                <div class="stat-card">
                    <div class="icon"><i class="fas fa-terminal"></i></div>
                    <div class="value" id="stat-commands">-</div>
                    <div class="label">Comandos Hoje</div>
                </div>
                <div class="stat-card">
                    <div class="icon"><i class="fas fa-check-circle"></i></div>
                    <div class="value" id="stat-success">-</div>
                    <div class="label">Taxa Sucesso</div>
                </div>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="main-content">
            <!-- Chat -->
            <div class="chat-container">
                <div class="chat-header">
                    <h2><i class="fas fa-comments"></i> Chat Inteligente</h2>
                </div>
                
                <div class="chat-messages" id="chat-messages">
                    <div class="message system">
                        <div>🤖 Olá! Sou a Alhica AI. Posso ajudar-te a gerir os teus servidores através de comandos naturais.</div>
                        <div class="timestamp">{{ current_time }}</div>
                    </div>
                </div>
                
                <div class="chat-input-container">
                    <input type="text" class="chat-input" id="chat-input" 
                           placeholder="Digite sua mensagem... (ex: 'Verificar status do servidor')" 
                           onkeypress="handleKeyPress(event)">
                    <button class="send-btn" id="send-btn" onclick="sendMessage()">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </div>
            
            <!-- Sidebar -->
            <div class="sidebar">
                <!-- Servidores -->
                <div class="panel">
                    <h3><i class="fas fa-server"></i> Servidores</h3>
                    <div class="server-list" id="server-list">
                        <div style="text-align: center; color: #718096; padding: 20px;">
                            Nenhum servidor configurado
                        </div>
                    </div>
                    <button class="btn" onclick="showAddServerModal()">
                        <i class="fas fa-plus"></i> Adicionar Servidor
                    </button>
                </div>
                
                <!-- Modelos IA -->
                <div class="panel">
                    <h3><i class="fas fa-brain"></i> Modelos IA</h3>
                    <div id="ai-models">
                        <div class="server-item">
                            <div style="display: flex; align-items: center;">
                                <div class="server-status unknown" id="qwen-status"></div>
                                <span>Qwen 3 25B</span>
                            </div>
                            <small>Generalista</small>
                        </div>
                        <div class="server-item">
                            <div style="display: flex; align-items: center;">
                                <div class="server-status unknown" id="deepseek-status"></div>
                                <span>DeepSeek-Coder</span>
                            </div>
                            <small>Código</small>
                        </div>
                        <div class="server-item">
                            <div style="display: flex; align-items: center;">
                                <div class="server-status unknown" id="wizardcoder-status"></div>
                                <span>WizardCoder</span>
                            </div>
                            <small>Automação</small>
                        </div>
                    </div>
                </div>
                
                <!-- Ações Rápidas -->
                <div class="panel">
                    <h3><i class="fas fa-bolt"></i> Ações Rápidas</h3>
                    <button class="btn" style="width: 100%; margin-bottom: 10px;" 
                            onclick="quickCommand('Verificar status de todos os servidores')">
                        <i class="fas fa-heartbeat"></i> Status Geral
                    </button>
                    <button class="btn" style="width: 100%; margin-bottom: 10px;"
                            onclick="quickCommand('Mostrar uso de recursos')">
                        <i class="fas fa-chart-line"></i> Recursos
                    </button>
                    <button class="btn" style="width: 100%; margin-bottom: 10px;"
                            onclick="quickCommand('Listar processos em execução')">
                        <i class="fas fa-list"></i> Processos
                    </button>
                    <button class="btn secondary" style="width: 100%;"
                            onclick="clearChat()">
                        <i class="fas fa-trash"></i> Limpar Chat
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Modal Adicionar Servidor -->
    <div class="modal" id="add-server-modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3><i class="fas fa-server"></i> Adicionar Servidor</h3>
                <button class="close-btn" onclick="hideAddServerModal()">&times;</button>
            </div>
            
            <form id="add-server-form" onsubmit="addServer(event)">
                <div class="form-group">
                    <label for="hostname">Hostname/IP *</label>
                    <input type="text" id="hostname" name="hostname" required 
                           placeholder="192.168.1.100 ou servidor.exemplo.com">
                </div>
                
                <div class="form-group">
                    <label for="port">Porta SSH</label>
                    <input type="number" id="port" name="port" value="22" min="1" max="65535">
                </div>
                
                <div class="form-group">
                    <label for="username">Utilizador *</label>
                    <input type="text" id="username" name="username" required 
                           placeholder="root, ubuntu, admin...">
                </div>
                
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" 
                           placeholder="Deixar vazio se usar chave SSH">
                </div>
                
                <div class="form-group">
                    <label for="description">Descrição</label>
                    <textarea id="description" name="description" rows="3" 
                              placeholder="Servidor web de produção, base de dados..."></textarea>
                </div>
                
                <div class="form-group">
                    <label for="tags">Tags (separadas por vírgula)</label>
                    <input type="text" id="tags" name="tags" 
                           placeholder="produção, web, database">
                </div>
                
                <div style="display: flex; gap: 10px; justify-content: flex-end;">
                    <button type="button" class="btn secondary" onclick="hideAddServerModal()">
                        Cancelar
                    </button>
                    <button type="submit" class="btn">
                        <i class="fas fa-plus"></i> Adicionar
                    </button>
                </div>
            </form>
        </div>
    </div>
    
    <script>
        // Variáveis globais
        let socket;
        let sessionId = generateSessionId();
        let currentUser = null;
        let isProcessing = false;
        
        // Inicializar aplicação
        document.addEventListener('DOMContentLoaded', function() {
            initializeSocket();
            loadStats();
            loadServers();
            checkAIModels();
            
            // Verificar autenticação
            checkAuth();
        });
        
        function generateSessionId() {
            return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
        }
        
        function initializeSocket() {
            socket = io();
            
            socket.on('connect', function() {
                console.log('Conectado ao servidor');
                socket.emit('join_session', {session_id: sessionId});
            });
            
            socket.on('disconnect', function() {
                console.log('Desconectado do servidor');
            });
            
            socket.on('ai_response', function(data) {
                addMessage('ai', data.response, data.model_used);
                
                if (data.execution_plan && data.execution_plan.length > 0) {
                    showExecutionPlan(data.execution_plan, data.risk_level);
                }
                
                setProcessing(false);
            });
            
            socket.on('command_result', function(data) {
                if (data.success) {
                    addMessage('system', `✅ Comando executado com sucesso: ${data.command}`);
                    if (data.stdout) {
                        addMessage('system', `Resultado: ${data.stdout}`);
                    }
                } else {
                    addMessage('system', `❌ Erro ao executar comando: ${data.command}`);
                    if (data.stderr) {
                        addMessage('system', `Erro: ${data.stderr}`);
                    }
                }
            });
            
            socket.on('error', function(data) {
                addMessage('system', `❌ Erro: ${data.message}`);
                setProcessing(false);
            });
        }
        
        function checkAuth() {
            fetch('/api/auth/check')
                .then(response => response.json())
                .then(data => {
                    if (data.authenticated) {
                        currentUser = data.user;
                    } else {
                        // Redirecionar para login se necessário
                        // window.location.href = '/login';
                    }
                })
                .catch(error => {
                    console.error('Erro ao verificar autenticação:', error);
                });
        }
        
        function loadStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('stat-users').textContent = data.total_users || 0;
                    document.getElementById('stat-servers').textContent = data.total_servers || 0;
                    document.getElementById('stat-commands').textContent = data.commands_today || 0;
                    document.getElementById('stat-success').textContent = (data.success_rate || 0) + '%';
                })
                .catch(error => {
                    console.error('Erro ao carregar estatísticas:', error);
                });
        }
        
        function loadServers() {
            fetch('/api/servers')
                .then(response => response.json())
                .then(data => {
                    const serverList = document.getElementById('server-list');
                    
                    if (data.length === 0) {
                        serverList.innerHTML = `
                            <div style="text-align: center; color: #718096; padding: 20px;">
                                Nenhum servidor configurado
                            </div>
                        `;
                        return;
                    }
                    
                    serverList.innerHTML = data.map(server => `
                        <div class="server-item">
                            <div style="display: flex; align-items: center;">
                                <div class="server-status ${server.status}"></div>
                                <div>
                                    <div style="font-weight: 500;">${server.hostname}</div>
                                    <small style="color: #718096;">${server.description || 'Sem descrição'}</small>
                                </div>
                            </div>
                            <button class="btn" style="font-size: 12px; padding: 4px 8px;" 
                                    onclick="testServer(${server.id})">
                                <i class="fas fa-plug"></i>
                            </button>
                        </div>
                    `).join('');
                })
                .catch(error => {
                    console.error('Erro ao carregar servidores:', error);
                });
        }
        
        function checkAIModels() {
            fetch('/api/ai/health')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('qwen-status').className = 
                        'server-status ' + (data.qwen ? 'online' : 'offline');
                    document.getElementById('deepseek-status').className = 
                        'server-status ' + (data.deepseek ? 'online' : 'offline');
                    document.getElementById('wizardcoder-status').className = 
                        'server-status ' + (data.wizardcoder ? 'online' : 'offline');
                })
                .catch(error => {
                    console.error('Erro ao verificar modelos IA:', error);
                });
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            
            if (!message || isProcessing) return;
            
            addMessage('user', message);
            input.value = '';
            setProcessing(true);
            
            socket.emit('chat_message', {
                message: message,
                session_id: sessionId
            });
        }
        
        function quickCommand(command) {
            const input = document.getElementById('chat-input');
            input.value = command;
            sendMessage();
        }
        
        function addMessage(type, content, model = null) {
            const messagesContainer = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            
            let modelInfo = '';
            if (model) {
                const modelNames = {
                    'qwen': 'Qwen 3 25B',
                    'deepseek': 'DeepSeek-Coder',
                    'wizardcoder': 'WizardCoder'
                };
                modelInfo = `<small style="opacity: 0.8;">[${modelNames[model] || model}]</small><br>`;
            }
            
            messageDiv.innerHTML = `
                <div>${modelInfo}${content}</div>
                <div class="timestamp">${new Date().toLocaleTimeString()}</div>
            `;
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function setProcessing(processing) {
            isProcessing = processing;
            const sendBtn = document.getElementById('send-btn');
            const chatInput = document.getElementById('chat-input');
            
            if (processing) {
                sendBtn.innerHTML = '<div class="loading"></div>';
                sendBtn.disabled = true;
                chatInput.disabled = true;
            } else {
                sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
                sendBtn.disabled = false;
                chatInput.disabled = false;
                chatInput.focus();
            }
        }
        
        function showExecutionPlan(plan, riskLevel) {
            const riskColors = {
                'low': '#48bb78',
                'medium': '#ed8936',
                'high': '#f56565',
                'critical': '#e53e3e'
            };
            
            const planHtml = `
                <div class="execution-plan">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <strong>Plano de Execução</strong>
                        <span style="background: ${riskColors[riskLevel]}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">
                            ${riskLevel.toUpperCase()}
                        </span>
                    </div>
                    ${plan.map((step, index) => `
                        <div class="execution-step">
                            <div class="step-icon pending">${index + 1}</div>
                            <div>
                                <div style="font-weight: 500;">${step.command || step.description}</div>
                                <small style="color: #718096;">${step.description || ''}</small>
                            </div>
                        </div>
                    `).join('')}
                    <div style="margin-top: 15px; text-align: center;">
                        <button class="btn" onclick="executeCurrentPlan()">
                            <i class="fas fa-play"></i> Executar Plano
                        </button>
                        <button class="btn secondary" onclick="cancelCurrentPlan()">
                            <i class="fas fa-times"></i> Cancelar
                        </button>
                    </div>
                </div>
            `;
            
            addMessage('system', planHtml);
        }
        
        function executeCurrentPlan() {
            // Implementar execução do plano
            addMessage('system', '🚀 Executando plano...');
        }
        
        function cancelCurrentPlan() {
            addMessage('system', '❌ Plano cancelado pelo utilizador');
        }
        
        function clearChat() {
            const messagesContainer = document.getElementById('chat-messages');
            messagesContainer.innerHTML = `
                <div class="message system">
                    <div>🤖 Chat limpo. Como posso ajudar?</div>
                    <div class="timestamp">${new Date().toLocaleTimeString()}</div>
                </div>
            `;
        }
        
        function showAddServerModal() {
            document.getElementById('add-server-modal').style.display = 'block';
        }
        
        function hideAddServerModal() {
            document.getElementById('add-server-modal').style.display = 'none';
            document.getElementById('add-server-form').reset();
        }
        
        function addServer(event) {
            event.preventDefault();
            
            const formData = new FormData(event.target);
            const serverData = {
                hostname: formData.get('hostname'),
                port: parseInt(formData.get('port')) || 22,
                username: formData.get('username'),
                password: formData.get('password'),
                description: formData.get('description'),
                tags: formData.get('tags').split(',').map(tag => tag.trim()).filter(tag => tag)
            };
            
            fetch('/api/servers', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(serverData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    hideAddServerModal();
                    loadServers();
                    loadStats();
                    addMessage('system', `✅ Servidor ${serverData.hostname} adicionado com sucesso!`);
                } else {
                    alert('Erro ao adicionar servidor: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Erro ao adicionar servidor:', error);
                alert('Erro ao adicionar servidor');
            });
        }
        
        function testServer(serverId) {
            fetch(`/api/servers/${serverId}/test`, {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        addMessage('system', `✅ Conexão com servidor testada com sucesso!`);
                    } else {
                        addMessage('system', `❌ Falha ao conectar: ${data.error}`);
                    }
                    loadServers();
                })
                .catch(error => {
                    console.error('Erro ao testar servidor:', error);
                });
        }
        
        // Atualizar dados periodicamente
        setInterval(() => {
            loadStats();
            checkAIModels();
        }, 30000); // 30 segundos
        
        // Fechar modal ao clicar fora
        window.onclick = function(event) {
            const modal = document.getElementById('add-server-modal');
            if (event.target === modal) {
                hideAddServerModal();
            }
        }
    </script>
</body>
</html>
"""

# Rotas da API
@app.route('/')
def index():
    """Página principal"""
    return render_template_string(HTML_TEMPLATE, current_time=datetime.now().strftime('%H:%M:%S'))

@app.route('/api/auth/check')
def check_auth():
    """Verificar autenticação"""
    # Implementação simplificada - usar admin por padrão
    return jsonify({
        'authenticated': True,
        'user': {
            'id': 1,
            'username': 'admin',
            'role': 'admin'
        }
    })

@app.route('/api/stats')
def get_stats():
    """Obter estatísticas do sistema"""
    try:
        stats = core.get_system_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/servers')
def get_servers():
    """Obter lista de servidores"""
    try:
        # Usar user_id = 1 (admin) por padrão
        servers = core.get_servers(1)
        return jsonify(servers)
    except Exception as e:
        logger.error(f"Erro ao obter servidores: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/servers', methods=['POST'])
def add_server():
    """Adicionar servidor"""
    try:
        data = request.get_json()
        
        # Criar configuração do servidor
        server_config = ServerConfig(
            hostname=data['hostname'],
            port=data.get('port', 22),
            username=data['username'],
            password=data.get('password'),
            description=data.get('description', ''),
            tags=data.get('tags', [])
        )
        
        # Adicionar servidor (user_id = 1 por padrão)
        server_id = core.add_server(1, server_config)
        
        return jsonify({
            'success': True,
            'server_id': server_id,
            'message': 'Servidor adicionado com sucesso'
        })
        
    except Exception as e:
        logger.error(f"Erro ao adicionar servidor: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/servers/<int:server_id>/test', methods=['POST'])
def test_server(server_id):
    """Testar conexão com servidor"""
    try:
        # Obter configuração do servidor
        servers = core.database.execute_query(
            "SELECT * FROM servers WHERE id = ?",
            (server_id,)
        )
        
        if not servers:
            return jsonify({'success': False, 'error': 'Servidor não encontrado'}), 404
        
        server_data = servers[0]
        
        # Desencriptar password se existir
        password = None
        if server_data['password_encrypted']:
            password = core.security.decrypt(server_data['password_encrypted'])
        
        server_config = ServerConfig(
            hostname=server_data['hostname'],
            port=server_data['port'],
            username=server_data['username'],
            password=password
        )
        
        # Testar conexão
        success = core.ssh_manager.test_connection(server_config)
        
        # Atualizar status na base de dados
        status = 'online' if success else 'offline'
        core.database.execute_update(
            "UPDATE servers SET status = ?, last_seen = ? WHERE id = ?",
            (status, datetime.now() if success else None, server_id)
        )
        
        return jsonify({
            'success': success,
            'message': 'Conexão bem-sucedida' if success else 'Falha na conexão'
        })
        
    except Exception as e:
        logger.error(f"Erro ao testar servidor {server_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/health')
def ai_health():
    """Verificar saúde dos modelos IA"""
    try:
        core.ai_manager._check_models_health()
        return jsonify(core.ai_manager.model_health)
    except Exception as e:
        logger.error(f"Erro ao verificar saúde da IA: {e}")
        return jsonify({'error': str(e)}), 500

# Eventos SocketIO
@socketio.on('connect')
def handle_connect():
    """Cliente conectado"""
    logger.info(f"Cliente conectado: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Cliente desconectado"""
    logger.info(f"Cliente desconectado: {request.sid}")

@socketio.on('join_session')
def handle_join_session(data):
    """Cliente juntou-se à sessão"""
    session_id = data.get('session_id')
    join_room(session_id)
    active_sessions[request.sid] = session_id
    logger.info(f"Cliente {request.sid} juntou-se à sessão {session_id}")

@socketio.on('chat_message')
def handle_chat_message(data):
    """Processar mensagem do chat"""
    try:
        message = data.get('message')
        session_id = data.get('session_id')
        
        if not message or not session_id:
            emit('error', {'message': 'Mensagem ou sessão inválida'})
            return
        
        logger.info(f"Mensagem recebida: {message}")
        
        # Processar mensagem com IA (user_id = 1 por padrão)
        result = core.process_chat_message(1, session_id, message)
        
        if result['success']:
            # Enviar resposta da IA
            emit('ai_response', {
                'response': result['response'],
                'model_used': result['model_used'],
                'execution_plan': result.get('execution_plan', []),
                'risk_level': result.get('risk_level', 'low'),
                'requires_confirmation': result.get('requires_confirmation', False)
            }, room=session_id)
            
            # Se há plano de execução e não requer confirmação, executar
            if (result.get('execution_plan') and 
                not result.get('requires_confirmation', False) and
                len(result.get('execution_plan', [])) > 0):
                
                # Executar em primeiro servidor disponível (simplificado)
                servers = core.get_servers(1)
                if servers:
                    server_id = servers[0]['id']
                    try:
                        execution_results = core.execute_plan(
                            1, session_id, server_id, result['execution_plan']
                        )
                        
                        for exec_result in execution_results:
                            emit('command_result', {
                                'command': exec_result.command,
                                'success': exec_result.success,
                                'stdout': exec_result.stdout,
                                'stderr': exec_result.stderr,
                                'server': exec_result.server
                            }, room=session_id)
                            
                    except Exception as e:
                        emit('error', {'message': f'Erro na execução: {str(e)}'}, room=session_id)
        else:
            emit('error', {'message': result.get('error', 'Erro desconhecido')})
            
    except Exception as e:
        logger.error(f"Erro ao processar mensagem do chat: {e}")
        emit('error', {'message': 'Erro interno do servidor'})

if __name__ == '__main__':
    # Criar diretórios necessários
    os.makedirs('/var/log/alhica-ai', exist_ok=True)
    os.makedirs('/var/lib/alhica-ai', exist_ok=True)
    os.makedirs('/etc/alhica', exist_ok=True)
    
    logger.info("Iniciando Alhica AI Web Interface...")
    
    # Executar aplicação
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)

