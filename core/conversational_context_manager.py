#!/usr/bin/env python3
"""
Alhica AI - Sistema de Gestão de Contexto Conversacional
Sistema avançado para manter contexto, memória e continuidade nas conversas

Copyright (c) 2024 Alhica AI Team
"""

import os
import json
import logging
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import uuid
import threading
from contextlib import contextmanager

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/alhica-ai/context_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    """Mensagem individual na conversa"""
    id: str
    user_id: str
    session_id: str
    content: str
    message_type: str  # user, assistant, system
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    processed: bool = False

@dataclass
class ConversationContext:
    """Contexto completo de uma conversa"""
    session_id: str
    user_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    current_intent: Optional[str] = None
    active_entities: Dict[str, Any] = field(default_factory=dict)
    conversation_state: str = "active"  # active, waiting, completed, error
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextualEntity:
    """Entidade contextual extraída da conversa"""
    name: str
    value: Any
    entity_type: str
    confidence: float
    source_message_id: str
    first_mentioned: datetime
    last_mentioned: datetime
    frequency: int = 1
    aliases: List[str] = field(default_factory=list)

@dataclass
class ConversationSummary:
    """Resumo de uma conversa"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime]
    message_count: int
    main_topics: List[str]
    resolved_intents: List[str]
    key_entities: Dict[str, Any]
    outcome: str  # successful, partial, failed, abandoned
    satisfaction_score: Optional[float] = None

class ConversationalContextManager:
    """Sistema de gestão de contexto conversacional"""
    
    def __init__(self, db_path: str = "/opt/alhica-ai/data/conversation_context.db"):
        self.db_path = db_path
        self.active_sessions: Dict[str, ConversationContext] = {}
        self.session_lock = threading.RLock()
        self.max_context_length = 50  # Máximo de mensagens por contexto
        self.context_timeout = timedelta(hours=2)  # Timeout de sessão
        
        # Cache de entidades
        self.entity_cache: Dict[str, Dict[str, ContextualEntity]] = defaultdict(dict)
        
        # Configurar base de dados
        self._setup_database()
        
        # Carregar sessões ativas
        self._load_active_sessions()
        
        logger.info("💬 Sistema de contexto conversacional inicializado")
    
    def _setup_database(self):
        """Configurar base de dados para contexto conversacional"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                # Tabela de sessões
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS conversation_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                        end_time DATETIME,
                        last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                        conversation_state TEXT DEFAULT 'active',
                        message_count INTEGER DEFAULT 0,
                        metadata TEXT DEFAULT '{}',
                        preferences TEXT DEFAULT '{}'
                    )
                ''')
                
                # Tabela de mensagens
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        intent TEXT,
                        entities TEXT DEFAULT '{}',
                        metadata TEXT DEFAULT '{}',
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        processed BOOLEAN DEFAULT FALSE,
                        FOREIGN KEY (session_id) REFERENCES conversation_sessions (session_id)
                    )
                ''')
                
                # Tabela de entidades contextuais
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS contextual_entities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        value TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        source_message_id TEXT NOT NULL,
                        first_mentioned DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_mentioned DATETIME DEFAULT CURRENT_TIMESTAMP,
                        frequency INTEGER DEFAULT 1,
                        aliases TEXT DEFAULT '[]',
                        FOREIGN KEY (session_id) REFERENCES conversation_sessions (session_id)
                    )
                ''')
                
                # Tabela de resumos de conversa
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS conversation_summaries (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        start_time DATETIME NOT NULL,
                        end_time DATETIME,
                        message_count INTEGER NOT NULL,
                        main_topics TEXT DEFAULT '[]',
                        resolved_intents TEXT DEFAULT '[]',
                        key_entities TEXT DEFAULT '{}',
                        outcome TEXT DEFAULT 'active',
                        satisfaction_score REAL,
                        summary_text TEXT,
                        FOREIGN KEY (session_id) REFERENCES conversation_sessions (session_id)
                    )
                ''')
                
                # Tabela de preferências do utilizador
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        user_id TEXT PRIMARY KEY,
                        communication_style TEXT DEFAULT 'formal',
                        preferred_language TEXT DEFAULT 'pt',
                        confirmation_level TEXT DEFAULT 'medium',
                        notification_preferences TEXT DEFAULT '{}',
                        custom_aliases TEXT DEFAULT '{}',
                        learning_enabled BOOLEAN DEFAULT TRUE,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Índices para performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_session ON conversation_messages(session_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON conversation_messages(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_entities_session ON contextual_entities(session_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user ON conversation_sessions(user_id)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Erro ao configurar base de dados: {e}")
    
    def _load_active_sessions(self):
        """Carregar sessões ativas da base de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Carregar sessões ativas das últimas 24 horas
                cursor.execute('''
                    SELECT session_id, user_id, conversation_state, last_activity, metadata, preferences
                    FROM conversation_sessions 
                    WHERE conversation_state = 'active' 
                    AND last_activity > datetime('now', '-24 hours')
                ''')
                
                for row in cursor.fetchall():
                    session_id, user_id, state, last_activity, metadata_json, preferences_json = row
                    
                    # Carregar mensagens da sessão
                    cursor.execute('''
                        SELECT id, content, message_type, intent, entities, metadata, timestamp, processed
                        FROM conversation_messages 
                        WHERE session_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (session_id, self.max_context_length))
                    
                    messages = []
                    for msg_row in cursor.fetchall():
                        msg_id, content, msg_type, intent, entities_json, msg_metadata_json, timestamp, processed = msg_row
                        
                        message = ConversationMessage(
                            id=msg_id,
                            user_id=user_id,
                            session_id=session_id,
                            content=content,
                            message_type=msg_type,
                            intent=intent,
                            entities=json.loads(entities_json or '{}'),
                            metadata=json.loads(msg_metadata_json or '{}'),
                            timestamp=datetime.fromisoformat(timestamp),
                            processed=bool(processed)
                        )
                        messages.append(message)
                    
                    # Criar contexto
                    context = ConversationContext(
                        session_id=session_id,
                        user_id=user_id,
                        messages=list(reversed(messages)),  # Ordem cronológica
                        conversation_state=state,
                        last_activity=datetime.fromisoformat(last_activity),
                        metadata=json.loads(metadata_json or '{}'),
                        preferences=json.loads(preferences_json or '{}')
                    )
                    
                    # Determinar intenção atual
                    context.current_intent = self._determine_current_intent(context)
                    
                    # Extrair entidades ativas
                    context.active_entities = self._extract_active_entities(context)
                    
                    self.active_sessions[session_id] = context
                
                logger.info(f"Carregadas {len(self.active_sessions)} sessões ativas")
                
        except Exception as e:
            logger.error(f"Erro ao carregar sessões ativas: {e}")
    
    @contextmanager
    def session_context(self, session_id: str):
        """Context manager para acesso thread-safe às sessões"""
        with self.session_lock:
            yield self.active_sessions.get(session_id)
    
    def create_session(self, user_id: str, initial_message: str = None) -> str:
        """Criar nova sessão de conversa"""
        session_id = str(uuid.uuid4())
        
        with self.session_lock:
            # Criar contexto
            context = ConversationContext(
                session_id=session_id,
                user_id=user_id,
                preferences=self._load_user_preferences(user_id)
            )
            
            # Salvar na base de dados
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO conversation_sessions 
                        (session_id, user_id, metadata, preferences)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        session_id, 
                        user_id, 
                        json.dumps(context.metadata),
                        json.dumps(context.preferences)
                    ))
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"Erro ao criar sessão na DB: {e}")
            
            # Adicionar mensagem inicial se fornecida
            if initial_message:
                self.add_message(session_id, initial_message, "user")
            
            self.active_sessions[session_id] = context
            
        logger.info(f"Nova sessão criada: {session_id} para utilizador {user_id}")
        return session_id
    
    def add_message(self, session_id: str, content: str, message_type: str, 
                   intent: str = None, entities: Dict[str, Any] = None,
                   metadata: Dict[str, Any] = None) -> str:
        """Adicionar mensagem à conversa"""
        message_id = str(uuid.uuid4())
        entities = entities or {}
        metadata = metadata or {}
        
        with self.session_context(session_id) as context:
            if not context:
                logger.warning(f"Sessão não encontrada: {session_id}")
                return message_id
            
            # Criar mensagem
            message = ConversationMessage(
                id=message_id,
                user_id=context.user_id,
                session_id=session_id,
                content=content,
                message_type=message_type,
                intent=intent,
                entities=entities,
                metadata=metadata
            )
            
            # Adicionar ao contexto
            context.messages.append(message)
            context.last_activity = datetime.now()
            
            # Manter limite de mensagens
            if len(context.messages) > self.max_context_length:
                # Arquivar mensagens antigas
                old_messages = context.messages[:-self.max_context_length]
                context.messages = context.messages[-self.max_context_length:]
                self._archive_messages(old_messages)
            
            # Atualizar intenção atual
            if intent:
                context.current_intent = intent
            
            # Processar entidades
            if entities:
                self._process_entities(context, message, entities)
            
            # Salvar na base de dados
            self._save_message(message)
            self._update_session_activity(session_id)
            
        logger.debug(f"Mensagem adicionada: {message_id} na sessão {session_id}")
        return message_id
    
    def get_context(self, session_id: str, include_history: bool = True) -> Optional[ConversationContext]:
        """Obter contexto da conversa"""
        with self.session_context(session_id) as context:
            if not context:
                return None
            
            if not include_history:
                # Retornar apenas contexto atual sem histórico completo
                summary_context = ConversationContext(
                    session_id=context.session_id,
                    user_id=context.user_id,
                    messages=context.messages[-5:] if context.messages else [],  # Últimas 5 mensagens
                    current_intent=context.current_intent,
                    active_entities=context.active_entities,
                    conversation_state=context.conversation_state,
                    last_activity=context.last_activity,
                    metadata=context.metadata,
                    preferences=context.preferences
                )
                return summary_context
            
            return context
    
    def get_relevant_context(self, session_id: str, query: str, max_messages: int = 10) -> Dict[str, Any]:
        """Obter contexto relevante para uma query específica"""
        with self.session_context(session_id) as context:
            if not context:
                return {}
            
            query_lower = query.lower()
            relevant_messages = []
            relevant_entities = {}
            
            # Procurar mensagens relevantes
            for message in reversed(context.messages[-max_messages:]):
                relevance_score = 0
                
                # Score baseado em palavras-chave
                content_lower = message.content.lower()
                common_words = set(query_lower.split()) & set(content_lower.split())
                relevance_score += len(common_words) * 0.3
                
                # Score baseado em intenção
                if message.intent and message.intent in query_lower:
                    relevance_score += 0.5
                
                # Score baseado em entidades
                for entity_name, entity_value in message.entities.items():
                    if entity_name.lower() in query_lower or str(entity_value).lower() in query_lower:
                        relevance_score += 0.4
                        relevant_entities[entity_name] = entity_value
                
                if relevance_score > 0.2:
                    relevant_messages.append({
                        'message': message,
                        'relevance': relevance_score
                    })
            
            # Ordenar por relevância
            relevant_messages.sort(key=lambda x: x['relevance'], reverse=True)
            
            return {
                'session_id': session_id,
                'current_intent': context.current_intent,
                'relevant_messages': [item['message'] for item in relevant_messages[:5]],
                'relevant_entities': relevant_entities,
                'active_entities': context.active_entities,
                'user_preferences': context.preferences,
                'conversation_state': context.conversation_state
            }
    
    def update_intent(self, session_id: str, new_intent: str, confidence: float = 1.0):
        """Atualizar intenção atual da sessão"""
        with self.session_context(session_id) as context:
            if context:
                context.current_intent = new_intent
                context.metadata['intent_confidence'] = confidence
                context.metadata['intent_updated'] = datetime.now().isoformat()
                
                self._update_session_metadata(session_id, context.metadata)
    
    def add_entity(self, session_id: str, entity_name: str, entity_value: Any, 
                  entity_type: str, confidence: float = 1.0, source_message_id: str = None):
        """Adicionar ou atualizar entidade contextual"""
        with self.session_context(session_id) as context:
            if not context:
                return
            
            # Atualizar entidades ativas
            context.active_entities[entity_name] = {
                'value': entity_value,
                'type': entity_type,
                'confidence': confidence,
                'updated': datetime.now().isoformat()
            }
            
            # Criar ou atualizar entidade contextual
            entity_key = f"{session_id}:{entity_name}"
            
            if entity_key in self.entity_cache[session_id]:
                # Atualizar entidade existente
                entity = self.entity_cache[session_id][entity_key]
                entity.value = entity_value
                entity.confidence = max(entity.confidence, confidence)
                entity.last_mentioned = datetime.now()
                entity.frequency += 1
            else:
                # Criar nova entidade
                entity = ContextualEntity(
                    name=entity_name,
                    value=entity_value,
                    entity_type=entity_type,
                    confidence=confidence,
                    source_message_id=source_message_id or "",
                    first_mentioned=datetime.now(),
                    last_mentioned=datetime.now()
                )
                self.entity_cache[session_id][entity_key] = entity
            
            # Salvar na base de dados
            self._save_entity(context.user_id, session_id, entity)
    
    def get_entity_value(self, session_id: str, entity_name: str) -> Optional[Any]:
        """Obter valor de uma entidade específica"""
        with self.session_context(session_id) as context:
            if context and entity_name in context.active_entities:
                return context.active_entities[entity_name]['value']
        return None
    
    def get_conversation_summary(self, session_id: str) -> Optional[ConversationSummary]:
        """Obter resumo da conversa"""
        with self.session_context(session_id) as context:
            if not context:
                return None
            
            # Extrair tópicos principais
            main_topics = self._extract_main_topics(context)
            
            # Extrair intenções resolvidas
            resolved_intents = list(set([msg.intent for msg in context.messages if msg.intent]))
            
            # Entidades-chave
            key_entities = {name: data['value'] for name, data in context.active_entities.items()}
            
            # Determinar outcome
            outcome = self._determine_conversation_outcome(context)
            
            summary = ConversationSummary(
                session_id=session_id,
                user_id=context.user_id,
                start_time=context.messages[0].timestamp if context.messages else datetime.now(),
                end_time=context.last_activity if context.conversation_state != 'active' else None,
                message_count=len(context.messages),
                main_topics=main_topics,
                resolved_intents=resolved_intents,
                key_entities=key_entities,
                outcome=outcome
            )
            
            return summary
    
    def end_session(self, session_id: str, outcome: str = "completed", satisfaction_score: float = None):
        """Terminar sessão de conversa"""
        with self.session_context(session_id) as context:
            if not context:
                return
            
            context.conversation_state = "completed"
            context.last_activity = datetime.now()
            
            # Criar resumo final
            summary = self.get_conversation_summary(session_id)
            if summary:
                summary.outcome = outcome
                summary.satisfaction_score = satisfaction_score
                summary.end_time = datetime.now()
                
                # Salvar resumo
                self._save_conversation_summary(summary)
            
            # Atualizar na base de dados
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        UPDATE conversation_sessions 
                        SET conversation_state = ?, end_time = CURRENT_TIMESTAMP
                        WHERE session_id = ?
                    ''', (outcome, session_id))
                    conn.commit()
            except Exception as e:
                logger.error(f"Erro ao terminar sessão: {e}")
            
            # Remover das sessões ativas
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
        
        logger.info(f"Sessão terminada: {session_id} com outcome: {outcome}")
    
    def cleanup_expired_sessions(self):
        """Limpar sessões expiradas"""
        expired_sessions = []
        current_time = datetime.now()
        
        with self.session_lock:
            for session_id, context in self.active_sessions.items():
                if current_time - context.last_activity > self.context_timeout:
                    expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.end_session(session_id, outcome="timeout")
        
        if expired_sessions:
            logger.info(f"Limpas {len(expired_sessions)} sessões expiradas")
    
    def get_user_conversation_history(self, user_id: str, limit: int = 10) -> List[ConversationSummary]:
        """Obter histórico de conversas do utilizador"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT session_id, start_time, end_time, message_count, 
                           main_topics, resolved_intents, key_entities, outcome, satisfaction_score
                    FROM conversation_summaries 
                    WHERE user_id = ? 
                    ORDER BY start_time DESC 
                    LIMIT ?
                ''', (user_id, limit))
                
                summaries = []
                for row in cursor.fetchall():
                    summary = ConversationSummary(
                        session_id=row[0],
                        user_id=user_id,
                        start_time=datetime.fromisoformat(row[1]),
                        end_time=datetime.fromisoformat(row[2]) if row[2] else None,
                        message_count=row[3],
                        main_topics=json.loads(row[4] or '[]'),
                        resolved_intents=json.loads(row[5] or '[]'),
                        key_entities=json.loads(row[6] or '{}'),
                        outcome=row[7],
                        satisfaction_score=row[8]
                    )
                    summaries.append(summary)
                
                return summaries
                
        except Exception as e:
            logger.error(f"Erro ao obter histórico: {e}")
            return []
    
    def _determine_current_intent(self, context: ConversationContext) -> Optional[str]:
        """Determinar intenção atual baseada nas mensagens recentes"""
        recent_messages = context.messages[-3:] if context.messages else []
        
        for message in reversed(recent_messages):
            if message.intent and message.message_type == "user":
                return message.intent
        
        return None
    
    def _extract_active_entities(self, context: ConversationContext) -> Dict[str, Any]:
        """Extrair entidades ativas do contexto"""
        active_entities = {}
        
        # Processar mensagens recentes
        recent_messages = context.messages[-10:] if context.messages else []
        
        for message in recent_messages:
            for entity_name, entity_value in message.entities.items():
                active_entities[entity_name] = {
                    'value': entity_value,
                    'source': message.id,
                    'timestamp': message.timestamp.isoformat()
                }
        
        return active_entities
    
    def _process_entities(self, context: ConversationContext, message: ConversationMessage, entities: Dict[str, Any]):
        """Processar entidades de uma mensagem"""
        for entity_name, entity_value in entities.items():
            self.add_entity(
                context.session_id,
                entity_name,
                entity_value,
                entity_type="extracted",
                confidence=0.8,
                source_message_id=message.id
            )
    
    def _extract_main_topics(self, context: ConversationContext) -> List[str]:
        """Extrair tópicos principais da conversa"""
        topics = []
        
        # Baseado em intenções
        intents = [msg.intent for msg in context.messages if msg.intent]
        intent_counts = Counter(intents)
        topics.extend([intent for intent, count in intent_counts.most_common(3)])
        
        # Baseado em entidades
        entity_types = []
        for msg in context.messages:
            for entity_name in msg.entities.keys():
                entity_types.append(entity_name)
        
        entity_counts = Counter(entity_types)
        topics.extend([entity for entity, count in entity_counts.most_common(2)])
        
        return list(set(topics))[:5]  # Máximo 5 tópicos únicos
    
    def _determine_conversation_outcome(self, context: ConversationContext) -> str:
        """Determinar outcome da conversa"""
        if not context.messages:
            return "abandoned"
        
        last_messages = context.messages[-3:]
        
        # Procurar por indicadores de sucesso
        success_indicators = ["sucesso", "obrigado", "perfeito", "resolvido", "funcionou"]
        for message in last_messages:
            if any(indicator in message.content.lower() for indicator in success_indicators):
                return "successful"
        
        # Procurar por indicadores de erro
        error_indicators = ["erro", "problema", "não funciona", "falhou"]
        for message in last_messages:
            if any(indicator in message.content.lower() for indicator in error_indicators):
                return "failed"
        
        # Se há intenções resolvidas, considerar parcialmente bem-sucedido
        if context.current_intent:
            return "partial"
        
        return "active"
    
    def _load_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Carregar preferências do utilizador"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT communication_style, preferred_language, confirmation_level,
                           notification_preferences, custom_aliases, learning_enabled
                    FROM user_preferences 
                    WHERE user_id = ?
                ''', (user_id,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'communication_style': result[0],
                        'preferred_language': result[1],
                        'confirmation_level': result[2],
                        'notification_preferences': json.loads(result[3] or '{}'),
                        'custom_aliases': json.loads(result[4] or '{}'),
                        'learning_enabled': bool(result[5])
                    }
        except Exception as e:
            logger.error(f"Erro ao carregar preferências: {e}")
        
        # Preferências padrão
        return {
            'communication_style': 'formal',
            'preferred_language': 'pt',
            'confirmation_level': 'medium',
            'notification_preferences': {},
            'custom_aliases': {},
            'learning_enabled': True
        }
    
    def _save_message(self, message: ConversationMessage):
        """Salvar mensagem na base de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO conversation_messages 
                    (id, session_id, user_id, content, message_type, intent, entities, metadata, timestamp, processed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    message.id,
                    message.session_id,
                    message.user_id,
                    message.content,
                    message.message_type,
                    message.intent,
                    json.dumps(message.entities),
                    json.dumps(message.metadata),
                    message.timestamp.isoformat(),
                    message.processed
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao salvar mensagem: {e}")
    
    def _save_entity(self, user_id: str, session_id: str, entity: ContextualEntity):
        """Salvar entidade na base de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Verificar se entidade já existe
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id FROM contextual_entities 
                    WHERE session_id = ? AND name = ?
                ''', (session_id, entity.name))
                
                if cursor.fetchone():
                    # Atualizar entidade existente
                    conn.execute('''
                        UPDATE contextual_entities 
                        SET value = ?, confidence = ?, last_mentioned = ?, frequency = ?
                        WHERE session_id = ? AND name = ?
                    ''', (
                        json.dumps(entity.value),
                        entity.confidence,
                        entity.last_mentioned.isoformat(),
                        entity.frequency,
                        session_id,
                        entity.name
                    ))
                else:
                    # Inserir nova entidade
                    conn.execute('''
                        INSERT INTO contextual_entities 
                        (session_id, user_id, name, value, entity_type, confidence, 
                         source_message_id, first_mentioned, last_mentioned, frequency, aliases)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id,
                        user_id,
                        entity.name,
                        json.dumps(entity.value),
                        entity.entity_type,
                        entity.confidence,
                        entity.source_message_id,
                        entity.first_mentioned.isoformat(),
                        entity.last_mentioned.isoformat(),
                        entity.frequency,
                        json.dumps(entity.aliases)
                    ))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao salvar entidade: {e}")
    
    def _save_conversation_summary(self, summary: ConversationSummary):
        """Salvar resumo da conversa"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO conversation_summaries 
                    (session_id, user_id, start_time, end_time, message_count,
                     main_topics, resolved_intents, key_entities, outcome, satisfaction_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    summary.session_id,
                    summary.user_id,
                    summary.start_time.isoformat(),
                    summary.end_time.isoformat() if summary.end_time else None,
                    summary.message_count,
                    json.dumps(summary.main_topics),
                    json.dumps(summary.resolved_intents),
                    json.dumps(summary.key_entities),
                    summary.outcome,
                    summary.satisfaction_score
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao salvar resumo: {e}")
    
    def _update_session_activity(self, session_id: str):
        """Atualizar atividade da sessão"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE conversation_sessions 
                    SET last_activity = CURRENT_TIMESTAMP, 
                        message_count = message_count + 1
                    WHERE session_id = ?
                ''', (session_id,))
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao atualizar atividade da sessão: {e}")
    
    def _update_session_metadata(self, session_id: str, metadata: Dict[str, Any]):
        """Atualizar metadata da sessão"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE conversation_sessions 
                    SET metadata = ?
                    WHERE session_id = ?
                ''', (json.dumps(metadata), session_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao atualizar metadata: {e}")
    
    def _archive_messages(self, messages: List[ConversationMessage]):
        """Arquivar mensagens antigas"""
        # Por agora, apenas log. Implementar arquivamento real se necessário
        logger.debug(f"Arquivando {len(messages)} mensagens antigas")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obter estatísticas do sistema de contexto"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Sessões ativas
                active_sessions = len(self.active_sessions)
                
                # Total de sessões
                cursor.execute("SELECT COUNT(*) FROM conversation_sessions")
                total_sessions = cursor.fetchone()[0]
                
                # Total de mensagens
                cursor.execute("SELECT COUNT(*) FROM conversation_messages")
                total_messages = cursor.fetchone()[0]
                
                # Sessões por outcome
                cursor.execute('''
                    SELECT outcome, COUNT(*) 
                    FROM conversation_summaries 
                    GROUP BY outcome
                ''')
                outcomes = dict(cursor.fetchall())
                
                # Média de mensagens por sessão
                cursor.execute('''
                    SELECT AVG(message_count) 
                    FROM conversation_summaries
                ''')
                avg_messages = cursor.fetchone()[0] or 0
                
                # Utilizadores únicos
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversation_sessions")
                unique_users = cursor.fetchone()[0]
                
                return {
                    'active_sessions': active_sessions,
                    'total_sessions': total_sessions,
                    'total_messages': total_messages,
                    'unique_users': unique_users,
                    'average_messages_per_session': round(avg_messages, 2),
                    'session_outcomes': outcomes,
                    'context_timeout_hours': self.context_timeout.total_seconds() / 3600,
                    'max_context_length': self.max_context_length
                }
                
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            return {}

def main():
    """Função principal para teste"""
    context_manager = ConversationalContextManager()
    
    # Teste básico
    user_id = "test_user"
    
    # Criar sessão
    session_id = context_manager.create_session(user_id, "Olá, preciso de ajuda")
    print(f"Sessão criada: {session_id}")
    
    # Adicionar mensagens
    context_manager.add_message(session_id, "Quero instalar nginx no servidor web-01", "user", 
                               intent="install_software", 
                               entities={"software": "nginx", "server": "web-01"})
    
    context_manager.add_message(session_id, "Vou instalar nginx no servidor web-01 para si.", "assistant")
    
    context_manager.add_message(session_id, "Também preciso de configurar SSL", "user",
                               intent="configure_service",
                               entities={"service": "ssl", "server": "web-01"})
    
    # Obter contexto
    context = context_manager.get_context(session_id)
    print(f"\nContexto atual:")
    print(f"  Intenção: {context.current_intent}")
    print(f"  Entidades ativas: {context.active_entities}")
    print(f"  Mensagens: {len(context.messages)}")
    
    # Obter contexto relevante
    relevant = context_manager.get_relevant_context(session_id, "configurar ssl")
    print(f"\nContexto relevante para 'configurar ssl':")
    print(f"  Mensagens relevantes: {len(relevant['relevant_messages'])}")
    print(f"  Entidades relevantes: {relevant['relevant_entities']}")
    
    # Obter resumo
    summary = context_manager.get_conversation_summary(session_id)
    print(f"\nResumo da conversa:")
    print(f"  Tópicos principais: {summary.main_topics}")
    print(f"  Intenções resolvidas: {summary.resolved_intents}")
    print(f"  Outcome: {summary.outcome}")
    
    # Estatísticas
    stats = context_manager.get_statistics()
    print(f"\nEstatísticas:")
    print(json.dumps(stats, indent=2))
    
    # Terminar sessão
    context_manager.end_session(session_id, "successful", 4.5)
    print(f"\nSessão terminada")

if __name__ == "__main__":
    main()

