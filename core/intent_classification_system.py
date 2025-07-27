#!/usr/bin/env python3
"""
Alhica AI - Sistema Avançado de Classificação de Intenções
Sistema inteligente para classificar intenções de utilizadores e otimizar respostas

Copyright (c) 2024 Alhica AI Team
"""

import os
import json
import logging
import sqlite3
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/alhica-ai/intent_classifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Intent:
    """Definição de uma intenção"""
    name: str
    category: str
    description: str
    keywords: List[str]
    patterns: List[str]
    examples: List[str]
    confidence_threshold: float = 0.7
    requires_confirmation: bool = False
    risk_level: str = "low"
    estimated_time: int = 60  # segundos
    model_preference: Optional[str] = None  # qwen, deepseek, wizardcoder

@dataclass
class IntentPrediction:
    """Resultado da classificação de intenção"""
    intent_name: str
    confidence: float
    category: str
    model_recommendation: str
    risk_assessment: Dict[str, Any]
    execution_plan: Dict[str, Any]
    alternatives: List[Tuple[str, float]]  # (intent_name, confidence)

class IntentClassificationSystem:
    """Sistema avançado de classificação de intenções"""
    
    def __init__(self, db_path: str = "/opt/alhica-ai/data/intent_classifier.db"):
        self.db_path = db_path
        self.intents = self._load_predefined_intents()
        self.models = {}
        self.vectorizers = {}
        self.training_data = []
        
        # Configurar base de dados
        self._setup_database()
        
        # Carregar dados de treino
        self._load_training_data()
        
        # Treinar modelos
        self._train_models()
        
        logger.info("🎯 Sistema de classificação de intenções inicializado")
    
    def _load_predefined_intents(self) -> Dict[str, Intent]:
        """Carregar intenções predefinidas"""
        intents = {}
        
        # Intenções de Administração de Sistema
        intents["install_software"] = Intent(
            name="install_software",
            category="system_administration",
            description="Instalar software ou pacotes no sistema",
            keywords=["install", "instalar", "setup", "add", "adicionar"],
            patterns=[
                r"install\s+\w+",
                r"instalar\s+\w+",
                r"setup\s+\w+",
                r"adicionar\s+\w+"
            ],
            examples=[
                "install nginx",
                "instalar docker",
                "setup mysql database",
                "adicionar novo pacote"
            ],
            confidence_threshold=0.8,
            requires_confirmation=True,
            risk_level="medium",
            estimated_time=300,
            model_preference="deepseek"
        )
        
        intents["check_status"] = Intent(
            name="check_status",
            category="monitoring",
            description="Verificar status de serviços ou sistema",
            keywords=["check", "verificar", "status", "estado", "monitor"],
            patterns=[
                r"check\s+\w+",
                r"verificar\s+\w+",
                r"status\s+\w+",
                r"estado\s+do\s+\w+"
            ],
            examples=[
                "check nginx status",
                "verificar estado do mysql",
                "monitor cpu usage",
                "status do servidor"
            ],
            confidence_threshold=0.7,
            requires_confirmation=False,
            risk_level="low",
            estimated_time=10,
            model_preference="qwen"
        )
        
        intents["restart_service"] = Intent(
            name="restart_service",
            category="system_administration",
            description="Reiniciar serviços ou sistema",
            keywords=["restart", "reiniciar", "reboot", "reload"],
            patterns=[
                r"restart\s+\w+",
                r"reiniciar\s+\w+",
                r"reboot\s+\w+",
                r"reload\s+\w+"
            ],
            examples=[
                "restart nginx",
                "reiniciar apache",
                "reboot server",
                "reload configuration"
            ],
            confidence_threshold=0.9,
            requires_confirmation=True,
            risk_level="high",
            estimated_time=60,
            model_preference="wizardcoder"
        )
        
        intents["update_system"] = Intent(
            name="update_system",
            category="maintenance",
            description="Atualizar sistema ou pacotes",
            keywords=["update", "atualizar", "upgrade", "patch"],
            patterns=[
                r"update\s+\w*",
                r"atualizar\s+\w*",
                r"upgrade\s+\w*",
                r"patch\s+\w+"
            ],
            examples=[
                "update system",
                "atualizar pacotes",
                "upgrade all packages",
                "patch security vulnerabilities"
            ],
            confidence_threshold=0.8,
            requires_confirmation=True,
            risk_level="medium",
            estimated_time=600,
            model_preference="wizardcoder"
        )
        
        intents["backup_data"] = Intent(
            name="backup_data",
            category="data_management",
            description="Fazer backup de dados ou configurações",
            keywords=["backup", "copy", "copiar", "save", "guardar"],
            patterns=[
                r"backup\s+\w+",
                r"copy\s+\w+",
                r"copiar\s+\w+",
                r"save\s+\w+"
            ],
            examples=[
                "backup database",
                "copy configuration files",
                "copiar logs",
                "save current state"
            ],
            confidence_threshold=0.7,
            requires_confirmation=False,
            risk_level="low",
            estimated_time=120,
            model_preference="wizardcoder"
        )
        
        intents["configure_service"] = Intent(
            name="configure_service",
            category="configuration",
            description="Configurar serviços ou aplicações",
            keywords=["configure", "configurar", "config", "setup", "set"],
            patterns=[
                r"configure\s+\w+",
                r"configurar\s+\w+",
                r"config\s+\w+",
                r"setup\s+\w+"
            ],
            examples=[
                "configure nginx",
                "configurar firewall",
                "config database",
                "setup ssl certificate"
            ],
            confidence_threshold=0.8,
            requires_confirmation=True,
            risk_level="high",
            estimated_time=300,
            model_preference="deepseek"
        )
        
        intents["remove_software"] = Intent(
            name="remove_software",
            category="system_administration",
            description="Remover software ou ficheiros",
            keywords=["remove", "remover", "delete", "deletar", "uninstall"],
            patterns=[
                r"remove\s+\w+",
                r"remover\s+\w+",
                r"delete\s+\w+",
                r"uninstall\s+\w+"
            ],
            examples=[
                "remove old packages",
                "remover ficheiros temporários",
                "delete logs",
                "uninstall apache"
            ],
            confidence_threshold=0.9,
            requires_confirmation=True,
            risk_level="high",
            estimated_time=120,
            model_preference="wizardcoder"
        )
        
        intents["list_information"] = Intent(
            name="list_information",
            category="information_retrieval",
            description="Listar informações do sistema",
            keywords=["list", "listar", "show", "mostrar", "display", "ls"],
            patterns=[
                r"list\s+\w+",
                r"listar\s+\w+",
                r"show\s+\w+",
                r"mostrar\s+\w+"
            ],
            examples=[
                "list running processes",
                "listar serviços ativos",
                "show disk usage",
                "mostrar utilizadores"
            ],
            confidence_threshold=0.6,
            requires_confirmation=False,
            risk_level="low",
            estimated_time=5,
            model_preference="qwen"
        )
        
        intents["create_resource"] = Intent(
            name="create_resource",
            category="resource_management",
            description="Criar recursos ou estruturas",
            keywords=["create", "criar", "make", "fazer", "mkdir", "new"],
            patterns=[
                r"create\s+\w+",
                r"criar\s+\w+",
                r"make\s+\w+",
                r"mkdir\s+\w+"
            ],
            examples=[
                "create directory",
                "criar utilizador",
                "make backup folder",
                "new database"
            ],
            confidence_threshold=0.7,
            requires_confirmation=False,
            risk_level="medium",
            estimated_time=30,
            model_preference="deepseek"
        )
        
        intents["troubleshoot_issue"] = Intent(
            name="troubleshoot_issue",
            category="problem_solving",
            description="Resolver problemas ou diagnosticar issues",
            keywords=["fix", "resolver", "troubleshoot", "debug", "diagnose", "problem"],
            patterns=[
                r"fix\s+\w+",
                r"resolver\s+\w+",
                r"troubleshoot\s+\w+",
                r"debug\s+\w+"
            ],
            examples=[
                "fix connection issue",
                "resolver problema de memória",
                "troubleshoot network",
                "debug application error"
            ],
            confidence_threshold=0.7,
            requires_confirmation=False,
            risk_level="medium",
            estimated_time=180,
            model_preference="deepseek"
        )
        
        return intents
    
    def _setup_database(self):
        """Configurar base de dados"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                # Tabela de classificações
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS intent_classifications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_input TEXT NOT NULL,
                        predicted_intent TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        actual_intent TEXT,
                        feedback_score INTEGER,
                        user_id TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabela de dados de treino
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS training_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL,
                        intent TEXT NOT NULL,
                        source TEXT DEFAULT 'manual',
                        confidence REAL DEFAULT 1.0,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabela de performance dos modelos
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        accuracy REAL,
                        precision_macro REAL,
                        recall_macro REAL,
                        f1_macro REAL,
                        training_samples INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabela de padrões de utilizador
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS user_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        intent TEXT NOT NULL,
                        frequency INTEGER DEFAULT 1,
                        avg_confidence REAL,
                        last_used DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Erro ao configurar base de dados: {e}")
    
    def _load_training_data(self):
        """Carregar dados de treino da base de dados e exemplos predefinidos"""
        self.training_data = []
        
        # Adicionar exemplos predefinidos
        for intent_name, intent in self.intents.items():
            for example in intent.examples:
                self.training_data.append((example, intent_name))
        
        # Carregar dados adicionais da base de dados
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT text, intent FROM training_data")
                db_data = cursor.fetchall()
                self.training_data.extend(db_data)
                
        except Exception as e:
            logger.warning(f"Erro ao carregar dados de treino da DB: {e}")
        
        logger.info(f"Carregados {len(self.training_data)} exemplos de treino")
    
    def _train_models(self):
        """Treinar múltiplos modelos de classificação"""
        if len(self.training_data) < 10:
            logger.warning("Poucos dados de treino disponíveis")
            return
        
        # Preparar dados
        texts = [item[0] for item in self.training_data]
        labels = [item[1] for item in self.training_data]
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Modelos para treinar
        models_config = {
            'naive_bayes': {
                'pipeline': Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                    ('classifier', MultinomialNB(alpha=0.1))
                ]),
                'params': {}
            },
            'svm': {
                'pipeline': Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                    ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
                ]),
                'params': {}
            },
            'random_forest': {
                'pipeline': Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
                ]),
                'params': {}
            }
        }
        
        best_model = None
        best_score = 0
        
        for model_name, config in models_config.items():
            try:
                logger.info(f"Treinando modelo: {model_name}")
                
                # Treinar modelo
                pipeline = config['pipeline']
                pipeline.fit(X_train, y_train)
                
                # Avaliar com validação cruzada
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
                avg_score = cv_scores.mean()
                
                # Testar no conjunto de teste
                test_score = pipeline.score(X_test, y_test)
                
                # Salvar modelo
                model_path = f"/opt/alhica-ai/models/intent_classifier_{model_name}.joblib"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                joblib.dump(pipeline, model_path)
                
                self.models[model_name] = {
                    'pipeline': pipeline,
                    'cv_score': avg_score,
                    'test_score': test_score,
                    'path': model_path
                }
                
                # Salvar performance na DB
                self._save_model_performance(model_name, test_score, len(X_train))
                
                logger.info(f"Modelo {model_name}: CV={avg_score:.3f}, Test={test_score:.3f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model_name
                    
            except Exception as e:
                logger.error(f"Erro ao treinar modelo {model_name}: {e}")
        
        if best_model:
            self.best_model = best_model
            logger.info(f"Melhor modelo: {best_model} (score: {best_score:.3f})")
        else:
            logger.error("Nenhum modelo foi treinado com sucesso")
    
    def _save_model_performance(self, model_name: str, accuracy: float, training_samples: int):
        """Salvar performance do modelo na base de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO model_performance 
                    (model_name, accuracy, training_samples)
                    VALUES (?, ?, ?)
                ''', (model_name, accuracy, training_samples))
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao salvar performance do modelo: {e}")
    
    def classify_intent(self, text: str, user_id: str = None) -> IntentPrediction:
        """Classificar intenção do texto"""
        logger.info(f"Classificando intenção: {text}")
        
        # Usar o melhor modelo disponível
        if not hasattr(self, 'best_model') or self.best_model not in self.models:
            return self._fallback_classification(text)
        
        try:
            model = self.models[self.best_model]['pipeline']
            
            # Predizer intenção
            predicted_proba = model.predict_proba([text])[0]
            predicted_intent = model.predict([text])[0]
            
            # Obter todas as probabilidades
            classes = model.classes_
            intent_probabilities = list(zip(classes, predicted_proba))
            intent_probabilities.sort(key=lambda x: x[1], reverse=True)
            
            confidence = intent_probabilities[0][1]
            alternatives = intent_probabilities[1:4]  # Top 3 alternativas
            
            # Obter informações da intenção
            intent_info = self.intents.get(predicted_intent)
            if not intent_info:
                return self._fallback_classification(text)
            
            # Recomendar modelo de IA
            model_recommendation = self._recommend_ai_model(predicted_intent, text)
            
            # Avaliar risco
            risk_assessment = self._assess_risk(predicted_intent, text, confidence)
            
            # Criar plano de execução
            execution_plan = self._create_execution_plan(predicted_intent, text, intent_info)
            
            # Salvar classificação
            self._save_classification(text, predicted_intent, confidence, user_id)
            
            # Atualizar padrões do utilizador
            if user_id:
                self._update_user_patterns(user_id, predicted_intent, confidence)
            
            prediction = IntentPrediction(
                intent_name=predicted_intent,
                confidence=confidence,
                category=intent_info.category,
                model_recommendation=model_recommendation,
                risk_assessment=risk_assessment,
                execution_plan=execution_plan,
                alternatives=alternatives
            )
            
            logger.info(f"Intenção classificada: {predicted_intent} (confiança: {confidence:.3f})")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Erro na classificação: {e}")
            return self._fallback_classification(text)
    
    def _fallback_classification(self, text: str) -> IntentPrediction:
        """Classificação de fallback baseada em palavras-chave"""
        logger.info("Usando classificação de fallback")
        
        text_lower = text.lower()
        best_match = None
        best_score = 0
        
        # Procurar por palavras-chave
        for intent_name, intent in self.intents.items():
            score = 0
            for keyword in intent.keywords:
                if keyword.lower() in text_lower:
                    score += 1
            
            # Normalizar score
            if intent.keywords:
                score = score / len(intent.keywords)
            
            if score > best_score:
                best_score = score
                best_match = intent_name
        
        if best_match and best_score > 0.3:
            intent_info = self.intents[best_match]
            confidence = min(best_score, 0.6)  # Máximo 0.6 para fallback
        else:
            # Default para informação geral
            best_match = "list_information"
            intent_info = self.intents[best_match]
            confidence = 0.3
        
        return IntentPrediction(
            intent_name=best_match,
            confidence=confidence,
            category=intent_info.category,
            model_recommendation=self._recommend_ai_model(best_match, text),
            risk_assessment=self._assess_risk(best_match, text, confidence),
            execution_plan=self._create_execution_plan(best_match, text, intent_info),
            alternatives=[]
        )
    
    def _recommend_ai_model(self, intent: str, text: str) -> str:
        """Recomendar modelo de IA baseado na intenção"""
        intent_info = self.intents.get(intent)
        
        # Preferência definida na intenção
        if intent_info and intent_info.model_preference:
            return intent_info.model_preference
        
        # Análise baseada no conteúdo
        text_lower = text.lower()
        
        # Palavras-chave para DeepSeek (código)
        code_keywords = ['install', 'configure', 'setup', 'script', 'code', 'programming']
        if any(keyword in text_lower for keyword in code_keywords):
            return "deepseek"
        
        # Palavras-chave para WizardCoder (automação)
        automation_keywords = ['automate', 'workflow', 'batch', 'schedule', 'deploy', 'backup']
        if any(keyword in text_lower for keyword in automation_keywords):
            return "wizardcoder"
        
        # Default para Qwen (geral)
        return "qwen"
    
    def _assess_risk(self, intent: str, text: str, confidence: float) -> Dict[str, Any]:
        """Avaliar risco da operação"""
        intent_info = self.intents.get(intent)
        base_risk = intent_info.risk_level if intent_info else "medium"
        
        risk_factors = []
        risk_score = {"low": 1, "medium": 2, "high": 3, "critical": 4}[base_risk]
        
        # Fatores que aumentam o risco
        high_risk_keywords = ['delete', 'remove', 'format', 'destroy', 'kill', 'stop']
        if any(keyword in text.lower() for keyword in high_risk_keywords):
            risk_score += 1
            risk_factors.append("Operação destrutiva detectada")
        
        # Baixa confiança aumenta risco
        if confidence < 0.5:
            risk_score += 1
            risk_factors.append("Baixa confiança na classificação")
        
        # Palavras de urgência
        urgency_keywords = ['urgent', 'emergency', 'critical', 'now', 'immediately']
        urgent = any(keyword in text.lower() for keyword in urgency_keywords)
        if urgent:
            risk_factors.append("Operação marcada como urgente")
        
        # Determinar nível final
        final_risk = ["low", "medium", "high", "critical"][min(risk_score - 1, 3)]
        
        return {
            "level": final_risk,
            "score": risk_score,
            "factors": risk_factors,
            "requires_confirmation": risk_score >= 3 or (intent_info and intent_info.requires_confirmation),
            "urgent": urgent
        }
    
    def _create_execution_plan(self, intent: str, text: str, intent_info: Intent) -> Dict[str, Any]:
        """Criar plano de execução"""
        return {
            "steps": [
                "Analisar comando com modelo de IA",
                "Gerar comandos específicos",
                "Validar segurança",
                "Executar com monitorização"
            ],
            "estimated_time": intent_info.estimated_time,
            "requires_confirmation": intent_info.requires_confirmation,
            "rollback_available": intent in ["install_software", "configure_service", "update_system"],
            "monitoring_required": intent_info.risk_level in ["high", "critical"],
            "parallel_execution": intent in ["check_status", "list_information", "backup_data"]
        }
    
    def _save_classification(self, text: str, intent: str, confidence: float, user_id: str = None):
        """Salvar classificação na base de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO intent_classifications 
                    (user_input, predicted_intent, confidence, user_id)
                    VALUES (?, ?, ?, ?)
                ''', (text, intent, confidence, user_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao salvar classificação: {e}")
    
    def _update_user_patterns(self, user_id: str, intent: str, confidence: float):
        """Atualizar padrões do utilizador"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Verificar se padrão já existe
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT frequency, avg_confidence FROM user_patterns 
                    WHERE user_id = ? AND intent = ?
                ''', (user_id, intent))
                
                result = cursor.fetchone()
                
                if result:
                    # Atualizar padrão existente
                    frequency, avg_confidence = result
                    new_frequency = frequency + 1
                    new_avg_confidence = (avg_confidence * frequency + confidence) / new_frequency
                    
                    conn.execute('''
                        UPDATE user_patterns 
                        SET frequency = ?, avg_confidence = ?, last_used = CURRENT_TIMESTAMP
                        WHERE user_id = ? AND intent = ?
                    ''', (new_frequency, new_avg_confidence, user_id, intent))
                else:
                    # Criar novo padrão
                    conn.execute('''
                        INSERT INTO user_patterns (user_id, intent, frequency, avg_confidence)
                        VALUES (?, ?, 1, ?)
                    ''', (user_id, intent, confidence))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Erro ao atualizar padrões do utilizador: {e}")
    
    def add_training_example(self, text: str, intent: str, source: str = "user_feedback"):
        """Adicionar exemplo de treino"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO training_data (text, intent, source)
                    VALUES (?, ?, ?)
                ''', (text, intent, source))
                conn.commit()
            
            # Adicionar aos dados de treino em memória
            self.training_data.append((text, intent))
            
            logger.info(f"Exemplo de treino adicionado: {text} -> {intent}")
            
        except Exception as e:
            logger.error(f"Erro ao adicionar exemplo de treino: {e}")
    
    def retrain_models(self):
        """Retreinar modelos com novos dados"""
        logger.info("Iniciando retreino dos modelos...")
        
        # Recarregar dados de treino
        self._load_training_data()
        
        # Retreinar modelos
        self._train_models()
        
        logger.info("Retreino concluído")
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Obter preferências do utilizador"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Intenções mais frequentes
                cursor.execute('''
                    SELECT intent, frequency, avg_confidence 
                    FROM user_patterns 
                    WHERE user_id = ? 
                    ORDER BY frequency DESC 
                    LIMIT 10
                ''', (user_id,))
                
                frequent_intents = [
                    {"intent": row[0], "frequency": row[1], "avg_confidence": row[2]}
                    for row in cursor.fetchall()
                ]
                
                # Estatísticas gerais
                cursor.execute('''
                    SELECT COUNT(*), AVG(confidence) 
                    FROM intent_classifications 
                    WHERE user_id = ?
                ''', (user_id,))
                
                total_classifications, avg_confidence = cursor.fetchone()
                
                return {
                    "user_id": user_id,
                    "total_classifications": total_classifications or 0,
                    "average_confidence": avg_confidence or 0,
                    "frequent_intents": frequent_intents,
                    "personalization_available": len(frequent_intents) >= 5
                }
                
        except Exception as e:
            logger.error(f"Erro ao obter preferências do utilizador: {e}")
            return {}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Obter estatísticas do sistema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total de classificações
                cursor.execute("SELECT COUNT(*) FROM intent_classifications")
                total_classifications = cursor.fetchone()[0]
                
                # Confiança média
                cursor.execute("SELECT AVG(confidence) FROM intent_classifications")
                avg_confidence = cursor.fetchone()[0] or 0
                
                # Intenções mais comuns
                cursor.execute('''
                    SELECT predicted_intent, COUNT(*) as count 
                    FROM intent_classifications 
                    GROUP BY predicted_intent 
                    ORDER BY count DESC 
                    LIMIT 10
                ''')
                top_intents = [{"intent": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                # Performance dos modelos
                cursor.execute('''
                    SELECT model_name, accuracy, training_samples 
                    FROM model_performance 
                    ORDER BY timestamp DESC 
                    LIMIT 5
                ''')
                model_performance = [
                    {"model": row[0], "accuracy": row[1], "samples": row[2]}
                    for row in cursor.fetchall()
                ]
                
                # Classificações por dia (últimos 7 dias)
                cursor.execute('''
                    SELECT DATE(timestamp) as date, COUNT(*) as count 
                    FROM intent_classifications 
                    WHERE timestamp >= datetime('now', '-7 days')
                    GROUP BY DATE(timestamp) 
                    ORDER BY date
                ''')
                daily_stats = [{"date": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                return {
                    "total_classifications": total_classifications,
                    "average_confidence": round(avg_confidence, 3),
                    "top_intents": top_intents,
                    "model_performance": model_performance,
                    "daily_classifications": daily_stats,
                    "available_intents": len(self.intents),
                    "training_examples": len(self.training_data)
                }
                
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            return {}

def main():
    """Função principal para teste"""
    classifier = IntentClassificationSystem()
    
    # Comandos de teste
    test_commands = [
        "install nginx on server-01",
        "check the status of mysql database",
        "restart apache service",
        "backup all configuration files",
        "update system packages",
        "configure firewall rules",
        "list running processes",
        "remove old log files",
        "create new user account",
        "fix network connectivity issue"
    ]
    
    print("🧪 Testando sistema de classificação de intenções...\n")
    
    for command in test_commands:
        print(f"Comando: {command}")
        prediction = classifier.classify_intent(command, user_id="test_user")
        
        print(f"  Intenção: {prediction.intent_name}")
        print(f"  Categoria: {prediction.category}")
        print(f"  Confiança: {prediction.confidence:.3f}")
        print(f"  Modelo recomendado: {prediction.model_recommendation}")
        print(f"  Nível de risco: {prediction.risk_assessment['level']}")
        print(f"  Confirmação necessária: {prediction.risk_assessment['requires_confirmation']}")
        print(f"  Tempo estimado: {prediction.execution_plan['estimated_time']}s")
        print()
    
    # Mostrar estatísticas
    stats = classifier.get_system_statistics()
    print("📊 Estatísticas do Sistema:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()

