#!/usr/bin/env python3
"""
Alhica AI - Parser de Linguagem Natural Avançado
Sistema inteligente para conversão de pedidos em linguagem natural para comandos executáveis

Copyright (c) 2024 Alhica AI Team
"""

import os
import re
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import difflib

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/alhica-ai/nlp_parser.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ParsedCommand:
    """Comando parseado da linguagem natural"""
    original_text: str
    action: str
    target: str
    parameters: Dict[str, Any]
    confidence: float
    server_hint: Optional[str] = None
    urgency: str = "normal"  # low, normal, high, critical
    estimated_time: Optional[int] = None  # segundos
    risk_level: str = "low"  # low, medium, high, critical
    requires_confirmation: bool = False

@dataclass
class CommandPattern:
    """Padrão de comando para reconhecimento"""
    pattern: str
    action: str
    parameters: List[str]
    examples: List[str]
    risk_level: str = "low"
    requires_confirmation: bool = False

class NaturalLanguageParser:
    """Parser avançado de linguagem natural para comandos de sistema"""
    
    def __init__(self, db_path: str = "/opt/alhica-ai/data/nlp_parser.db"):
        self.db_path = db_path
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Inicializar NLTK
        self._setup_nltk()
        
        # Carregar modelo spaCy se disponível
        self.nlp = self._load_spacy_model()
        
        # Padrões de comando
        self.command_patterns = self._load_command_patterns()
        
        # Dicionários de sinónimos
        self.action_synonyms = self._load_action_synonyms()
        self.target_synonyms = self._load_target_synonyms()
        
        # Base de dados para aprendizagem
        self._setup_database()
        
        logger.info("🧠 Parser de linguagem natural inicializado")
    
    def _setup_nltk(self):
        """Configurar NLTK com downloads necessários"""
        try:
            nltk_data_path = Path.home() / 'nltk_data'
            nltk_data_path.mkdir(exist_ok=True)
            
            # Downloads necessários
            downloads = [
                'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                'vader_lexicon', 'omw-1.4'
            ]
            
            for item in downloads:
                try:
                    nltk.download(item, quiet=True)
                except Exception as e:
                    logger.warning(f"Erro ao baixar {item}: {e}")
            
            # Configurar stopwords
            try:
                self.stop_words = set(stopwords.words('english'))
                self.stop_words.update(stopwords.words('portuguese'))
            except:
                self.stop_words = set()
                
        except Exception as e:
            logger.warning(f"Erro na configuração do NLTK: {e}")
    
    def _load_spacy_model(self):
        """Carregar modelo spaCy se disponível"""
        try:
            import spacy
            # Tentar carregar modelo em português
            try:
                return spacy.load("pt_core_news_sm")
            except:
                # Fallback para inglês
                try:
                    return spacy.load("en_core_web_sm")
                except:
                    logger.warning("Nenhum modelo spaCy encontrado")
                    return None
        except ImportError:
            logger.warning("spaCy não disponível")
            return None
    
    def _load_command_patterns(self) -> List[CommandPattern]:
        """Carregar padrões de comando predefinidos"""
        return [
            # Instalação de software
            CommandPattern(
                pattern=r"install(?:ar)?\s+(.+?)(?:\s+(?:on|no|em)\s+(.+))?",
                action="install",
                parameters=["software", "server"],
                examples=[
                    "instalar nginx",
                    "install docker on server-01",
                    "instalar wordpress no servidor web"
                ],
                risk_level="medium",
                requires_confirmation=True
            ),
            
            # Verificação de status
            CommandPattern(
                pattern=r"(?:check|verificar|status|estado)(?:\s+(?:the|o|do))?\s+(.+?)(?:\s+(?:on|no|em)\s+(.+))?",
                action="check_status",
                parameters=["service", "server"],
                examples=[
                    "verificar nginx",
                    "check disk space on server-01",
                    "status do mysql"
                ],
                risk_level="low"
            ),
            
            # Reiniciar serviços
            CommandPattern(
                pattern=r"(?:restart|reiniciar|reboot)\s+(.+?)(?:\s+(?:on|no|em)\s+(.+))?",
                action="restart",
                parameters=["service", "server"],
                examples=[
                    "restart nginx",
                    "reiniciar apache",
                    "reboot server-01"
                ],
                risk_level="high",
                requires_confirmation=True
            ),
            
            # Atualização de sistema
            CommandPattern(
                pattern=r"(?:update|atualizar|upgrade)(?:\s+(.+?))?(?:\s+(?:on|no|em)\s+(.+))?",
                action="update",
                parameters=["target", "server"],
                examples=[
                    "update system",
                    "atualizar pacotes",
                    "upgrade all packages on server-01"
                ],
                risk_level="medium",
                requires_confirmation=True
            ),
            
            # Gestão de ficheiros
            CommandPattern(
                pattern=r"(?:backup|copy|copiar|move|mover)\s+(.+?)(?:\s+(?:to|para|from|de)\s+(.+?))?(?:\s+(?:on|no|em)\s+(.+))?",
                action="file_operation",
                parameters=["source", "destination", "server"],
                examples=[
                    "backup /etc/nginx/nginx.conf",
                    "copy file.txt to /backup/",
                    "mover logs para /archive/"
                ],
                risk_level="medium"
            ),
            
            # Monitorização
            CommandPattern(
                pattern=r"(?:monitor|monitorizar|watch|observar)\s+(.+?)(?:\s+(?:on|no|em)\s+(.+))?",
                action="monitor",
                parameters=["target", "server"],
                examples=[
                    "monitor cpu usage",
                    "watch memory on server-01",
                    "monitorizar disco"
                ],
                risk_level="low"
            ),
            
            # Listagem
            CommandPattern(
                pattern=r"(?:list|listar|show|mostrar|ls)\s+(.+?)(?:\s+(?:on|no|em)\s+(.+))?",
                action="list",
                parameters=["target", "server"],
                examples=[
                    "list running processes",
                    "show disk usage",
                    "listar serviços ativos"
                ],
                risk_level="low"
            ),
            
            # Configuração
            CommandPattern(
                pattern=r"(?:configure|configurar|config|setup)\s+(.+?)(?:\s+(?:on|no|em)\s+(.+))?",
                action="configure",
                parameters=["service", "server"],
                examples=[
                    "configure nginx",
                    "setup database",
                    "configurar firewall"
                ],
                risk_level="high",
                requires_confirmation=True
            ),
            
            # Remoção
            CommandPattern(
                pattern=r"(?:remove|remover|delete|deletar|uninstall|desinstalar)\s+(.+?)(?:\s+(?:from|de|on|no|em)\s+(.+))?",
                action="remove",
                parameters=["target", "server"],
                examples=[
                    "remove old logs",
                    "uninstall apache",
                    "deletar ficheiros temporários"
                ],
                risk_level="high",
                requires_confirmation=True
            ),
            
            # Criação
            CommandPattern(
                pattern=r"(?:create|criar|make|fazer|mkdir)\s+(.+?)(?:\s+(?:on|no|em)\s+(.+))?",
                action="create",
                parameters=["target", "server"],
                examples=[
                    "create directory /backup",
                    "make new user",
                    "criar base de dados"
                ],
                risk_level="medium"
            )
        ]
    
    def _load_action_synonyms(self) -> Dict[str, List[str]]:
        """Carregar sinónimos para ações"""
        return {
            "install": ["instalar", "install", "setup", "configurar", "add", "adicionar"],
            "remove": ["remover", "remove", "delete", "deletar", "uninstall", "desinstalar", "rm"],
            "restart": ["reiniciar", "restart", "reboot", "reload", "recarregar"],
            "check": ["verificar", "check", "status", "estado", "test", "testar"],
            "update": ["atualizar", "update", "upgrade", "refresh", "refrescar"],
            "backup": ["backup", "copy", "copiar", "save", "guardar"],
            "monitor": ["monitorizar", "monitor", "watch", "observar", "track"],
            "list": ["listar", "list", "show", "mostrar", "display", "exibir", "ls"],
            "configure": ["configurar", "configure", "config", "setup", "set"],
            "create": ["criar", "create", "make", "fazer", "mkdir", "new", "novo"]
        }
    
    def _load_target_synonyms(self) -> Dict[str, List[str]]:
        """Carregar sinónimos para alvos/serviços"""
        return {
            "nginx": ["nginx", "web server", "servidor web"],
            "apache": ["apache", "apache2", "httpd"],
            "mysql": ["mysql", "mariadb", "database", "base de dados", "db"],
            "postgresql": ["postgresql", "postgres", "pgsql"],
            "docker": ["docker", "container", "contentor"],
            "system": ["system", "sistema", "os", "operating system"],
            "disk": ["disk", "disco", "storage", "armazenamento", "hdd", "ssd"],
            "memory": ["memory", "memória", "ram"],
            "cpu": ["cpu", "processor", "processador"],
            "network": ["network", "rede", "net", "interface"],
            "firewall": ["firewall", "iptables", "ufw"],
            "logs": ["logs", "log files", "ficheiros de log", "registos"]
        }
    
    def _setup_database(self):
        """Configurar base de dados para aprendizagem"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS parsed_commands (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        original_text TEXT NOT NULL,
                        parsed_action TEXT NOT NULL,
                        parsed_target TEXT,
                        parameters TEXT,
                        confidence REAL,
                        success BOOLEAN,
                        user_feedback TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS command_corrections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        original_command TEXT NOT NULL,
                        corrected_action TEXT NOT NULL,
                        corrected_target TEXT,
                        corrected_parameters TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS user_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        pattern TEXT NOT NULL,
                        action TEXT NOT NULL,
                        frequency INTEGER DEFAULT 1,
                        last_used DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Erro ao configurar base de dados: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """Pré-processar texto para análise"""
        # Converter para minúsculas
        text = text.lower().strip()
        
        # Remover pontuação desnecessária
        text = re.sub(r'[^\w\s\-\./]', ' ', text)
        
        # Normalizar espaços
        text = re.sub(r'\s+', ' ', text)
        
        # Expandir contrações comuns
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extrair entidades do texto"""
        entities = {
            'servers': [],
            'services': [],
            'files': [],
            'numbers': [],
            'ips': []
        }
        
        # Extrair IPs
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        entities['ips'] = re.findall(ip_pattern, text)
        
        # Extrair caminhos de ficheiros
        file_pattern = r'(?:/[^\s]+|[^\s]+\.[a-zA-Z]{2,4})'
        entities['files'] = re.findall(file_pattern, text)
        
        # Extrair números
        number_pattern = r'\b\d+\b'
        entities['numbers'] = re.findall(number_pattern, text)
        
        # Extrair nomes de servidores (padrões comuns)
        server_patterns = [
            r'\bserver[-_]?\d+\b',
            r'\b\w+[-_]server\b',
            r'\b\w+[-_]0\d+\b'
        ]
        
        for pattern in server_patterns:
            entities['servers'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Usar spaCy se disponível
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT']:
                    entities['services'].append(ent.text)
        
        return entities
    
    def match_command_pattern(self, text: str) -> Tuple[Optional[CommandPattern], Optional[re.Match]]:
        """Encontrar padrão de comando que corresponde ao texto"""
        best_match = None
        best_pattern = None
        best_score = 0
        
        for pattern in self.command_patterns:
            match = re.search(pattern.pattern, text, re.IGNORECASE)
            if match:
                # Calcular score baseado na qualidade do match
                score = len(match.group(0)) / len(text)
                
                # Bonus se o padrão tem exemplos similares
                for example in pattern.examples:
                    similarity = difflib.SequenceMatcher(None, text, example).ratio()
                    if similarity > 0.6:
                        score += similarity * 0.5
                
                if score > best_score:
                    best_score = score
                    best_match = match
                    best_pattern = pattern
        
        return best_pattern, best_match
    
    def resolve_synonyms(self, word: str, synonym_dict: Dict[str, List[str]]) -> str:
        """Resolver sinónimos para forma canónica"""
        word_lower = word.lower()
        
        for canonical, synonyms in synonym_dict.items():
            if word_lower in [s.lower() for s in synonyms]:
                return canonical
        
        return word_lower
    
    def estimate_risk_and_time(self, action: str, target: str, parameters: Dict) -> Tuple[str, int, bool]:
        """Estimar risco, tempo e necessidade de confirmação"""
        risk_levels = {
            "install": ("medium", 300, True),
            "remove": ("high", 120, True),
            "restart": ("high", 60, True),
            "update": ("medium", 600, True),
            "configure": ("high", 180, True),
            "check_status": ("low", 10, False),
            "list": ("low", 5, False),
            "monitor": ("low", 0, False),
            "backup": ("medium", 120, False),
            "create": ("medium", 30, False)
        }
        
        base_risk, base_time, base_confirm = risk_levels.get(action, ("medium", 60, False))
        
        # Ajustar baseado no alvo
        high_risk_targets = ["system", "database", "firewall", "kernel"]
        if any(target_word in target.lower() for target_word in high_risk_targets):
            if base_risk == "low":
                base_risk = "medium"
            elif base_risk == "medium":
                base_risk = "high"
            base_confirm = True
        
        # Ajustar tempo baseado em parâmetros
        if "all" in str(parameters).lower():
            base_time *= 3
        
        return base_risk, base_time, base_confirm
    
    def parse_command(self, text: str, user_id: str = None) -> ParsedCommand:
        """Analisar comando em linguagem natural"""
        logger.info(f"Analisando comando: {text}")
        
        # Pré-processar texto
        processed_text = self.preprocess_text(text)
        
        # Extrair entidades
        entities = self.extract_entities(processed_text)
        
        # Encontrar padrão correspondente
        pattern, match = self.match_command_pattern(processed_text)
        
        if not pattern or not match:
            # Tentar análise baseada em palavras-chave
            return self._fallback_parse(text, processed_text, entities)
        
        # Extrair parâmetros do match
        groups = match.groups()
        parameters = {}
        
        for i, param_name in enumerate(pattern.parameters):
            if i < len(groups) and groups[i]:
                param_value = groups[i].strip()
                
                # Resolver sinónimos
                if param_name == "action":
                    param_value = self.resolve_synonyms(param_value, self.action_synonyms)
                elif param_name in ["service", "target", "software"]:
                    param_value = self.resolve_synonyms(param_value, self.target_synonyms)
                
                parameters[param_name] = param_value
        
        # Adicionar entidades extraídas
        parameters.update(entities)
        
        # Determinar alvo principal
        target = parameters.get("service") or parameters.get("target") or parameters.get("software") or "unknown"
        
        # Estimar risco e tempo
        risk_level, estimated_time, requires_confirmation = self.estimate_risk_and_time(
            pattern.action, target, parameters
        )
        
        # Calcular confiança
        confidence = self._calculate_confidence(text, pattern, match, entities)
        
        # Detectar servidor
        server_hint = self._detect_server_hint(text, entities)
        
        # Determinar urgência
        urgency = self._detect_urgency(text)
        
        parsed_command = ParsedCommand(
            original_text=text,
            action=pattern.action,
            target=target,
            parameters=parameters,
            confidence=confidence,
            server_hint=server_hint,
            urgency=urgency,
            estimated_time=estimated_time,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation or pattern.requires_confirmation
        )
        
        # Salvar na base de dados para aprendizagem
        self._save_parsed_command(parsed_command, user_id)
        
        logger.info(f"Comando analisado: {pattern.action} -> {target} (confiança: {confidence:.2f})")
        
        return parsed_command
    
    def _fallback_parse(self, original_text: str, processed_text: str, entities: Dict) -> ParsedCommand:
        """Análise de fallback baseada em palavras-chave"""
        logger.info("Usando análise de fallback")
        
        # Tokenizar
        tokens = word_tokenize(processed_text)
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Procurar ação
        action = "unknown"
        for token in tokens:
            resolved_action = self.resolve_synonyms(token, self.action_synonyms)
            if resolved_action != token:
                action = resolved_action
                break
        
        # Procurar alvo
        target = "unknown"
        for token in tokens:
            resolved_target = self.resolve_synonyms(token, self.target_synonyms)
            if resolved_target != token:
                target = resolved_target
                break
        
        # Se não encontrou alvo específico, usar entidades
        if target == "unknown":
            if entities.get('services'):
                target = entities['services'][0]
            elif entities.get('files'):
                target = entities['files'][0]
        
        # Parâmetros básicos
        parameters = entities.copy()
        parameters['tokens'] = tokens
        
        # Estimativas conservadoras para fallback
        risk_level = "medium"
        estimated_time = 60
        requires_confirmation = True
        confidence = 0.3  # Baixa confiança para fallback
        
        return ParsedCommand(
            original_text=original_text,
            action=action,
            target=target,
            parameters=parameters,
            confidence=confidence,
            server_hint=self._detect_server_hint(original_text, entities),
            urgency=self._detect_urgency(original_text),
            estimated_time=estimated_time,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation
        )
    
    def _calculate_confidence(self, text: str, pattern: CommandPattern, match: re.Match, entities: Dict) -> float:
        """Calcular confiança da análise"""
        confidence = 0.5  # Base
        
        # Bonus por match de padrão
        match_quality = len(match.group(0)) / len(text)
        confidence += match_quality * 0.3
        
        # Bonus por entidades encontradas
        entity_count = sum(len(v) for v in entities.values() if isinstance(v, list))
        confidence += min(entity_count * 0.05, 0.2)
        
        # Bonus por similaridade com exemplos
        best_similarity = 0
        for example in pattern.examples:
            similarity = difflib.SequenceMatcher(None, text.lower(), example.lower()).ratio()
            best_similarity = max(best_similarity, similarity)
        
        confidence += best_similarity * 0.2
        
        # Penalidade por ambiguidade
        if "unknown" in [match.group(i) for i in range(1, len(match.groups()) + 1) if match.group(i)]:
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def _detect_server_hint(self, text: str, entities: Dict) -> Optional[str]:
        """Detectar dica de servidor no texto"""
        # Procurar por IPs
        if entities.get('ips'):
            return entities['ips'][0]
        
        # Procurar por nomes de servidor
        if entities.get('servers'):
            return entities['servers'][0]
        
        # Procurar por padrões de servidor no texto
        server_patterns = [
            r'\bon\s+(\w+[-_]?\w*)',
            r'\bem\s+(\w+[-_]?\w*)',
            r'\bno\s+(\w+[-_]?\w*)',
            r'\bserver\s+(\w+)',
            r'\bservidor\s+(\w+)'
        ]
        
        for pattern in server_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _detect_urgency(self, text: str) -> str:
        """Detectar urgência do comando"""
        urgency_keywords = {
            "critical": ["urgent", "urgente", "critical", "crítico", "emergency", "emergência", "now", "agora", "immediately", "imediatamente"],
            "high": ["quick", "rápido", "fast", "asap", "soon", "breve", "priority", "prioridade"],
            "low": ["later", "depois", "when possible", "quando possível", "eventually", "eventualmente"]
        }
        
        text_lower = text.lower()
        
        for level, keywords in urgency_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return level
        
        return "normal"
    
    def _save_parsed_command(self, command: ParsedCommand, user_id: str = None):
        """Salvar comando analisado na base de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO parsed_commands 
                    (original_text, parsed_action, parsed_target, parameters, confidence)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    command.original_text,
                    command.action,
                    command.target,
                    json.dumps(command.parameters),
                    command.confidence
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao salvar comando analisado: {e}")
    
    def learn_from_feedback(self, original_text: str, correct_action: str, 
                           correct_target: str, correct_parameters: Dict = None):
        """Aprender com feedback do utilizador"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO command_corrections 
                    (original_command, corrected_action, corrected_target, corrected_parameters)
                    VALUES (?, ?, ?, ?)
                ''', (
                    original_text,
                    correct_action,
                    correct_target,
                    json.dumps(correct_parameters or {})
                ))
                conn.commit()
                
            logger.info(f"Feedback registado para: {original_text}")
            
        except Exception as e:
            logger.error(f"Erro ao registar feedback: {e}")
    
    def get_parsing_statistics(self) -> Dict:
        """Obter estatísticas de análise"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total de comandos analisados
                cursor.execute("SELECT COUNT(*) FROM parsed_commands")
                total_commands = cursor.fetchone()[0]
                
                # Confiança média
                cursor.execute("SELECT AVG(confidence) FROM parsed_commands")
                avg_confidence = cursor.fetchone()[0] or 0
                
                # Ações mais comuns
                cursor.execute('''
                    SELECT parsed_action, COUNT(*) as count 
                    FROM parsed_commands 
                    GROUP BY parsed_action 
                    ORDER BY count DESC 
                    LIMIT 10
                ''')
                top_actions = cursor.fetchall()
                
                # Comandos com baixa confiança
                cursor.execute('''
                    SELECT COUNT(*) FROM parsed_commands 
                    WHERE confidence < 0.5
                ''')
                low_confidence_count = cursor.fetchone()[0]
                
                return {
                    'total_commands': total_commands,
                    'average_confidence': round(avg_confidence, 3),
                    'low_confidence_percentage': round((low_confidence_count / max(total_commands, 1)) * 100, 1),
                    'top_actions': [{'action': action, 'count': count} for action, count in top_actions]
                }
                
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            return {}
    
    def suggest_command_completion(self, partial_text: str, limit: int = 5) -> List[str]:
        """Sugerir completação de comando"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Procurar comandos similares
                cursor.execute('''
                    SELECT original_text, COUNT(*) as frequency
                    FROM parsed_commands 
                    WHERE original_text LIKE ? 
                    GROUP BY original_text
                    ORDER BY frequency DESC, original_text
                    LIMIT ?
                ''', (f"{partial_text}%", limit))
                
                suggestions = [row[0] for row in cursor.fetchall()]
                
                # Se não há sugestões diretas, procurar por similaridade
                if not suggestions:
                    cursor.execute('''
                        SELECT original_text 
                        FROM parsed_commands 
                        ORDER BY confidence DESC
                        LIMIT 50
                    ''')
                    
                    all_commands = [row[0] for row in cursor.fetchall()]
                    
                    # Usar difflib para encontrar comandos similares
                    similar_commands = difflib.get_close_matches(
                        partial_text, all_commands, n=limit, cutoff=0.3
                    )
                    
                    suggestions = similar_commands
                
                return suggestions
                
        except Exception as e:
            logger.error(f"Erro ao sugerir completação: {e}")
            return []

def main():
    """Função principal para teste"""
    parser = NaturalLanguageParser()
    
    # Comandos de teste
    test_commands = [
        "instalar nginx no servidor web-01",
        "verificar o status do mysql",
        "fazer backup do ficheiro /etc/nginx/nginx.conf",
        "reiniciar apache no servidor 192.168.1.100",
        "mostrar o uso de disco em todos os servidores",
        "atualizar o sistema urgentemente",
        "configurar firewall no servidor de produção"
    ]
    
    print("🧪 Testando parser de linguagem natural...\n")
    
    for command in test_commands:
        print(f"Comando: {command}")
        parsed = parser.parse_command(command)
        
        print(f"  Ação: {parsed.action}")
        print(f"  Alvo: {parsed.target}")
        print(f"  Confiança: {parsed.confidence:.2f}")
        print(f"  Risco: {parsed.risk_level}")
        print(f"  Servidor: {parsed.server_hint}")
        print(f"  Urgência: {parsed.urgency}")
        print(f"  Confirmação necessária: {parsed.requires_confirmation}")
        print()
    
    # Mostrar estatísticas
    stats = parser.get_parsing_statistics()
    print("📊 Estatísticas:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()

