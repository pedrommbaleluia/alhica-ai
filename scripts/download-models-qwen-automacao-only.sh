#!/bin/bash
#
# Script de Download Otimizado - Apenas Modelo Qwen-Automacao
# Versão: 3.2.0 Single-Model Optimized
# Data: Janeiro 2025
# Configuração: Qwen3-235B-Thinking (Automação) APENAS
# Redução: ~1.4TB de armazenamento (apenas 470GB necessários)
#

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configurações
INSTALL_DIR="/opt/alhica-ai"
MODELS_DIR="${INSTALL_DIR}/models"
VENV_DIR="${INSTALL_DIR}/venv"
LOG_DIR="${INSTALL_DIR}/logs"
LOG_FILE="${LOG_DIR}/download_automation_model.log"

# Configuração do modelo único
MODEL_NAME="automation"
MODEL_DISPLAY_NAME="Qwen3-235B-Thinking (Automação)"
MODEL_HF_NAME="Qwen/Qwen3-235B-A22B-Thinking-2507"
MODEL_LOCAL_DIR="${MODELS_DIR}/qwen-automacao"
EXPECTED_FILES=118
EXPECTED_SIZE="470GB"

# Função para log
log() {
    echo -e "${2:-$BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Função para verificar se é root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log "❌ Este script precisa ser executado como root" "$RED"
        exit 1
    fi
}

# Função para verificar conectividade
check_connectivity() {
    log "🌐 Verificando conectividade..." "$BLUE"
    
    # Testar conectividade geral
    if ! ping -c 3 8.8.8.8 &> /dev/null; then
        log "❌ Sem conectividade à internet" "$RED"
        return 1
    fi
    
    # Testar Hugging Face Hub
    if ! curl -s --connect-timeout 10 https://huggingface.co &> /dev/null; then
        log "❌ Não é possível conectar ao Hugging Face Hub" "$RED"
        return 1
    fi
    
    log "✅ Conectividade verificada" "$GREEN"
    return 0
}

# Função para verificar espaço em disco
check_disk_space() {
    log "💾 Verificando espaço em disco..." "$BLUE"
    
    # Verificar diretório de destino ou pai
    local check_dir="$MODELS_DIR"
    if [ ! -d "$check_dir" ]; then
        check_dir="$(dirname "$MODELS_DIR")"
        if [ ! -d "$check_dir" ]; then
            check_dir="/"
        fi
    fi
    
    local available_space=$(df "$check_dir" | tail -1 | awk '{print $4}')
    local available_gb=$((available_space / 1024 / 1024))
    local required_gb=500  # 470GB + margem de segurança
    
    log "📊 Espaço disponível: ${available_gb}GB" "$BLUE"
    log "📊 Espaço necessário: ${required_gb}GB" "$BLUE"
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        log "❌ Espaço insuficiente. Necessário: ${required_gb}GB, Disponível: ${available_gb}GB" "$RED"
        exit 1
    fi
    
    log "✅ Espaço em disco suficiente" "$GREEN"
}

# Função para configurar variáveis de ambiente
setup_environment() {
    log "🔧 Configurando variáveis de ambiente..." "$BLUE"
    
    # Configurações para otimização
    export HF_HUB_CACHE="${MODELS_DIR}/.cache"
    export TRANSFORMERS_CACHE="${MODELS_DIR}/.cache"
    export HF_HOME="${MODELS_DIR}/.cache"
    
    # Configurações de rede
    export HF_HUB_ENABLE_HF_TRANSFER=0  # Desabilitar por compatibilidade
    export CURL_CA_BUNDLE=""
    export REQUESTS_CA_BUNDLE=""
    
    # Timeouts
    export HF_HUB_DOWNLOAD_TIMEOUT=3600  # 1 hora por arquivo
    
    log "✅ Variáveis de ambiente configuradas" "$GREEN"
}

# Função para criar infraestrutura básica
create_basic_infrastructure() {
    log "🏗️ Criando infraestrutura básica..." "$BLUE"
    
    # Criar diretórios
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$MODELS_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "${MODELS_DIR}/.cache"
    
    # Instalar dependências básicas se necessário
    if ! command -v python3 &> /dev/null; then
        log "📦 Instalando Python3..." "$BLUE"
        apt update
        apt install -y python3 python3-pip python3-venv
    fi
    
    # Criar ambiente virtual se não existir
    if [ ! -d "$VENV_DIR" ]; then
        log "🐍 Criando ambiente virtual..." "$BLUE"
        python3 -m venv "$VENV_DIR"
    fi
    
    log "✅ Infraestrutura básica criada" "$GREEN"
}

# Função para configurar ambiente Python
setup_python_environment() {
    log "🐍 Configurando ambiente Python..." "$BLUE"
    
    # Verificar se ambiente virtual existe
    if [ ! -d "$VENV_DIR" ]; then
        log "🏗️ Ambiente virtual não encontrado - criando infraestrutura..." "$YELLOW"
        create_basic_infrastructure
    fi
    
    # Ativar ambiente virtual
    source "${VENV_DIR}/bin/activate"
    
    # Atualizar pip
    pip install --upgrade pip setuptools wheel
    
    # Instalar dependências essenciais
    log "📦 Instalando dependências Python essenciais..." "$BLUE"
    
    # Instalar transformers e dependências
    pip install transformers huggingface-hub accelerate torch
    
    # Tentar instalar hf_transfer (opcional)
    log "📦 Tentando instalar hf_transfer (opcional)..." "$YELLOW"
    if pip install hf_transfer; then
        log "✅ hf_transfer instalado - downloads serão mais rápidos!" "$GREEN"
        export HF_HUB_ENABLE_HF_TRANSFER=1
    else
        log "⚠️ hf_transfer não disponível - usando método padrão" "$YELLOW"
        export HF_HUB_ENABLE_HF_TRANSFER=0
    fi
    
    log "✅ Ambiente Python configurado" "$GREEN"
}

# Função para verificar integridade do modelo
check_model_integrity() {
    local model_dir="$1"
    local expected_files="$2"
    
    if [ ! -d "$model_dir" ]; then
        return 1
    fi
    
    # Verificar arquivos essenciais
    if [ ! -f "${model_dir}/config.json" ]; then
        return 1
    fi
    
    # Contar arquivos .safetensors
    local safetensors_count=$(find "$model_dir" -name "*.safetensors" 2>/dev/null | wc -l)
    
    if [ "$safetensors_count" -ge "$expected_files" ]; then
        return 0
    else
        return 1
    fi
}

# Função para download robusto do modelo
robust_download() {
    local model_name="$1"
    local model_hf_name="$2"
    local model_dir="$3"
    local expected_files="$4"
    local max_retries=5
    
    log "🎯 Processando modelo: $model_name ($model_hf_name)" "$CYAN"
    
    # Verificar se modelo já existe e está completo
    if check_model_integrity "$model_dir" "$expected_files"; then
        local existing_files=$(find "$model_dir" -name "*.safetensors" 2>/dev/null | wc -l)
        log "✅ Modelo $model_name já existe e está completo ($existing_files ficheiros)" "$GREEN"
        return 0
    fi
    
    # Criar diretório do modelo
    mkdir -p "$model_dir"
    
    # Tentar download com retry
    for attempt in $(seq 1 $max_retries); do
        log "📥 Baixando $model_name (tentativa $attempt)..." "$BLUE"
        log "🤖 Baixando modelo $model_name: $model_hf_name..." "$BLUE"
        log "📍 Destino: $model_dir" "$BLUE"
        
        # Verificar conectividade antes de cada tentativa
        if ! check_connectivity; then
            log "⚠️ Problema de conectividade. Aguardando 30 segundos..." "$YELLOW"
            sleep 30
            continue
        fi
        
        # Método 1: Tentar com huggingface-cli (mais robusto)
        if command -v huggingface-cli &> /dev/null; then
            log "📥 Tentando com huggingface-cli..." "$BLUE"
            
            if timeout 3600 huggingface-cli download \
                "$model_hf_name" \
                --local-dir "$model_dir" \
                --resume-download \
                --local-dir-use-symlinks False; then
                
                log "✅ Download concluído com huggingface-cli!" "$GREEN"
                
                # Verificar integridade
                if check_model_integrity "$model_dir" "$expected_files"; then
                    log "✅ Modelo $model_name verificado e completo" "$GREEN"
                    return 0
                else
                    log "⚠️ Modelo incompleto, tentando novamente..." "$YELLOW"
                    continue
                fi
            else
                log "❌ Falha com huggingface-cli (código: $?)" "$RED"
            fi
        fi
        
        # Método 2: Fallback para Python
        log "🔄 Tentando com Python..." "$BLUE"
        
        # Ativar ambiente virtual
        source "${VENV_DIR}/bin/activate"
        
        if timeout 3600 python -c "
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

try:
    print('📥 Baixando tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        '$model_hf_name',
        cache_dir='$model_dir',
        local_files_only=False,
        resume_download=True
    )
    tokenizer.save_pretrained('$model_dir')
    print('✅ Tokenizer baixado')
    
    print('📥 Baixando modelo...')
    snapshot_download(
        repo_id='$model_hf_name',
        local_dir='$model_dir',
        resume_download=True,
        local_files_only=False
    )
    print('✅ Modelo baixado')
    
except Exception as e:
    print(f'❌ Erro: {e}')
    exit(1)
"; then
            log "✅ Download concluído com Python!" "$GREEN"
            
            # Verificar integridade
            if check_model_integrity "$model_dir" "$expected_files"; then
                log "✅ Modelo $model_name verificado e completo" "$GREEN"
                return 0
            else
                log "⚠️ Modelo incompleto, tentando novamente..." "$YELLOW"
                continue
            fi
        else
            log "❌ Erro no download (código: $?)" "$RED"
        fi
        
        # Aguardar antes da próxima tentativa
        if [ $attempt -lt $max_retries ]; then
            local wait_time=$((attempt * 30))
            log "⏳ Aguardando ${wait_time}s antes da próxima tentativa..." "$YELLOW"
            sleep $wait_time
        fi
    done
    
    log "❌ Falha no download do modelo $model_name após $max_retries tentativas" "$RED"
    return 1
}

# Função para relatório final
generate_final_report() {
    log "📋 RELATÓRIO FINAL - MODELO ÚNICO QWEN-AUTOMACAO" "$CYAN"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "$CYAN"
    
    # Verificar modelo
    local status="❌ FALHOU"
    local files_count=0
    local size="0GB"
    
    if check_model_integrity "$MODEL_LOCAL_DIR" "$EXPECTED_FILES"; then
        status="✅ COMPLETO"
        files_count=$(find "$MODEL_LOCAL_DIR" -name "*.safetensors" 2>/dev/null | wc -l)
        size=$(du -sh "$MODEL_LOCAL_DIR" 2>/dev/null | cut -f1 || echo "N/A")
    fi
    
    log "🤖 AUTOMATION (Qwen3-235B-Thinking): $status" "$CYAN"
    log "📊 Ficheiros .safetensors: $files_count/$EXPECTED_FILES" "$BLUE"
    log "💾 Tamanho: $size" "$BLUE"
    
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "$CYAN"
    
    if [ "$status" = "✅ COMPLETO" ]; then
        log "🎉 MODELO BAIXADO COM SUCESSO!" "$GREEN"
        log "💾 Espaço total usado: $size" "$GREEN"
        log "🚀 Pode prosseguir com a instalação da Alhica AI!" "$GREEN"
        log "📋 Próximo passo: ./install-alhica-infrastructure-single-model.sh" "$BLUE"
        return 0
    else
        log "❌ DOWNLOAD INCOMPLETO" "$RED"
        log "🔄 Execute novamente este script para continuar" "$YELLOW"
        return 1
    fi
}

# Função principal
main() {
    log "🚀 Script de Download Otimizado - Modelo Único Qwen-Automacao" "$CYAN"
    log "📋 Configuração: Qwen3-235B-Thinking (Automação) APENAS" "$CYAN"
    log "💾 Redução de armazenamento: ~1.4TB economizados (apenas 470GB necessários)" "$CYAN"
    log "⚡ Tempo estimado: 4-6 horas (vs 12-18 horas para 3 modelos)" "$CYAN"
    
    # Verificações iniciais
    check_root
    check_connectivity
    check_disk_space
    setup_environment
    setup_python_environment
    
    # Download do modelo único
    log "🎯 Iniciando download do modelo Qwen-Automacao..." "$CYAN"
    
    if robust_download "$MODEL_NAME" "$MODEL_HF_NAME" "$MODEL_LOCAL_DIR" "$EXPECTED_FILES"; then
        log "✅ Download do modelo concluído com sucesso!" "$GREEN"
    else
        log "❌ Falha no download do modelo" "$RED"
        exit 1
    fi
    
    # Relatório final
    generate_final_report
}

# Executar função principal
main "$@"

