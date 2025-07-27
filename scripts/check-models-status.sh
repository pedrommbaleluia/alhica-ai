#!/bin/bash

# Script para verificar status dos modelos Qwen3
# Alhica AI - Verificação de Modelos

echo "🔍 VERIFICAÇÃO DE STATUS DOS MODELOS QWEN3"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Diretórios dos modelos
CONVERSATION_DIR="/opt/alhica-ai/models/qwen-conversacao"
CODE_DIR="/opt/alhica-ai/models/qwen-codigo"
AUTOMATION_DIR="/opt/alhica-ai/models/qwen-automacao"

# Função para verificar modelo
check_model() {
    local model_name="$1"
    local model_dir="$2"
    local expected_files="$3"
    
    echo "🎯 Verificando modelo: $model_name"
    echo "📍 Diretório: $model_dir"
    
    if [ ! -d "$model_dir" ]; then
        echo "❌ Diretório não encontrado"
        echo "📊 Status: NÃO INICIADO"
        echo ""
        return 1
    fi
    
    # Contar ficheiros .safetensors
    local safetensors_count=$(find "$model_dir" -name "*.safetensors" -type f | wc -l)
    
    # Verificar ficheiros essenciais
    local config_exists=false
    local tokenizer_exists=false
    
    [ -f "$model_dir/config.json" ] && config_exists=true
    [ -f "$model_dir/tokenizer.json" ] || [ -f "$model_dir/merges.txt" ] && tokenizer_exists=true
    
    echo "📊 Ficheiros .safetensors: $safetensors_count/$expected_files"
    echo "📋 Config.json: $([ "$config_exists" = true ] && echo "✅" || echo "❌")"
    echo "🔤 Tokenizer: $([ "$tokenizer_exists" = true ] && echo "✅" || echo "❌")"
    
    # Calcular tamanho total
    local total_size=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
    echo "💾 Tamanho total: $total_size"
    
    # Determinar status
    if [ "$safetensors_count" -eq "$expected_files" ] && [ "$config_exists" = true ] && [ "$tokenizer_exists" = true ]; then
        echo "🎉 Status: ✅ COMPLETO"
    elif [ "$safetensors_count" -gt 0 ]; then
        local percentage=$((safetensors_count * 100 / expected_files))
        echo "🔄 Status: 📥 EM PROGRESSO ($percentage%)"
    else
        echo "❌ Status: NÃO INICIADO"
    fi
    
    echo ""
}

# Verificar cada modelo
check_model "CONVERSATION (Qwen3-235B-Instruct)" "$CONVERSATION_DIR" 118
check_model "CODE (Qwen3-Coder-480B)" "$CODE_DIR" 241
check_model "AUTOMATION (Qwen3-235B-Thinking)" "$AUTOMATION_DIR" 118

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Resumo final
echo "📋 RESUMO FINAL:"

# Contar modelos completos
complete_count=0

# Verificar conversation
if [ -d "$CONVERSATION_DIR" ]; then
    conv_files=$(find "$CONVERSATION_DIR" -name "*.safetensors" -type f | wc -l)
    if [ "$conv_files" -eq 118 ] && [ -f "$CONVERSATION_DIR/config.json" ]; then
        echo "✅ CONVERSATION: Completo (118/118 ficheiros)"
        ((complete_count++))
    else
        echo "🔄 CONVERSATION: Em progresso ($conv_files/118 ficheiros)"
    fi
else
    echo "❌ CONVERSATION: Não iniciado"
fi

# Verificar code
if [ -d "$CODE_DIR" ]; then
    code_files=$(find "$CODE_DIR" -name "*.safetensors" -type f | wc -l)
    if [ "$code_files" -eq 241 ] && [ -f "$CODE_DIR/config.json" ]; then
        echo "✅ CODE: Completo (241/241 ficheiros)"
        ((complete_count++))
    else
        echo "🔄 CODE: Em progresso ($code_files/241 ficheiros)"
    fi
else
    echo "❌ CODE: Não iniciado"
fi

# Verificar automation
if [ -d "$AUTOMATION_DIR" ]; then
    auto_files=$(find "$AUTOMATION_DIR" -name "*.safetensors" -type f | wc -l)
    if [ "$auto_files" -eq 118 ] && [ -f "$AUTOMATION_DIR/config.json" ]; then
        echo "✅ AUTOMATION: Completo (118/118 ficheiros)"
        ((complete_count++))
    else
        echo "🔄 AUTOMATION: Em progresso ($auto_files/118 ficheiros)"
    fi
else
    echo "❌ AUTOMATION: Não iniciado"
fi

echo ""
echo "📊 Modelos completos: $complete_count/3"

if [ "$complete_count" -eq 3 ]; then
    echo "🎉 TODOS OS MODELOS BAIXADOS COM SUCESSO!"
    echo "🚀 Pode prosseguir com a instalação da Alhica AI!"
else
    echo "⏳ Download ainda em progresso..."
    echo "💡 Execute este script novamente para verificar o progresso"
fi

echo ""
echo "🔄 Para verificar novamente: sudo ./check-models-status.sh"

