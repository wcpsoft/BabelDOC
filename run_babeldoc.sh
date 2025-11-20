#!/bin/bash
# configurable_babeldoc.sh

set -e

# 配置变量
OPENAI_API_KEY="<YOUR_TOKEN>"
MODEL="<YOUR_MODEL_ID>"
BASE_URL="<YOUR_MODEL_BASE_URL>"
API_QPS="40"
# 显示帮助信息
show_help() {
    echo "用法: $0 <文件1> [文件2] [文件3] ..."
    echo ""
    echo "示例:"
    echo "  $0 example.pdf"
    echo ""
    echo "配置变量 (请在脚本中修改):"
    echo "  OPENAI_API_KEY: $OPENAI_API_KEY"
    echo "  MODEL: $MODEL"
    echo "  BASE_URL: $BASE_URL"
    echo "  API_QPS: $API_QPS"
    exit 1
}

# 检查是否有参数
if [ $# -eq 0 ]; then
    echo "❌ 错误: 请指定要处理的文件"
    show_help
fi

# 从命令行参数构建文件列表
FILES="$*"

echo "🔧 使用配置:"
echo "- Model: $MODEL"
echo "- Base URL: $BASE_URL"
echo "- Files: $FILES"
echo ""

# 函数：检查并验证文件
check_file() {
    local file="$1"
    
    # 检查文件是否存在
    if [ ! -e "$file" ]; then
        echo "❌ 错误: 文件 '$file' 不存在"
        return 1
    fi
    
    # 检查是否为普通文件
    if [ ! -f "$file" ]; then
        echo "❌ 错误: '$file' 不是普通文件"
        return 1
    fi
    
    # 检查文件是否可读
    if [ ! -r "$file" ]; then
        echo "❌ 错误 错误: 没有读取 '$file' 的权限"
        return 1
    fi
    
    # 检查文件大小
    local file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
    if [ "$file_size" -eq 0 ]; then
        echo "⚠️  警告: 文件 '$file' 为空"
    fi
    
    echo "✅ 文件 '$file' 验证通过"
    return 0
}

# 检查依赖
if ! command -v babeldoc &> /dev/null; then
    echo "❌ 错误 错误: 请先安装 babeldoc"
    exit 1
fi

# 检查API密钥是否已设置
if [ "$OPENAI_API_KEY" = "your-api-key-here" ] || [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ 错误 错误: 请设置有效的 OpenAI API 密钥"
    exit 1
fi

# 逐个检查每个文件
echo "📁 开始文件检查..."
all_files_valid=true

for file in "$@"; do
    if ! check_file "$file"; then
        all_files_valid=false
    fi
done

echo ""

# 如果有文件验证失败，则退出
if [ "$all_files_valid" = false ]; then
    echo "❌ 文件检查失败，请解决上述问题后重试"
    exit 1
fi

echo "✅ 所有文件检查通过，开始处理..."
echo ""

# 执行命令
babeldoc \
    --openai \
    --openai-model "$MODEL" \
    --openai-base-url "$BASE_URL" \
    --openai-api-key "$OPENAI_API_KEY" \
    --qps "$API_QPS" \
    --files "$@"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 任务完成！"
else
    echo ""
    echo "💥 处理过程中出现错误"
    exit 1
fi
