#!/bin/bash
# AutoDL 新机一键初始化脚本
# 执行：chmod +x autodl_init.sh && ./autodl_init.sh

# 初始化 conda 环境
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate base
fi

echo "===== 1. 配置Git全局用户信息 ====="
git config --global user.name "donglinAI"
git config --global user.email "donglin_cui@163.com"
echo "Git config done."

echo "===== 2. 生成SSH密钥 ed25519 ====="
SSH_PRIVATE_KEY="$HOME/.ssh/id_ed25519"
if [ -f "$SSH_PRIVATE_KEY" ]; then
    echo "✅ 检测到已存在SSH私钥 $SSH_PRIVATE_KEY，跳过生成密钥"
    echo "===== 公钥内容 ====="
    cat "$SSH_PRIVATE_KEY.pub"
else
    echo "🔧 未检测到SSH密钥，开始生成 ed25519 密钥对"
    # -N "" 代表密钥不设置密码；-f 指定密钥文件路径，避免交互式询问
    ssh-keygen -t ed25519 -C "donglin_cui@163.com" -N "" -f "$SSH_PRIVATE_KEY"
    echo "===== 公钥内容（复制下面全部，添加到GitHub SSH Keys） ====="
    cat "$SSH_PRIVATE_KEY.pub"
fi

echo -e "\n===== 3. 加载AutoDL网络加速 ====="
source /etc/network_turbo

echo "===== 4. 设置HF_HOME与ModelScope环境变量 ====="
# HuggingFace
export HF_HOME=/root/autodl-tmp/hf_cache
if ! grep -qxF 'export HF_HOME=/root/autodl-tmp/hf_cache' ~/.bashrc; then
    echo 'export HF_HOME=/root/autodl-tmp/hf_cache' >> ~/.bashrc
    echo "✅ HF_HOME已写入 ~/.bashrc"
else
    echo "✅ ~/.bashrc 已存在HF_HOME配置，跳过写入"
fi

# ModelScope
export MODELSCOPE_HUB=/root/autodl-tmp/modelscope_cache
if ! grep -qxF 'export MODELSCOPE_HUB=/root/autodl-tmp/modelscope_cache' ~/.bashrc; then
    echo 'export MODELSCOPE_HUB=/root/autodl-tmp/modelscope_cache' >> ~/.bashrc
    echo "✅ MODELSCOPE_HUB已写入 ~/.bashrc"
else
    echo "✅ ~/.bashrc 已存在MODELSCOPE_HUB配置，跳过写入"
fi

echo "===== 5. Clone 仓库到 autodl-tmp ====="
cd /root/autodl-tmp
if [ ! -d "ai-from-scratch" ]; then
    git clone git@github.com:donglinAI/ai-from-scratch.git
else
    echo "✅ ai-from-scratch仓库已存在，跳过clone"
fi

echo -e "\n==================== 初始化完成 ===================="
echo "⚠️ 重要：如果是新生成密钥，需要手动复制公钥添加到 GitHub -> Settings -> SSH and GPG keys"
echo "测试ssh连接命令：ssh -T git@github.com"
echo "HF_HOME=/root/autodl-tmp/hf_cache"
echo "仓库位置：/root/autodl-tmp/ai-from-scratch"