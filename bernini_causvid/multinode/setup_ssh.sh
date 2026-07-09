#!/usr/bin/env bash
# 用密码把"本机(主节点)公钥"授权到每个 worker，实现主 -> worker 免密。
# 幂等：已免密的 worker 会自动跳过。密码只在内存中传给 expect，不写入任何文件。
#
# 用法：
#   WORKER_PW='你的密码' bash setup_ssh.sh 29.209.114.22 [worker2 ...]
#   bash setup_ssh.sh 29.209.114.22                # 不带 WORKER_PW 时交互式输入密码
set -e
CLUSTER_DIR=/apdcephfs_hzlf/share_1227201/xinyu/.cluster
PUB="$CLUSTER_DIR/master_node_id_rsa.pub"
CONDA_PY=/apdcephfs_hzlf/share_1227201/xinyu/conda_setup/miniconda3/envs/causal_forcing/bin/python

[ "$#" -ge 1 ] || { echo "用法: [WORKER_PW=密码] bash setup_ssh.sh <worker_ip> [worker2 ...]"; exit 1; }
command -v expect >/dev/null || { echo "缺少 expect，请先安装（yum install -y expect）"; exit 1; }

# 1) 确保主节点密钥对存在，并把公钥放到共享目录
mkdir -p "$CLUSTER_DIR" ~/.ssh && chmod 700 ~/.ssh
[ -f ~/.ssh/id_rsa ] || ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
cp ~/.ssh/id_rsa.pub "$PUB" && chmod 644 "$PUB"
touch ~/.ssh/config && chmod 600 ~/.ssh/config

# 2) 先确定哪些 host 还没免密（已免密的完全不需要密码）
NEED_PW=()
for host in "$@"; do
    grep -q "Host $host\$" ~/.ssh/config 2>/dev/null || \
        printf 'Host %s\n    StrictHostKeyChecking no\n    UserKnownHostsFile /dev/null\n    ConnectTimeout 8\n\n' "$host" >> ~/.ssh/config
    if ssh -o BatchMode=yes -o ConnectTimeout=6 "$host" 'true' 2>/dev/null; then
        echo "[$host] 已免密，跳过。"
    else
        NEED_PW+=("$host")
    fi
done

if [ "${#NEED_PW[@]}" -eq 0 ]; then
    echo "全部 worker 已免密，无需密码。"
    exit 0
fi

# 3) 有 host 需要授权，才要密码
if [ -z "${WORKER_PW:-}" ]; then
    [ -t 0 ] || { echo "以下 worker 未免密且未提供 WORKER_PW: ${NEED_PW[*]}"; echo "请用: WORKER_PW='密码' 重新运行"; exit 1; }
    read -r -s -p "输入 worker 登录密码: " WORKER_PW; echo
fi
export WORKER_PW

for host in "${NEED_PW[@]}"; do
    echo "[$host] 用密码授权公钥中..."
    # 远端命令通过环境变量传给 expect（用 Tcl 的 $env(REMOTE_CMD) 引用），
    # 避免命令里的 $()/$KEY 等被 Tcl 当成变量替换，也保证它作为单个参数传给 ssh。
    export REMOTE_CMD="mkdir -p ~/.ssh && chmod 700 ~/.ssh && KEY=\$(cat $PUB) && grep -qF \"\$KEY\" ~/.ssh/authorized_keys 2>/dev/null || echo \"\$KEY\" >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys && echo AUTHORIZED_OK && (ls $PUB >/dev/null && echo SHARED_FS_OK || echo SHARED_FS_MISSING) && (ls $CONDA_PY >/dev/null && echo CONDA_OK || echo CONDA_MISSING)"

    expect -c "
        set timeout 40
        spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$host \$env(REMOTE_CMD)
        expect {
            -re {[Pp]assword:} { send \"\$env(WORKER_PW)\r\"; exp_continue }
            -re {yes/no} { send \"yes\r\"; exp_continue }
            eof
        }
        catch wait result
        exit [lindex \$result 3]
    " | grep -aE "AUTHORIZED_OK|SHARED_FS_OK|SHARED_FS_MISSING|CONDA_OK|CONDA_MISSING|denied" || true
    unset REMOTE_CMD

    if ssh -o BatchMode=yes -o ConnectTimeout=6 "$host" 'true' 2>/dev/null; then
        echo "[$host] 免密 OK ✅"
    else
        echo "[$host] 免密仍失败 ❌（检查密码 / 网络 / 对端 sshd）"; exit 2
    fi
done
unset WORKER_PW
echo "全部 worker 免密配置完成。"
