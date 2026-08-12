#!/usr/bin/env bash
# 把 Bernini-R 1.3B DiT（diffusers 格式）转换成 Causal-Forcing 用的 vendored-Wan 布局。
# 从 Causal-Forcing 仓库根目录运行。
set -e

CF_ROOT="/apdcephfs_hzlf/share_1227201/xinyu/Causal-Forcing"                                  # 仓库根目录
BERNINI_DIR="/apdcephfs_hzlf/share_1227201/xinyu/my_project/Bernini/Bernini-R-1.3B-Diffusers"  # 输入：Bernini-R 1.3B diffusers 权重目录
PY="${PY:-/opt/conda/envs/causvid/bin/python}"

cd "$CF_ROOT"
# 参数说明：
#   --bernini_dir  待转换的 diffusers 权重目录
#   --wan_ref      参考的 vendored-Wan 布局（提供键名/形状模板，用于 825 参数精确匹配）
#   --out_dir      输出：转换后的 vendored-Wan 布局权重
"$PY" bernini_causvid/tools/convert_bernini_to_wan.py \
    --bernini_dir "$BERNINI_DIR" \
    --wan_ref wan_models/Wan2.1-T2V-1.3B \
    --out_dir wan_models/Bernini-R-1.3B
