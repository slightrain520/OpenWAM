# OpenWAM 新服务器部署指南（基于旧机实测经验）

> 来源：2026-09-25 在一台 10 vCPU / 31GB RAM / RTX 4090 24GB / Ubuntu 22.04 机器上
> 从零配置 Docker 环境、构建镜像、下载资产、跑通 debug 训练的完整实测记录。
> 本文档供在新服务器上用 Agent 配置环境时参考，**每条坑都标注了"新机是否适用"**，避免重复踩坑。

---

## 0. 旧机产出物清单（需要传到新机的东西）

| 产出物 | 路径（旧机） | 大小 | 传输方式 | 是否必须 |
| --- | --- | --- | --- | --- |
| Docker 镜像 | `openwam:cu128` | 27.7GB（gzip 后实测 **9.0GB**） | 已导出 `/home/ubuntu/openwam-cu128-image.tar.gz`，rsync 后 `docker load` | ✅ 强烈建议（免重建，构建需配代理） |
| 仓库代码 | `/home/ubuntu/OpenWAM/` | ~200MB | rsync（**含未提交文件**：`.env`、`compose.gpu-driver.yaml`、`compose.download.yaml`） | ✅ 必须 |
| LIBERO 数据集 | `assets/benchmark_data/libero` | 1.9GB | rsync | ✅（评测用；训练也用） |
| 视频骨干 Wan2.2-TI2V-5B | `assets/video_backbone_ckpt/Wan2.2-TI2V-5B` | 32GB | rsync 或新机重下（ModelScope 快） | 训练才需要 |
| 视频骨干 Wan2.1-VACE-1.3B | `assets/video_backbone_ckpt/Wan2.1-VACE-1.3B` | 18GB | 同上 | 可选（小显存训练用） |
| debug 训练产物 | `outputs/openwam_checkpoints/` | <10MB | 可选 | 仅参考，无实用价值 |
| 官方 checkpoint | —— | 24.8GB | **新机直接从 HF 下载**（见 §4.4） | ✅ serving 必须 |

**rsync 示例**（在旧机执行；`NEW_HOST` 换成新机地址）：

```bash
# 1) 仓库（排除 .git 可选；注意 -a 保留权限，容器 UID 1000 需要文件属主正确）
rsync -avP --exclude .git /home/ubuntu/OpenWAM/ NEW_HOST:/home/ubuntu/OpenWAM/

# 2) 镜像导出包
rsync -avP /home/ubuntu/openwam-cu128-image.tar.gz NEW_HOST:/home/ubuntu/

# 3) 数据集（训练骨干按需）
rsync -avP /home/ubuntu/OpenWAM/assets/benchmark_data/ NEW_HOST:/home/ubuntu/OpenWAM/assets/benchmark_data/
rsync -avP /home/ubuntu/OpenWAM/assets/video_backbone_ckpt/ NEW_HOST:/home/ubuntu/OpenWAM/assets/video_backbone_ckpt/
```

> 带宽参考：旧机出口仅 100Mbps（≈12MB/s），14GB 镜像 ≈ 20 分钟，50GB 骨干 ≈ 70 分钟。
> 若新机带宽好，骨干/数据在新机直接用下载脚本重下可能更快（ModelScope 直连曾实测 9.6MB/s）。

---

## 1. 踩坑记录（按"新机是否适用"分类）

### 1.1 【视新机网络而定】代理配置

旧机代理 `http://127.0.0.1:17897` 只监听 127.0.0.1，导致一串连锁配置：

| 坑 | 现象 | 对策（旧机做法） | 新机判断 |
| --- | --- | --- | --- |
| apt 走代理 | apt 拉官方 docker 源超时 | `/etc/apt/apt.conf.d/95proxy` 写 Acquire::http::Proxy | 新机能直连外网则不需要 |
| docker daemon 走代理 | pull 镜像超时 | `/etc/systemd/system/docker.service.d/http-proxy.conf` + `systemctl daemon-reload && systemctl restart docker` | 同上 |
| build 走代理 | 镜像构建时 pip 超时 | `make docker-build DOCKER_BUILD_ARGS='--network=host --build-arg HTTP_PROXY=... --build-arg HTTPS_PROXY=... --build-arg http_proxy=... --build-arg https_proxy=... --build-arg NO_PROXY=...'`（**大小写两套都要**，`--network=host` 让容器内够得着 127.0.0.1 代理） | 若新机能直连，直接 `make docker-build` 即可 |

### 1.2 【视新机网卡而定】docker0 MTU 导致容器内 TLS 卡死

- **现象**：容器内 `pip`/`curl https` 大文件卡死或极慢，小包正常；表现为 TLS 握手无响应。
- **根因**：旧机物理网卡 MTU=1442（云商隧道网络），docker0 默认 1500 → 出方向大包被丢。
- **对策**：`/etc/docker/daemon.json` 设 `"mtu": 1440`（物理 MTU - 2 以内），重启 docker。
- **新机判断**：`ip link show` 看主网卡 MTU；**是标准 1500 就完全不用配**；若是 1450/1442 之类隧道网络，照抄此条。

### 1.3 【视 GPU 型号而定】GeForce + 驱动版本低于镜像 CUDA 要求 → Error 804

- **现象**：容器内任何 CUDA 调用报 `CUDA error 804 (forward compatibility was attempted on non supported hw)`。
- **根因**：`openwam:cu128` 镜像（CUDA 12.8）内 ldconfig 缓存把 `cuda-compat-570` 的 libcuda 排在宿主 550 驱动的真 libcuda 之前；**GeForce 卡不支持 compat 包**（数据中心卡才支持）。
- **对策**：仓库根目录的 `compose.gpu-driver.yaml` 覆盖文件，给服务注入
  `LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH}` 强制 compat 目录优先（compat 库在 GeForce 上恰好可用）。
  `.env` 里 `COMPOSE_FILE=compose.yaml:compose.gpu-driver.yaml` 启用。
- **新机判断**：
  - 数据中心卡（A100/A800/H100/H800/L40S…）或驱动 ≥ 镜像 CUDA 要求 → **不需要**，`.env` 的 `COMPOSE_FILE` 只留 `compose.yaml`；
  - GeForce + 驱动低于 12.8 对应版本（570）→ 照抄覆盖文件，或干脆 `apt` 升级宿主驱动到 ≥570 后不用覆盖。

### 1.4 【通用】权限与常用操作

- shell 无 docker 组权限时所有 docker 命令要包 `sg docker -c "..."`。新机正确做法：`sudo usermod -aG docker $USER` 后**重新登录**，一劳永逸。
- 镜像构建参数、验证命令见 `assets/openwam_usage_docs/docker.md`，验收三件套：
  ```bash
  make docker-check                    # CPU 环境自检（2066 项）
  make gpu-check                       # 容器内 CUDA/PyTorch/GPU 自检
  make docker-integration-check PYTHON=python3   # 13 项集成测试
  ```
- **机器可能中途重启**：长任务（下载/训练）一律后台跑 + 输出日志文件，例如
  `nohup ... > /tmp/train.log 2>&1 &`，重启后凭日志和产物恢复。

### 1.5 【通用】下载源选择

| 资产 | 推荐源 | 原因 |
| --- | --- | --- |
| 视频骨干（Wan2.x / Cosmos） | **ModelScope**：`--source modelscope` + `-e HTTP_PROXY= -e HTTPS_PROXY= -e NO_PROXY='*'` | 旧机实测 HF 经代理仅 ~75KB/s/连接，ModelScope 直连 9.6MB/s（快 128 倍） |
| LIBERO / RoboTwin 数据集 | HF（脚本默认） | 量小（1.9GB），可接受 |
| **官方 checkpoint（24.8GB）** | **仅 HF 源** | `download_openwam_checkpoints.py` 无 ModelScope 选项；新机需保证 HF 带宽，或配代理 |

下载在容器内进行时用 `compose.download.yaml` 辅助服务（host 网络 + 代理 env）：
```bash
sg docker -c "docker compose -f compose.yaml -f compose.download.yaml --profile download run --rm download \
  python scripts/download_assets/download_video_backbone.py --name Wan2.2-TI2V-5B --source modelscope --yes"
```

### 1.6 【通用】train/serve 容器不读宿主仓库的 yaml —— 必须显式传路径

- **现象**：下载脚本明明把 `configs/model/video_backbone/*.yaml` 的 `model_path` 更新成了正确路径，训练仍报
  `model_path does not exist: /path/to/Wan2.1-VACE-1.3B`。
- **根因**：`train`/`serve` 服务把源码**烘焙进镜像**（不挂载宿主仓库），容器读到的是构建时的旧配置；下载脚本只改了宿主 checkout。
- **对策**：训练/部署命令一律显式覆盖（容器内路径）：
  ```bash
  model.video_backbone.model_path=/opt/openwam/assets/video_backbone_ckpt/<骨干名>
  dataloader.dataset_dir=/opt/openwam/assets/benchmark_data/<数据集名>
  ```
- Hydra 语法注意：**选组**用斜杠 `model/video_backbone=wan21_vace_1_3b`，**改值**用点 `model.video_backbone.model_path=...`。

### 1.7 【通用·最重要的容量经验】CPU 内存两个坑：加载瞬态 & 优化器卸载

24GB 卡 + 31GB RAM 机器实测，两次 OOM 被杀后定位：

**坑 A —— 模型加载瞬态峰值 ≈ 参数量 × 4B + 权重文件总大小**

构建路径先在 CPU 上 fp32 随机初始化整个架构，再用 `torch.load` 物化权重文件逐一 assign：
- 8.6B 总参数（VACE-1.3B 骨干架构）实测**纯加载阶段匿名内存峰值 ~46GB** → 31GB RAM 直接 OOM，
  加 24GB swap 仍不够（第二次 OOM 时 swap 耗尽为证）。
- 结论：**RAM 必须 ≥ 加载瞬态**，即 `总参数×4B + 权重文件总大小`，另留余量。

**坑 B —— ZeRO-2 优化器 CPU 卸载 = 可训练参数 × 12B（fp32 param+m+v）**

- 实测 2.8B 可训练参数 → 33.6GB CPU 常驻优化器状态。
- 31GB RAM 下热数据超物理内存 → **每步从磁盘换页**（vmstat 持续 si/so），每步从正常 1-2 分钟恶化到 514→902 秒。
- 结论：若用 `offload_optimizer_device=cpu`，RAM 需 ≥ 可训练参数×12B + 数据加载/框架开销；否则优化器放 GPU（吃显存）。

**坑 C —— 显存边界（RTX 4090 24GB 实测）**

| 配置 | 显存占用 | 结论 |
| --- | --- | --- |
| `wan21_vace_1_3b` + batch=1 + `offload_optimizer_device=cpu` + 梯度检查点 | **23.4GB / 24GB（满载）** | 勉强能跑，是 24GB 卡的极限 |
| 默认 `wan22_ti2v_5b`（batch=16 默认配置） | 仅 5B DiT 的 bf16 权重+梯度 ≈ 20GB+ | **完全不可行** |

**坑 D —— batch=1 的 loss 波动是正常的**

flow matching 每步采样**单个样本 + 单个随机时间步 t**，t 大小不同损失天然差一个数量级。
实测 6 步 loss：0.26 → 0.50 → 4.66 → **0.13** → 2.38 → 2.45（第 4 步回落比第 1 步还低）——
无趋势波动，**不要误判为训练崩溃**；判断健康与否应看长窗口均值和 grad_norm 是否有限。

---

## 2. 新机安装顺序（完整清单）

```bash
# ── 1. Docker Engine + Compose（官方 apt 源）──
# 参考 https://docs.docker.com/engine/install/ubuntu/；需要代理时先配 1.1 节三件套
sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER && exec newgrp docker   # 之后重新登录生效

# ── 2. /etc/docker/daemon.json ──
# nvidia runtime 必须有；mtu 仅当物理网卡 MTU≠1500 才设（见 1.2）
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "runtimes": { "nvidia": { "path": "nvidia-container-runtime", "runtimeArgs": [] } },
  "default-runtime": "nvidia"
}
EOF
sudo systemctl restart docker

# ── 3. NVIDIA Container Toolkit ──
# 参考 https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

# ── 4. 载入镜像 ──
docker load < /home/ubuntu/openwam-cu128-image.tar.gz   # 或 zcat | docker load

# ── 5. 仓库与 .env ──
# rsync 过去的仓库里检查 .env：
#   COMPOSE_FILE=compose.yaml            ← 数据中心卡/新驱动；GeForce 旧驱动则加 :compose.gpu-driver.yaml
#   OPENWAM_CHECKPOINT_DIR=<官方 ckpt 绝对路径>   ← serve 服务的挂载源
#   OPENWAM_IMAGE=openwam:cu128
sudo chown -R 1000:1000 /home/ubuntu/OpenWAM/assets   # 容器 UID 1000 需要写权限

# ── 6. 验收三件套 ──
make docker-check && make gpu-check && make docker-integration-check PYTHON=python3

# ── 7. 下载官方 checkpoint（serving 用，24.8GB，HF 源）──
python scripts/download_assets/download_openwam_checkpoints.py \
  --family alpha --name OpenWAM-Alpha-Sim-LIBERO --yes
# 产物在 assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO/
# ⚠️ 必须与仿真器配对：LIBERO 评测用 OpenWAM-Alpha-Sim-LIBERO（eef 表示，客户端会 ping 校验）
#    RoboTwin 评测用 OpenWAM-Alpha-Sim-RoboTwin-{Clean2Random,Full}

# ── 8. LIBERO 仿真客户端环境（独立 conda，与 Docker 完全隔离）──
CONDA_BIN=/path/to/conda/bin/conda \
LIBERO_ENV_PREFIX=/path/to/envs/libero \
LIBERO_PATH=/path/to/LIBERO \
bash benchmarks/libero/setup_env.sh          # 克隆 LIBERO(锁定commit)+打补丁+建环境+版本校验
LIBERO_PATH=/path/to/LIBERO LIBERO_PYTHON=/path/to/envs/libero/bin/python \
bash benchmarks/libero/run_smoke.sh env      # 冒烟：env | import | task
```

---

## 3. 启动 serving 与评测（观察推理效果）

```bash
# ── 方式一：compose serve 服务（.env 设好 OPENWAM_CHECKPOINT_DIR 后）──
docker compose up -d serve && docker compose logs -f serve
# 注意 healthcheck start_period=10min：加载 24.8GB 权重较慢，属正常

# ── 方式二：脚本直起（文档主推）──
bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO
# 单文件/多卡：deploy.sh <ckpt_dir> --ckpt-name checkpoint_step_XXXX.safetensors
# NUM_GPUS=4 PORT_BASE=8848 可一卡一 server

# ── 评测（先冒烟再全量）──
# 单任务（对着已运行的 server）：
LIBERO_PATH=/path/to/LIBERO LIBERO_PYTHON=/path/to/envs/libero/bin/python \
bash benchmarks/libero/single_eval.sh libero_spatial 0 8848 127.0.0.1

# 全量 4 套件×10任务×50试验（协议锁定 seed=42），先加 --smoke：
SERVER_PYTHON=<openwam环境python> LIBERO_PYTHON=<libero环境python> LIBERO_PATH=<LIBERO checkout> \
GPUS=0,1 REPLICAS_PER_GPU=1 \
bash benchmarks/libero/run_eval.sh \
  assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO checkpoint_step_XXXX.safetensors \
  --compile-enabled false --render-gpus 2,3
```

### 3.1 评测阶段的三个已知坑（README 记录，旧机验证过条目来源）

1. **渲染卡与推理卡必须分开**：MuJoCo EGL 渲染器与持续 CUDA 计算共用物理 GPU 时，
   `mjr_readPixels` 内 abort → 客户端 `exit=-6`（驱动层冲突，驱动 570.124.06 复现）。
   多卡用 `--render-gpus` 分配；**单卡机器先跑 --smoke 碰运气，崩了就得加卡**。
2. **`--compile-enabled false` 先行**：`torch._inductor` Triton launcher 有
   `Fatal Python error: none_dealloc` 风险；跑稳后再尝试开编译提速。
3. **checkpoint 与动作表示必须匹配**：客户端 ping 校验，LIBERO 评测必须是 `eef` 表示的 checkpoint。
   目录里必须有 `config.yaml` + `normalization_stats.npy`。

### 3.2 结果核对

`run_eval.sh` 输出 `summary.csv` / `summary.json`（输出目录加 `--output-dir` 可断点续跑）。
与论文基准对照（Spatial/Object/Goal/Long 四套件成功率）确认环境搭建正确。

---

## 4. 新机硬件配置建议（按用途）

### 4.1 内存需求推算（旧机实测外推）

| 需求项 | 公式 | 推理（serving） | 训练（默认 wan22_ti2v_5b） |
| --- | --- | --- | --- |
| 加载瞬态 RAM | 总参数×4B + 权重文件总量 | ~9.6B 参数 + 24.8GB ckpt ≈ **50-60GB** | ~9.6B 参数 + 34GB 骨干 ≈ **70-75GB** |
| 稳态 RAM | 优化器卸载(可训练×12B) + 框架/dataloader | 少量（优化器在 GPU） | ~6.5B 可训练×12B ≈ **78GB** + 开销 |
| 显存 | 权重 bf16 + 梯度 + 激活 | 权重~19GB + KV/激活/CUDA上下文 ≈ **35-45GB** | 权重19GB + 梯度13GB + batch16激活(检查点) → **80GB 从容** |

（训练可训练参数从旧机实测推算：VACE-1.3B 架构可训练 2.8B，其中 video 2.15B；5B 骨干架构可训练约 6.5B）

### 4.2 推荐档位

| 档位 | 配置 | 覆盖范围 |
| --- | --- | --- |
| **推荐** | **2× H100/A100 80GB，32 vCPU，256GB RAM，1TB NVMe** | 训练（可单卡或双卡 DDP/ZeRO）+ 评测（1 卡推理 1 卡渲染）全覆盖，无 OOM 之虞 |
| 均衡 | 1× 80GB（推理/训练）+ 1× 24-48GB（渲染），32 vCPU，192GB RAM，1TB | 训练单卡 batch 适中；评测渲染分离 |
| 下限 | 1× 80GB，16 vCPU，128GB RAM，500GB | 能 serving、能小 batch 训练；**评测渲染共卡有 exit=-6 风险** |

**明确避雷（有实测代价）**：
- ❌ 24GB 消费卡：serving 放不下 24.8GB checkpoint；训练实测满载极限（1.3B 骨干 batch=1）
- ❌ 32GB RAM：加载即 OOM（实测 8.6B 参数加载峰值 ~46GB）
- ❌ RAM 够但 <128GB 的训练机：优化器卸载 78GB 必然颠簸（实测 33.6GB 已让 31GB 机器每步慢 5-8 倍）
- ❌ 机械盘/网络盘：checkpoint 保存（~24GB/个）与 swap 兜底都需要 NVMe
- 磁盘最低 500GB：镜像 28GB + ckpt 25GB + 骨干 50GB + 数据 + 训练 checkpoint 滚动保存 + 系统

### 4.3 网络要求

- 国际带宽（HF）：checkpoint 24.8GB 仅 HF 源；国内机器需配代理或保证 HF CDN 带宽
- ModelScope 可达：骨干/数据下载快 100 倍量级
- 两机间传输带宽：镜像包 14GB + 资产最多 ~85GB

### 4.4 训练阶段补充（新机上正式训练时）

```bash
# 默认配置（wan22_ti2v_5b + robotwin 数据）在 80GB 卡上可直接跑：
docker compose run --rm train train dataloader=robotwin \
  model.video_backbone.model_path=/opt/openwam/assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  dataloader.dataset_dir=/opt/openwam/assets/benchmark_data/robotwin   # 路径按实际数据集
# RAM ≥192GB 时可去掉 offload_optimizer_device=cpu 换显存换速度，或保持 CPU 卸载提速
```

---

## 5. 旧机遗留物说明（迁移后旧机可清理）

- 应急 swap 文件已全部删除（曾临时建 4 个共 88GB，训练终止后已 `swapoff` + `rm`，磁盘回到 44%）
- `/tmp/opencode/*.log`：构建/下载/训练全程日志（docker-build、gpu-check、dl-*、train-debug），迁移前可打包带走作参考
- 旧机 `outputs/openwam_checkpoints/2026-09-25_13-05-15_debug/`：debug 产物仅 6 步，无 checkpoint 落盘（save@10 未到即终止），可不迁移
