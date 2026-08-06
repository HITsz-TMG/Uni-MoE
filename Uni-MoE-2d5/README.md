# Uni-MoE-2d5

Uni-MoE-2d5 6B 的公开推理代码，提供：

- vLLM `0.22.1` out-of-tree 模型插件，无需修改 vLLM 镜像源码；
- Hugging Face Transformers + Ascend NPU 推理入口；
- 文本、图像、音频和视频 OpenAI-compatible 客户端；
- 支持多模态上传和流式输出的 Web demo。

代码已经在官方镜像
`quay.io/ascend/vllm-ascend:v0.22.1rc1-a3`、Ascend 910C/A3 环境中完成
vLLM、Hugging Face 和 demo 四模态验证。模型权重不包含在本仓库中。

## 运行环境

| 组件 | 版本/要求 |
|---|---|
| Hardware | Ascend 910C / A3 |
| Image | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` |
| vLLM | `0.22.1` |
| vLLM Ascend | `0.22.1rc1` |
| Python | 3.10、3.11 或 3.12 |
| HF attention | `sdpa` |

不要使用 pip 覆盖官方镜像自带的 `torch`、`torch_npu`、`vllm` 或
`vllm-ascend`。本项目通过 `vllm.general_plugins` 注册模型，安装后可以直接使用
标准镜像中的 `vllm serve`。

## Checkpoint 要求

模型目录至少需要包含：

```text
config.json
tokenizer_config.json
processor_config.json（或 preprocessor_config.json）
模型权重和 tokenizer 文件
```

关键元数据应满足：

```json
{
  "architectures": ["UniMoE2d5ForConditionalGeneration"],
  "model_type": "qwen3_vl_div_moe"
}
```

processor 元数据中的 `processor_class` 应为 `UniMoE2d5Processor`。可在启动前运行
预检确认 checkpoint 与运行环境是否匹配：


```bash
python scripts/preflight.py --model /path/to/UniMoE-2.5-6B
```

## 在官方镜像中部署

先在宿主机克隆代码，并准备 checkpoint：

```bash
git clone https://github.com/Tiachi/Uni-MoE-2d5-OpenSource.git
REPO=$(pwd)/Uni-MoE-2d5-OpenSource
MODEL=/path/to/UniMoE-2.5-6B
```

以下示例挂载 16 张 NPU。

```bash
IMAGE=quay.io/ascend/vllm-ascend:v0.22.1rc1-a3

device_args=()
for index in $(seq 0 15); do
  device_args+=(--device "/dev/davinci${index}")
done

docker run --rm -it \
  --name unimoe2d5 \
  --net=host \
  --ipc=host \
  "${device_args[@]}" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -e MODEL="${MODEL}" \
  -v "${REPO}:/workspace/Uni-MoE-2d5:ro" \
  -v "${MODEL}:${MODEL}:ro" \
  "${IMAGE}" bash
```

在容器内安装本项目。

```bash
cd /workspace/Uni-MoE-2d5
python -m pip install -e . --no-deps
python scripts/preflight.py --model "${MODEL}"
```

## vLLM 推理

单卡 eager 配置：

```bash
cd /workspace/Uni-MoE-2d5
export ASCEND_RT_VISIBLE_DEVICES=0

MODEL_PATH="${MODEL}" \
MAX_MODEL_LEN=8192 \
ENFORCE_EAGER=1 \
bash scripts/serve_vllm.sh
```

另一个终端中，图像、音频和视频：

```bash
cd /workspace/Uni-MoE-2d5
python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r demo/requirements.txt

python scripts/create_smoke_media.py --output-dir /tmp/unimoe2d5-smoke-media
python scripts/smoke_http.py
python examples/openai_client.py \
  --image /tmp/unimoe2d5-smoke-media/test.png --prompt "描述图片"
python examples/openai_client.py \
  --audio /tmp/unimoe2d5-smoke-media/test.wav --prompt "描述音频"
python examples/openai_client.py \
  --video /tmp/unimoe2d5-smoke-media/test.mp4 --prompt "概括视频"
```

16 卡 data parallel + expert parallel eager 配置：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

MODEL_PATH="${MODEL}" \
DP_SIZE=16 \
ENABLE_EXPERT_PARALLEL=1 \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.90 \
ENFORCE_EAGER=1 \
bash scripts/serve_vllm.sh
```

也可以把 `ENFORCE_EAGER` 设为 `0`，以非 eager 模式启动。此时启动脚本不会传入
`--enforce-eager`，vLLM Ascend 可以使用当前版本支持的图编译/图捕获路径：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

MODEL_PATH="${MODEL}" \
DP_SIZE=16 \
ENABLE_EXPERT_PARALLEL=1 \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.90 \
ENFORCE_EAGER=0 \
bash scripts/serve_vllm.sh
```

`scripts/serve_vllm.sh` 默认把公开模型名设置为 `unimoe2d5`，可通过 `SERVED_MODEL_NAME` 修改，需要 API 鉴权时，应从运行环境注入密钥：

```bash
read -rsp "vLLM API key: " VLLM_API_KEY
export VLLM_API_KEY
MODEL_PATH="${MODEL}" bash scripts/serve_vllm.sh --api-key "${VLLM_API_KEY}"
```

## Hugging Face 推理

```bash
cd /workspace/Uni-MoE-2d5
export ASCEND_RT_VISIBLE_DEVICES=0

python examples/infer_hf.py \
  --model "${MODEL}" \
  --prompt "请用一句话介绍你自己"
```

图像、音频和视频：

```bash
python examples/infer_hf.py --model "${MODEL}" \
  --image /path/to/image.jpg --prompt "描述图片"
python examples/infer_hf.py --model "${MODEL}" \
  --audio /path/to/audio.wav --prompt "描述音频"
python examples/infer_hf.py --model "${MODEL}" \
  --video /path/to/video.mp4 --prompt "概括视频"
```

## Web demo

先启动 vLLM 服务，再启动 demo：

```bash
cd /workspace/Uni-MoE-2d5
python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r demo/requirements.txt

VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
MODEL=auto \
DEMO_HOST=0.0.0.0 \
DEMO_PORT=8500 \
bash scripts/run_demo.sh
```

浏览器打开 `http://<server-address>:8500/`，可发送文本、图片、音频和视频。
若 vLLM 启用了 API key，只通过进程环境传递给 demo：

```bash
export VLLM_API_KEY
VLLM_BASE_URL=http://127.0.0.1:8000/v1 bash scripts/run_demo.sh
```

## 配置项

vLLM 启动脚本常用变量：

| 变量 | 默认值 |
|---|---|
| `HOST` / `PORT` | `0.0.0.0` / `8000` |
| `SERVED_MODEL_NAME` | `unimoe2d5` |
| `TP_SIZE` / `DP_SIZE` | `1` / `1` |
| `MAX_MODEL_LEN` | `8192` |
| `GPU_MEMORY_UTILIZATION` | `0.90` |
| `ENFORCE_EAGER` | `1` |
| `ENABLE_EXPERT_PARALLEL` | `0` |
| `LIMIT_MM_PER_PROMPT` | `{"image":1,"video":1,"audio":1}` |

## 本地检查

```bash
python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -e ".[test,demo]"
ruff check src/unimoe2d5/__init__.py src/unimoe2d5/compat.py \
  src/unimoe2d5/plugin.py examples demo scripts tests
python -m unittest discover -s tests -v
python -m pip wheel . --no-deps --wheel-dir dist
```

## 开源边界

本仓库只发布推理源码与 demo，不包含模型权重。

## License

代码采用 [Apache License 2.0](LICENSE)。第三方来源与归属说明见 [NOTICE](NOTICE)。
