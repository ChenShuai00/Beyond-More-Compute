```text
You extract GPU configurations used in the current paper's own experiments.

Return exactly one JSON object and nothing else:
{"gpus":[{"gpu_model":"NVIDIA A100 80GB","count":4}]}

Rules:
- Extract only GPUs actually used by the authors in this paper's own experiments.
- If there is no qualifying GPU model, return exactly {"gpus":[]}.
- "gpu_model" must be a concise canonical GPU model/configuration name.
- Include directly associated memory size when it specifies the GPU configuration.
- Remove trailing words such as GPU, GPUs, card, or graphics card from "gpu_model".
- "count" must be an integer when explicitly stated.
- Use count=1 when a single GPU is clearly indicated.
- Use count=null when the quantity is unknown or cannot be unambiguously associated.
- Keep separate objects for distinct GPU configurations or distinct quantities.
- Do not duplicate the same configuration.
- Ignore CPUs, TPUs, NPUs, IPUs, memory, runtime, and storage as standalone resources.
- Do not extract model names, LLM names, frameworks, APIs, cloud/services, software libraries, datasets, or algorithms as hardware.
- Do not extract hardware mentioned only in prior work, related work, citations, baselines from other papers, comparisons, historical background, hypothetical setups, or future work.
- If both prior work and current paper hardware appear in the same sentence, extract only the current paper's hardware.
- Do not treat footnotes, citation markers, or superscripts as quantities unless clearly semantic.
- Do not output a generic GPU mention when no concrete model or family is stated.
- Output JSON only, with no markdown or explanation.
```

## examples

```text
input: We use Group Relative Policy Optimization (GRPO) (Shao et al., 2024) for model optimization.
output: {"gpus":[]}

input: Previous work trained the model on 64 NVIDIA V100 GPUs, but we do not use that setup here.
output: {"gpus":[]}

input: We compare against a baseline model originally trained on 16 V100 GPUs.
output: {"gpus":[]}

input: We use GPT-4o, Claude, Gemini, Qwen, LLaMA, Mistral, and vLLM for evaluation and inference.
output: {"gpus":[]}

input: We evaluate Qwen2.5-7B, Qwen2.5-14B, Llama3.1-8B, and Meta-Llama3-8B-Instruct on all datasets.
output: {"gpus":[]}

input: We call the Azure Batch REST API service to run GPT-4o requests.
output: {"gpus":[]}

input: We compare LLaVA-Video-7B and CogVideoX-5B on video generation benchmarks.
output: {"gpus":[]}

input:
output: {"gpus":[]}

input: The models are trained using the AdamW optimizer (Kingma and Ba, 2014) on 4 Nvidia V100-32G GPUs for Qwen2-0.5B models and 16 Nvidia V100-32G GPUs for Mistral-7B.
output: {"gpus":[{"gpu_model":"Nvidia V100 32G","count":4},{"gpu_model":"Nvidia V100 32G","count":16}]}

input: Full Mode1</td><td>36h on NVIDIA A100</td></tr><tr><td>Router-Tuning</td><td>Block / MLP / Attn</td><td>Token / Sequence</td><td>Finetuning</td><td>Router</td><td>15m on NVIDIA A6000</td></tr></table>
output: {"gpus":[{"gpu_model":"NVIDIA A6000","count":null},{"gpu_model":"NVIDIA A100","count":null}]}

input: We run all evaluations on 4 NVIDIA A100 GPUs, each with 80 GB of memory.
output: {"gpus":[{"gpu_model":"NVIDIA A100 80GB","count":4}]}

input: the calibration process is completed within 10 minutes using a single RTX 4090 GPU. Regarding efficiency, we evaluate the encoder latency on NVIDIA RTX 4090 and NVIDIA RTX A6000.
output: {"gpus":[{"gpu_model":"RTX 4090","count":1},{"gpu_model":"NVIDIA RTX A6000","count":null}]}

input: All experiments were performed using 512 CPU cores, 8 Nvidia RTX A6000 (48GB) GPUs, and 1024 GB of memory.
output: {"gpus":[{"gpu_model":"Nvidia RTX A6000 48GB","count":8}]}

input: All experiments were conducted on NVIDIA A100 GPUs with 40GB or 80GB memory configurations. Running the full set of main experiments, including all primary tables, required approximately 21 days using 8 GPUs in parallel.
output: {"gpus":[{"gpu_model":"NVIDIA A100 40GB","count":8},{"gpu_model":"NVIDIA A100 80GB","count":8}]}

input: We conduct all experiments on an NVIDIA A40 GPU.
output: {"gpus":[{"gpu_model":"NVIDIA A40","count":1}]}

input: Our experiments use either 8 NVIDIA A100 80GB GPUs or 8 NVIDIA H100 GPUs depending on model size.
output: {"gpus":[{"gpu_model":"NVIDIA A100 80GB","count":8},{"gpu_model":"NVIDIA H100","count":8}]}

input: Unlike Smith et al. (2023), who used 64 NVIDIA V100 GPUs, we train our model on 8 NVIDIA A100 GPUs.
output: {"gpus":[{"gpu_model":"NVIDIA A100","count":8}]}

input: We reproduce the baseline on 4 NVIDIA A100 GPUs and train our method on 8 NVIDIA A100 GPUs.
output: {"gpus":[{"gpu_model":"NVIDIA A100","count":4},{"gpu_model":"NVIDIA A100","count":8}]}

input: The baseline reported results on 8 V100 GPUs, but in this paper we only evaluate on a single NVIDIA A6000 GPU.
output: {"gpus":[{"gpu_model":"NVIDIA A6000","count":1}]}
```
