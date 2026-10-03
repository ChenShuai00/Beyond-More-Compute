L1_INTERNAL:
Authors used local, institutional, private, lab, or otherwise self-managed hardware.
Positive evidence: GPU/CPU servers, workstation, local/internal/in-house cluster, NVIDIA/AMD hardware, A100, V100, H100, RTX, Tesla, TITAN, K80, CPU nodes, mobile CPU/GPU, or phrases like "trained on 8 GPUs".
Count L1 when the window says the authors actually ran, trained, fine-tuned, evaluated, rendered, benchmarked, measured, or implemented the current study on specific hardware, even if the evidence appears in runtime, memory, ablation, table, or appendix text. Examples: "using the same GPU devices", "training on the TITAN V GPU", "NVIDIA A100-SXM4-80GB GPUs", "on CPU", or "on a mobile GPU".
Author-side runtime or memory measurements count as L1 when they report the current study's experiments on GPU/per-GPU/CPU hardware: examples include "GPU pre-training accuracy and speed ablations", "GPU parallel training", "wall-clock computation times using the same GPU devices", "memory usage on each GPU", "GPU runtime memory requirements during training", or "a GPU implementation of our/baseline method" when it is part of the authors' measurement.
Hardware grants or donations count as L1 only when the window explicitly says the provided GPUs were used for this work, or that experiments ran on donated/provided GPUs. Do not count a bare "NVIDIA GPU grant", "NVIDIA Corporation GPU grant", "Facebook GPU donation", "NVIDIA support", or company gift as L1 without explicit experiment/use linkage.
Bare hardware names imply L1 when tied to author-side execution or measurement, unless the same evidence places the hardware in cloud or public HPC.
TPU context overrides bare hardware names: any TPU evidence, including bare TPU chips/cores/pods, TPU v2/v3/v4, TPU-v2/TPU-v3/TPU-v4, Google TPU, Cloud TPU, GCP TPU, or Colab TPU, is L2. AWS/GCP/Azure/Colab resources are L2, and named shared supercomputing/HPC facilities are L3, unless a separate local/private resource is also stated.
Do not add L1 for training/fine-tuning/evaluation, batch size, runtime, throughput, latency, CPU time, GPU memory, GPU hours, out-of-memory, architecture, training recipe, or "compute resources" unless the same window identifies a concrete non-cloud, non-public-HPC hardware resource used by the authors.
Do not add L1 for generic constraints or capability claims such as "fits in GPU memory", "GPU efficiency", "compiled to a CPU or GPU executable", "can run on a GPU", "GPU implementation" as only a hypothetical capability, or figure/table captions about per-GPU memory, unless they are explicitly tied to the authors' own experiment setup or measurements on specific hardware.
Do not add L1 for named public/shared facilities even if they mention GPU clusters or NVIDIA hardware. Examples include USTC Supercomputing Center, MCC Lab/USTC GPU cluster, Mila compute/shared resources, Compute Canada, Calcul Quebec, Digital Research Alliance of Canada/DRA, NERSC, ABCI, Beluga, LUMI, JUWELS, EuroHPC, DiRAC, TACC, ACCESS/XSEDE, university supercomputing centers, national labs, or similar shared facilities. These support L3 only unless a separate local/private hardware resource is also stated.

L2_CLOUD:

Authors used commercial cloud, rented cloud compute, or TPU compute.

Positive evidence: any TPU evidence including TPU chips/cores/pods, TPU v2/v3/v4, TPU-v2/TPU-v3/TPU-v4, Google TPU, Cloud TPU, GCP TPU, or Colab TPU; AWS, EC2, SageMaker compute/training jobs, Google Cloud, GCP, Google Colab/Colab Pro used for the authors' experiments, Vertex AI compute/training jobs, Azure/Azure ML compute, RunPod, CoreWeave, Lambda Labs, cloud instance/server/GPU, rented cloud compute, or cloud storage used in the current study.

Explicit AWS/Google/Azure or other commercial-cloud support counts as L2 when the window links the provider to cloud computing resources, cloud-compute credits, cloud platform access, cloud infrastructure, or cloud computation for the current paper, even if the evidence appears in acknowledgements. The wording does not need to say that the credits were directly "used to train" or that "experiments ran on" the cloud, as long as the acknowledgement explicitly identifies cloud computing resources, cloud credits, platform access, or cloud infrastructure support for this work.

Positive examples include: "Cloud computing resources are provided by AWS Cloud Credits for Research", "Google Cloud provided us access to their platform through the Research Credits Application program", "cloud computation resources from Microsoft Azure for Research", "donation of cloud computing credits through the UW Azure Cloud Computing Credits for Research program", "used Amazon Activate EC2 credits to train", "experiments ran on AWS cloud credits", or "training used Google Cloud credits".

Generic sponsor lists or acknowledgements that merely mention AWS/Google/Azure support, awards, grants, sponsorship, or cloud programs do not count unless they explicitly mention cloud computing resources, cloud-compute credits, cloud platform access, cloud infrastructure, or cloud computation for the current paper.

L3_PUBLIC_HPC:
Authors used a public, national, university, or shared supercomputing/HPC facility.
Positive evidence: supercomputing center, national supercomputer, public HPC center, NERSC, XSEDE/ACCESS, PRACE, Compute Canada, Calcul Quebec, Digital Research Alliance of Canada/DRA, ARCHER, JUWELS, ABCI, Beluga, LUMI, EuroHPC, DiRAC, Texas Advanced Computing Center/TACC, Mila compute/shared resources, USTC Supercomputing Center, MCC Lab/USTC GPU cluster, or named university supercomputing centers.
Public/shared HPC hardware is L3, not L1, unless separate local/private hardware is also stated.
Do not infer HPC from ambiguous acronyms such as TACC meaning test accuracy.

L4_EXTERNAL_API:
Authors used an externally hosted model, API, endpoint, web service, online tool, or remote service that performs computation outside their environment.
Positive evidence: OpenAI API, GPT-4 API, ChatGPT API, Claude API, Gemini API, Hugging Face Inference API, Replicate API, hosted endpoint, web service, online service, or verbs like queried, called, accessed, submitted prompts to, generated with, judged by, evaluated with, revised with, obtained responses from.
Closed/proprietary hosted models such as GPT-4, GPT-3.5, ChatGPT, Claude, Gemini, GPT-4o, GPT-4o-mini, o1, DALL-E, Midjourney, Runway, Kling, Luma, Pika, PixVerse, Vidu, or commercial Google/Azure/OpenAI services are L4 when they appear as part of this paper's method, experiment, baseline, judge, generator, evaluator, scorer, annotator, prompted model, result table row/column, or benchmark comparison target. Do not require the words API, query, or call when the window otherwise shows author-side experimental use.
Cloud-branded hosted APIs such as Google Cloud Vision API, AWS Rekognition, Google Cloud AutoML API, GCP Gemini API, Amazon Comprehend, Azure Face, Azure OpenAI API, and AWS Bedrock are L4 when the window shows author-side use.
A model name alone is not L4. Pure related work, background discussion, future use, docs links, token-price/cost tables, generic acknowledgements, and access-program thanks are not enough without author-use evidence.
OpenAI Researcher Access Program or similar hosted-model access programs usually support L4 when the acknowledgement says the access/contribution made this work possible or otherwise implies access to hosted OpenAI models/APIs for the study.

L5_OPEN_MODEL:
Authors used an existing reusable model artifact in the current study. L5 is about author-side reuse of a model resource, not merely mentioning a model, architecture, method name, repository, license, or citation.

Add L5 only when BOTH conditions are supported:

1. Reusable artifact evidence: the window identifies a pretrained/pre-trained, open-source/open-weight, public-checkpoint/public-weights, Hugging Face/model-zoo/official-checkpoint, foundation/base model, backbone, encoder, feature extractor, embedding model, retriever/reranker, detector, pseudo-labeler, or loaded/initialized/frozen checkpoint/weights. Named models such as BERT, RoBERTa, T5, BART, SBERT, CLIP, OpenCLIP, Stable Diffusion, LayoutLM, Wav2Vec2, Llama, Qwen, Gemma, Mistral/Mixtral, BLOOM/BLOOMZ, Phi, Aya, or DeepSeek count only when the window indicates reusable pretrained/open/public/local model use.
2. Author-side current-study use evidence: the window shows the authors used the artifact in this paper’s method, experiments, evaluation, comparison, ablation, feature extraction, inference, fine-tuning, initialization, prompting, judging, pseudo-labeling, retrieval/reranking, benchmarking, or result reporting. Use signals include used, reused, adopted, employed, loaded, initialized from, fine-tuned, frozen, encoded with, extracted features with, ran, evaluated, benchmarked, prompted, compared against, or used as a backbone/encoder/baseline/comparison target.

Count L5 for table/figure-only evidence when the table or figure reports this paper’s experiments, ablations, benchmarks, comparisons, or results, and an L5 artifact appears as a row, column, baseline, backbone, encoder, judged model, prompted model, or comparison target.

Do NOT add L5 for model names alone; related work; background; citations; future or hypothetical use; capability claims; license-only, repository-only, code/toolkit-only, or Hugging Face path-only mentions; reproduced prior-result tables; datasets generated by someone else’s model; architecture-only baselines; vocabulary-only use; or tokenizer-only preprocessing without linked model/checkpoint/encoder/fine-tuning/inference/evaluation evidence.

Architecture names such as ResNet, VGG, AlexNet, LeNet, ViT, Transformer, CNN, LSTM, U-Net, StyleGAN, PixelCNN, MobileNet, Inception, Xception, FastText, DDIM, PF, or TRADES are not L5 unless the same window shows pretrained/open/public/off-the-shelf/model-zoo/official/reused weights or clear author-side reuse of a model artifact.

Closed or proprietary hosted systems such as GPT-4, ChatGPT, Claude, Gemini, GPT-4o, o1, DALL-E, Midjourney, Runway, Kling, Luma, Pika, PixVerse, or Vidu are L4, not L5. If hosted/API use and open/pretrained model reuse both appear, output both L4 and L5. For DeepSeek-like cases, add L5 only with open-source/open-weight/public-checkpoint/local model evidence; otherwise use L4 for hosted API/service access.

L0_NO_SPECIFIC_RESOURCE:
Use when no L1-L5 label is supported.
Includes compute-related but non-specific evidence: training time, inference cost, computational budget, runtime, large-scale experiments, distributed/parallel training, acceleration, memory, latency, throughput, or "compute resources" without identifying the resource type.
Also includes ordinary no-compute windows: metrics only, dataset descriptions, citations, equations, definitions, method names, ambiguous abbreviations, and general performance claims without specific resource evidence.
