# Boundary Examples

Use these as label-boundary examples. Match the evidence pattern, not surface words.

## Example 1
Window: In each pair programming homework, the two students were in different physical locations and collaborated remotely using ZOOM to video chat and Google Colab to share their code.
Why: Collaboration tooling and coursework logistics do not show compute used for this paper's experiments.
Labels: ["L0_NO_SPECIFIC_RESOURCE"]

## Example 2
Window: Code for recreating the figures can be found at https://github.com/YutongWangML/neurips2024-multiclass-IR-figures. The code can be run on Google Colab with a CPU runtime in under one hour.
Why: Reproducibility capability is not author-side use of Colab/cloud compute.
Labels: ["L0_NO_SPECIFIC_RESOURCE"]

## Example 3
Window: The number of frames being processed per second (FPS) is computed in a single Quadro RTX 6000 GPU (24G memory).
Why: A specific local GPU is used for the paper's measurement.
Labels: ["L1_INTERNAL"]

## Example 4
Window: The bold results mean the best ones using the same pretrained model. We use the 50-step, 512 batch size experiment on an RTX-3090 to test the computational cost and the column time is the average computational cost per step in seconds.
Why: The window has both author-side GPU measurement and use of a pretrained model artifact.
Labels: ["L1_INTERNAL","L5_OPEN_MODEL"]

## Example 5
Window: The first layer of the CNN, the GRU, and the attentive-GRU model is the 300-dimensional word embedding that is initialized by using the vectors pre-trained on Google News dataset. Handcrafted features are added in one-hot representation.
Why: Pre-trained vectors initialize the model used in the current study.
Labels: ["L5_OPEN_MODEL"]

## Example 6
Window: For the BERT model, we use a base learning rate 3e-5 for 10 epochs with the same learning schedule described in Devlin et al. For training our model, we use a learning rate 1e-4 for pre-training the second-stage transformer.
Why: BERT is an existing reusable model artifact used in the current experiment.
Labels: ["L5_OPEN_MODEL"]

## Example 7
Window: Comparison with US-Net. We first compare our framework with US-Net on MobileNet v1 and MobileNet v2 backbones. The Accuracy-FLOPs curves are shown in Fig. 4.
Why: Existing backbone models are used as experimental components or comparison targets.
Labels: ["L5_OPEN_MODEL"]

## Example 8
Window: Generation of joint proposals. We apply the Faster R-CNN detector to produce human detection boxes, and perform a NMS procedure with detection score threshold 0.6.
Why: A reusable detector model is applied in the current method.
Labels: ["L5_OPEN_MODEL"]

## Example 9
Window: Table 5 reports the number of trainable parameters and inference speed of different models on ACE05 and SemEval. All models are performed on an NVidia Tesla V100 GPU for training and test. Models include BERT-large, +GCN, and +A-GCN.
Why: The table reports this paper's experiments with a local GPU and a reusable BERT model.
Labels: ["L1_INTERNAL","L5_OPEN_MODEL"]

## Example 10
Window: ChatGPT: We use the following instruction to prompt ChatGPT. FlanT5: We use the instruction template for the AI2 Reasoning Challenge to obtain answers from FlanT5.
Why: ChatGPT is hosted/API-style use, while FlanT5 is an open pretrained model artifact used in the same study.
Labels: ["L4_EXTERNAL_API","L5_OPEN_MODEL"]

## Example 11
Window: The results are averaged on 100 samples and run on 4 NVIDIA A6000 GPUs for open-source models. Gemini uses API token estimation.
Why: The window has local GPUs, open-source model use, and hosted Gemini/API evidence.
Labels: ["L1_INTERNAL","L4_EXTERNAL_API","L5_OPEN_MODEL"]

## Example 12
Window: We start from bert-base-uncased vocabulary to collect words. The maximum sequence length is set to 512 and the BERT-base vocab with size 30,522 is used.
Why: Vocabulary or tokenizer-only preprocessing is not reuse of the BERT model artifact.
Labels: ["L0_NO_SPECIFIC_RESOURCE"]

## Example 13
Window: Large language models such as ChatGPT, LLaMA, and GPT-4 have exhibited transformative abilities to generate fluent long-form text in response to user queries.
Why: Background model-name mentions do not show author-side use in this paper.
Labels: ["L0_NO_SPECIFIC_RESOURCE"]

## Example 14
Window: All of these models are designed to solve downstream natural language tasks. Table 4 lists the paths for accessing the models via Hugging Face. We use a GPU Tesla V100-PCIE-32GB.
Why: Hugging Face paths alone do not show model artifact reuse, but the local GPU is concrete compute.
Labels: ["L1_INTERNAL"]

## Example 15
Window: XLNet adopts the Transformer XL as its base architecture. XLNet pretrained the language model on a large corpus and fine-tuned the pretrained model on downstream tasks. It remains unexplored for this application.
Why: This is background about XLNet rather than author-side use in the current study.
Labels: ["L0_NO_SPECIFIC_RESOURCE"]
