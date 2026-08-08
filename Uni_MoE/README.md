
# Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts (4 experts)
[Yunxin Li](https://yunxinli.github.io), [Shenyuan Jiang](URL), [Baotian Hu](https://faculty.hitsz.edu.cn/hubaotian), [Longyue Wang](http://www.longyuewang.com/), [Wanqi Zhong](URL), [Wenhan Luo](https://whluo.github.io/), [Lin Ma](https://forestlinma.com/), [Min Zhang](https://faculty.hitsz.edu.cn/MinZhang)

</h4>

<h4> If you appreciate our project, please consider giving us a star ⭐ on GitHub to stay updated with the latest developments.  </h4>

**Usage and License Notices**: The data and checkpoint are intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA and Vicuna. The dataset and models trained using the dataset should not be used outside of research purposes.

If you use the Uni-MoE model with 8 experts, please go `./Uni_MoE_8e`

## 🌟 Structure

The model architecture of Uni-MoE is shown below. Three training stages contain: 1) Utilize pairs from different modalities and languages to build connectors that map these elements to a unified language space, establishing a foundation for multimodal understanding; 2) Develop modality-specific experts using cross-modal data to ensure deep understanding, preparing for a cohesive multi-expert model; 3) Incorporate multiple trained experts into LLMs and refine the unified multimodal model using the LoRA technique on mixed multimodal data.

<div align=center><img src="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/model.png" height="100%" width="75%"/></div>

## 🎨 Case Show


<div align=center><img src="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/case_figure.png" height="100%" width="75%"/></div>


## ⚡️ Install

The following instructions are for Linux installation.
We would like to recommend the requirements as follows.
* Python == 3.9.16
* CUDA Version >= 11.7

1. Clone this repository and navigate to the Uni-MoE folder
```bash
git clone https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs.git
cd UMOE-Scaling-Unified-Multimodal-LLMs/Uni_MoE
```

2. Install Package
```Shell
conda create -n unimoe python==3.9.16
conda activate unimoe
pip install -r env.txt
```

3. Replace all the absolute pathnames '/path/to/' with your specific path to the Uni-MoE file
**(Including all the eval_x.py/inference_x.py/train_mem_x.py/data.py/demo.py files and config.json files from the model weights)**

## ⚡️ Uni-MOE Weights

To use our model, all weights should be downloaded.

After downloading all of them, organize the weights as follows in 'Uni_MoE/checkpoints' folder:
```
└── checkpoints
    ├── Uni-MoE-audio-base
    ├── Uni-MoE-audio-e2
    ├── Uni-MoE-speech-base
    ├── Uni-MoE-speech-e2
    ├── Uni-MoE-speech-base-interval
    ├── Uni-MoE-speech-v1.5
    ├── clip-vit-large-patch14-336
    ├── whisper-small
    └── BEATs_iter3_plus_AS2M.pt
```
| Model  | Checkpoint |
|----------|-----------|
| vision encoder | [CLIP ViT-L/14 336px](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main) |
| speech encoder | [whisper small](https://huggingface.co/openai/whisper-small/tree/main) |
| audio encoder  | [BEATs_iter3+ (AS2M)](https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf) |
| Uni-MoE-audio-base-model | [Uni-MoE/Uni-MoE-audio-base](https://huggingface.co/VictorJsy/Uni-MoE-audio-base/tree/main) |
| Uni-MoE-audio-fine-tuned-chekpoint | [Uni-MoE/Uni-MoE-audio-e2](https://huggingface.co/VictorJsy/Uni-MoE-audio-e2/tree/main) |
| Uni-MoE-speech-base-model | [Uni-MoE/Uni-MoE-speech-base](https://huggingface.co/VictorJsy/Uni-MoE-speech-base/tree/main) |
| Uni-MoE-speech-fine-tuned-chekpoint | [Uni-MoE/Uni-MoE-speech-e2](https://huggingface.co/VictorJsy/Uni-MoE-speech-e2/tree/main) |
| Uni-MoE-speech-base-interval | [Uni-MoE/Uni-MoE-speech-base-interval](https://huggingface.co/VictorJsy/Uni-MoE-speech-base-interval) |
| Uni-MoE-speech-v1.5  | [Uni-MoE/Uni-MoE-speech-v1.5](https://huggingface.co/VictorJsy/Uni-MoE-speech-v1.5) |

* Uni-MoE-speech refers to the MOE-Task2 and Uni-MoE-audio refers to the MOE-Task3 in our paper.
* 'Uni-MoE-base' is the backbone containing LLMs and trained parameters gained from Training Stage 2: Training Modality-Specific Expert.

## 🗝️ Dataset

### Training Data
| DataSet  | Type |
|----------|-----------|
| [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | [image(train2014)](http://images.cocodataset.org/zips/train2014.zip) |
| [Video-Instruct-Dataset](https://github.com/mbzuai-oryx/Video-ChatGPT) | [video(from youtube)](https://www.youtube.com/) |
| [WavCaps](https://huggingface.co/datasets/cvssp/WavCaps/tree/main/json_files) | [audio](https://huggingface.co/datasets/cvssp/WavCaps/tree/main/Zip_files) |
| [AudioCaps](https://audiocaps.github.io/) | [audio(Cap)](https://audiocaps.github.io/) |
| [ClothoAQA](https://zenodo.org/records/6473207)  | [audio(QA)](https://zenodo.org/records/6473207) |
| [ClothoV1](https://zenodo.org/records/3490684) | [audio(Cap)](https://zenodo.org/records/3490684) |
| [MELD](https://affective-meld.github.io/) | [audio(Music)](https://affective-meld.github.io/) |
| [RACE](https://huggingface.co/datasets/race/tree/main) | [Speech(TTS)](https://www.cs.cmu.edu/~glai1/data/race/) |
| [LibriSpeech](https://www.openslr.org/12) | [Speech(Long)](https://www.openslr.org/12) |

We use TTS technical to convert long text to speech to construct long speech understanding data.

### Evaluation Data
| DataSet  | Input Type |
|----------|----------|
| [AOKVQA](https://allenai.org/project/a-okvqa/home) | Text-Image |
| [OKVQA](https://okvqa.allenai.org/) | Text-Image |
| [VQAv2](https://visualqa.org/) | Text-Image |
| [ClothoAQA](https://zenodo.org/records/6473207) | Text-Audio |
| [ClothoV1](https://zenodo.org/records/3490684) | Text-Audio |
| [ClothoV2](https://zenodo.org/records/3490684) | Text-Audio |
| [POPE](https://github.com/RUCAIBox/POPE) | Text-Image |
| [TextVQA](https://textvqa.org/dataset/) | Text-Image |
| [MM-Vet](https://github.com/yuweihao/MM-Vet) | Text-Image |
| [SEEDBench(Image)](https://github.com/ailab-cvc/seed-bench?tab=readme-ov-file) | Text-Image |
| [MMBench](https://mmbench.opencompass.org.cn/home) | Text-Image |
| [MMBench-Audio](https://mmbench.opencompass.org.cn/home) | Text-Image-Speech(Long) |
| [English-High-School-Listening](https://huggingface.co/datasets/VictorJsy/College-Entrance-English-Examination-Listening-Part/tree/main) | Text-Speech(Long) |
| [RACE](https://huggingface.co/datasets/race/tree/main) | Text-Speech(Long) |
| [MSVD](http://www.cs.utexas.edu/users/ml/clamp/videoDescription/) |Text-Video-Audio |
| [Activitynet-QA](https://github.com/MILVLG/activitynet-qa) |Text-Video-Audio |

#### College Entrance English Examination Listening Part

We build a real speech understanding dataset to check the practical long speech recognition capabilities: [English-High-School-Listening](https://huggingface.co/datasets/VictorJsy/College-Entrance-English-Examination-Listening-Part/tree/main)
It comprises 150 questions related to long audio segments with an average length of 109 seconds, and 50 questions about short audio segments with an average length of 14 seconds.

### Experimental Results

<div align=center><img src="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/Uni-MoE-Experiments.png" height="100%" width="90%"/></div>

## 🌈 How to infer and deploy your demo

1. Make sure that all the weights are downloaded and the running environment is set correctly.
2. run inference scripts [`inference_audio.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/inference_audio.sh) and [`inference_speech.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/inference_speech.sh) using ```bash inference_audio.sh``` ```bash inference_speech.sh``` or run the following commands to inference:
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python Uni_MoE_audio/inference_all.py
```
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python Uni_MoE_speech/inference_all.py
```
To launch the online demo ( It is highly recommended to launch the demo with [Uni-MoE-speech-v1.5](https://huggingface.co/VictorJsy/Uni-MoE-speech-v1.5) that need the basic parameters of [Uni-MoE-speech-base-interval](https://huggingface.co/VictorJsy/Uni-MoE-speech-base-interval)), run:
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python demo/demo.py
python demo/app.py
```


## 🌈 How to train and evaluate on datasets

Training:
1. Make sure that all the weights are downloaded and the environment is set correctly, especially for the base model.
2. Our training data can be downloaded from [UMOE-Speech-453k.json](https://huggingface.co/datasets/Uni-MoE/Uni-MoE-Training-Dataset/blob/main/Uni_MoE_Speech.json) and [UMOE-Cap-453k.json](https://huggingface.co/datasets/Uni-MoE/Uni-MoE-Training-Dataset/blob/main/Uni_MoE_Cap.json).
3. Relevant vision and audio files: [Dataset](#Training-Data)
4. Run training scripts: [`finetune_audio.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/finetune_audio.sh) or [`finetune_speech.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/finetune_speech.sh) using ```bash finetune_audio.sh``` ```bash finetune_speech.sh```, remember to modify the training set with your own preference.
5. For multiple GPUs training, run training scripts: [`finetune_speech_dp.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/finetune_speech_dp.sh) using ```bash finetune_speech_dp.sh```, remember to modify the training set with your own preference.

Evaluation:
1. Prepare the evaluation set using the form as [`samples.json`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/data_sample/samples.json).
2. Run evaluation scripts: [`eval_audio.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/eval_audio.sh) or [`eval_speech.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/eval_speech.sh) using ```bash eval_audio.sh``` ```bash eval_speech.sh``` or run the following commands to eval:
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python Uni_MoE_audio/eval.py\
 --data_path /path/to/clotho.json\
 --data_type clothov1\
 --output test.json
```
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python Uni_MoE_speech/eval.py\
 --data_path /path/to/vqa_eval.json\
 --data_type vqa\
 --output test.json
```
We recommend using 80GB GPU RAM to run all experiments.

## Citation

If you find Uni-MoE useful for your research and applications, please cite using this BibTeX:
```bibtex

@ARTICLE{li_unimoe,
  author={Li, Yunxin and Jiang, Shenyuan and Hu, Baotian and Wang, Longyue and Zhong, Wanqi and Luo, Wenhan and Ma, Lin and Zhang, Min},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Uni-MoE: Scaling Unified Multimodal LLMs With Mixture of Experts}, 
  year={2025},
  volume={47},
  number={5},
  pages={3424-3439},
  doi={10.1109/TPAMI.2025.3532688}}


```
