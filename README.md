
<p align="center">
    <img src="https://github.com/HITsz-TMG/Uni-MoE/blob/master/assets/uni_moe_logo.png" width="250" style="margin-bottom: 0.2;"/>
<p>
<h4 align="center">

🚀 Welcome to the repo of **Uni-MOE**

Uni-MoE is a MoE-based omnimodal large model and can understand and generate omnimodalities.


[![🤗Hugging Face](https://img.shields.io/badge/🤗Hugging_Face-UniMoE2.0-yellow)](https://huggingface.co/collections/HIT-TMG/lychee-uni-moe-20)
[![Project Page](https://img.shields.io/badge/Project_Page-UniMoE2.0-blue)](https://idealistxy.github.io/Uni-MoE-v2.github.io/)
[![Demo](https://img.shields.io/badge/Demo-UniMoE2.0-orange)](https://github.com/HITsz-TMG/Uni-MoE) 
[![Paper](https://img.shields.io/badge/Arxiv-UniMoE2.0-yellow)](https://arxiv.org/abs/2511.12609)

[![🤗Hugging Face](https://img.shields.io/badge/🤗Hugging_Face-UniMoE_Audio-yellow)](https://huggingface.co/foggyforest/UniMoE-Audio-preview)
[![Project Page](https://img.shields.io/badge/Project_Page-UniMoE_Audio-blue)](https://mukioxun.github.io/Uni-MoE-site/home.html)
[![Demo](https://img.shields.io/badge/Demo-UniMoE_Audio-orange)](https://github.com/HITsz-TMG/Uni-MoE) 
[![Paper](https://img.shields.io/badge/Arxiv-UniMoE_Audio-yellow)](https://arxiv.org/abs/2510.13344)

[![🤗Hugging Face](https://img.shields.io/badge/🤗Hugging_Face-UniMoE-yellow)](https://huggingface.co/Uni-MoE)
[![Project Page](https://img.shields.io/badge/Project_Page-UniMoE-blue)](https://uni-moe.github.io/)
[![Demo](https://img.shields.io/badge/Demo-UniMoE-orange)](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/tree/master?tab=readme-ov-file#-demo-video) 
[![Paper](https://img.shields.io/badge/Arxiv-UniMoE-yellow)](https://arxiv.org/abs/2405.11273)

[![](https://trendshift.io/api/badge/repositories/10407)](https://trendshift.io/repositories/10407)

</h4>

<h4 align="center"> If you appreciate our project, please consider giving us a star ⭐ on GitHub to stay updated with the latest developments.  </h4>



## 🔥 News

- [2025/11/24]  🔥 We have integrated our model Uni-MoE-2.0-Omni for evaluation within the Lmms-eval framework, see [here](https://github.com/HITsz-TMG/Uni-MoE/tree/master/Uni-MoE-2#evaluation).

- [2025/11/13] 🔥 We release the second version of [Uni-MoE-2.0-Omni](https://github.com/HITsz-TMG/Uni-MoE/tree/master/Uni-MoE-2). It achieves a significant leap in language-centric multimodal understanding, reasoning, and generation capabilities, while efficiently supporting cross-modal interactions across ten-plus modalities such as images, text, and speech through its dynamic MoE architecture and progressive training strategy.
  
- [2025/10/16] 🔥 We release a better [Uni-MoE-Audio](https://github.com/HITsz-TMG/Uni-MoE/tree/master/UniMoE-Audio), the first audio generation model with a unified speech and music generation.

- [2025/8/6] 🔥 We release a better Uni-MoE v1.5 at modelscope [here](https://www.modelscope.cn/models/victorjsyy/Uni-MoE) with a unified speech encoding approach.

- [2025/2/20]  🔥 Our paper has been accepted by **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**, 2025.

- [2024/8/28] 🔥 We release our video evaluation benchmark [VideoVista](https://videovista.github.io/) and the automatically generated video instruction tuning data [VideoVista-Train](https://huggingface.co/datasets/Uni-MoE/VideoVista_Train).

- [2024/5/31] 🔥 The checkpoint of Uni-MoE with 8 experts is now available for downloading and inference. For more details, please refer to the [Uni_MoE_8e](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/Uni_MoE_8e/README.md#%EF%B8%8F-uni-moe-weights) table. 
- [2024/4/28] 🔥 We have upgraded the Uni-MoE codebase to facilitate training across multiple Nodes and GPUs. Explore this enhanced functionality in our revamped [fine-tuning script](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/finetune_speech_dp.sh). Additionally, we have introduced a version that integrates distributed MoE modules. This enhancement allows for training our model with parallel processing at both the expert and modality levels, enhancing efficiency and scalability. For more details, please refer to the [Uni_MoE_1.0 with 8 experts](https://github.com/HITsz-TMG/Uni-MoE/tree/master/Uni_MoE/Uni_MoE_8e) documentation. 


## 📀 Demo Video

### 👀 Uni-MoE-2.0-Omni

https://github.com/user-attachments/assets/5e5ca44b-a39f-49bf-afca-78c73b7657ed

### 👀 Uni-MoE-Audio


<div align="center">
    <video src="https://private-user-images.githubusercontent.com/147797956/503537846-e553ef43-5494-4e76-8579-5d8b2955fa8d.mp4" width="100%" style="margin: 0; padding: 0;" controls>
    </video>
</div>

### 👀 Uni-MoE 1.0

Demo 2 contains the real-time understanding of speech (Starting from 30S).

https://private-user-images.githubusercontent.com/45393746/331798338-dfc848a2-1fd2-4f8d-9274-f21f7118ecd9.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTYwMzUwOTUsIm5iZiI6MTcxNjAzNDc5NSwicGF0aCI6Ii80NTM5Mzc0Ni8zMzE3OTgzMzgtZGZjODQ4YTItMWZkMi00ZjhkLTkyNzQtZjIxZjcxMThlY2Q5Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA1MTglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNTE4VDEyMTk1NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTYzYTNmZDNlM2FhOGE3MmM1MzM0Mzk4YTdlYTg3NTgzOTBmNzMyMjM4OTljYTA0ODQ0YmEzZDVlYmFhOWUwMzUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.vrqxhusaq3J_ULQKbeGOxEJH3wry6GjXLxwrFrP0jao

https://private-user-images.githubusercontent.com/45393746/331798343-fcd3eb7e-3dfa-4470-a2e6-b9b140efe0fa.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTYwMzUyNTEsIm5iZiI6MTcxNjAzNDk1MSwicGF0aCI6Ii80NTM5Mzc0Ni8zMzE3OTgzNDMtZmNkM2ViN2UtM2RmYS00NDcwLWEyZTYtYjliMTQwZWZlMGZhLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA1MTglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNTE4VDEyMjIzMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTIyNWU5OTM0NjM1MTgzMWIxNWI4MDllYzU5NWNlOTUxMGI1NzQ5MzkyNmRlNDFlMTY0YzYzMTJmZjk4ZjJmMWEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.Uz3PBfKbEjl5ZOfUSXrAaQQLrvKwCFK2uNPTjtKG3dU


## 🌟 Model Structure


### 🚀 Uni-MoE 2.0

We present Uni-MoE 2.0 from the Lychee family. As a fully open-source omnimodal large model (OLM), it substantially advances the capabilities of Lychee's Uni-MoE series in language-centric multimodal understanding, reasoning, and generating. Based on the Qwen2.5-7B dense architecture, we train Uni-MoE 2.0 from scratch through three core contributions: dynamic-capacity Mixture-of-Experts (MoE) design, a progressive training strategy enhanced with an iterative reinforcement strategy, and a carefully curated multimodal data matching technique. Uni-MoE 2.0 is capable of cross- and tri-modality understanding, as well as generating images, text, and speech.

<div align=center><img src="https://github.com/HITsz-TMG/Uni-MoE/blob/master/Uni-MoE-2/assets/images/architecture.png" height="100%" width="75%"/></div>



### 🚀 Uni-MoE-Audio

Uni-MoE-Audio introduces a dynamic-capacity routing mechanism based on Top-P sampling for adaptive expert allocation, together with a hybrid expert design that separates domain-specific computation (dynamic experts) from universal representations (shared experts). To address data imbalance and task conflicts, Uni-MoE-Audio adopts a structured three-stage training curriculum. From voice cloning and text-to-speech (TTS) to text-to-music (T2M) and video-to-music (V2M), Uni-MoE-Audio supports diverse creative workflows. Extensive experiments confirm its state-of-the-art performance and superior cross-task synergy, paving the way toward universal audio generation.

<div align=center><img src="https://github.com/HITsz-TMG/Uni-MoE/blob/master/UniMoE-Audio/assets/img/AudioLLM_model-MoE.png" height="100%" width="75%"/></div>

### 🚀 Uni-MoE 1.0
The model architecture of Uni-MoE is shown below. Three training stages contain: 1) Utilize pairs from different modalities and languages to build connectors that map these elements to a unified language space, establishing a foundation for multimodal understanding; 2) Develop modality-specific experts using cross-modal data to ensure deep understanding, preparing for a cohesive multi-expert model; 3) Incorporate multiple trained experts into LLMs and refine the unified multimodal model using the LoRA technique on mixed multimodal data.

<div align=center><img src="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/model.png" height="100%" width="75%"/></div>


## 🙏 Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs&type=Date)](https://star-history.dera.page/#HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs&type=Date)


## ❤️ Citation

If you find Uni-MoE useful for your research and applications, please cite using this BibTeX:

```bibtex

@article{li2025uni2omni,
  title={Uni-MoE-2.0-Omni: Scaling Language-Centric Omnimodal Large Model with Advanced MoE, Training and Data},
  author={Li, Yunxin and Chen, Xinyu and Jiang, Shenyuan and Shi, Haoyuan and Liu, Zhenyu and Zhang, Xuanyu and Deng, Nanhao and Xu, Zhenran and Ma, Yicheng and Zhang, Meishan and others},
  journal={arXiv preprint arXiv:2511.12609},
  year={2025}
}
```


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

```bibtex
@article{liu2025unimoe,
  title={UniMoE-Audio: Unified Speech and Music Generation with Dynamic-Capacity MoE},
  author={Liu, Zhenyu and Li, Yunxin and Zhang, Xuanyu and Teng, Qixun and Jiang, Shenyuan and Chen, Xinyu and Shi, Haoyuan and Li, Jinchao and Wang, Qi and Chen, Haolan and others},
  journal={arXiv preprint arXiv:2510.13344},
  year={2025}
}
```
