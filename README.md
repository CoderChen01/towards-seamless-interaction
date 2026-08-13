<p align="center">
  <a href="https://arxiv.org/abs/2512.15340">
    <img src="assets/logo.png" alt="Project Logo" width="220"/>
  </a>
</p>

<h1 align="center">
  🤖✨ Towards Seamless Interaction: Causal Turn-Level Modeling of Interactive 3D Conversational Head Dynamics
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2512.15340">
    <img src="https://img.shields.io/badge/📜_preprint-🗣️🤖_towards_seamless_interaction-b31b1b.svg?style=for-the-badge" />
  </a>
  <a href="./TUTORIAL.md">
    <img src="https://img.shields.io/badge/📖_tutorial-Reproducibility-cafffb.svg?style=for-the-badge" />
  </a>
</p>

<p align="center">
  <strong>
    <a target="_blank" href="https://scholar.google.com/citations?user=q3NWGzUAAAAJ&hl=en&authuser=1">Junjie Chen</a><sup>1,2</sup> ·
    <a target="_blank" href="https://scholar.google.com/citations?user=sdqv6pQAAAAJ&hl=en&authuser=1">Fei Wang</a><sup>1,2</sup> ·
    <a target="_blank" href="https://scholar.google.com/citations?user=odap6UMAAAAJ&hl=en">Zhihao Huang</a><sup>5,6</sup> ·
    <a target="_blank" href="">Qing Zhou</a><sup>8</sup> ·
    <a target="_blank" href="https://scholar.google.com/citations?user=UQ_bInoAAAAJ&hl=en&authuser=1">Kun Li</a><sup>7</sup>
  </strong>
  <br/>
  <strong>
    <a target="_blank" href="https://scholar.google.com/citations?user=DsEONuMAAAAJ&hl=zh-CN">Dan Guo</a><sup>1</sup> ·
    <a target="_blank" href="https://scholar.google.com/citations?user=AK9VF30AAAAJ&hl=en&authuser=1">Linfeng Zhang</a><sup>4</sup> ·
    <a target="_blank" href="https://scholar.google.com/citations?user=ro8lzsUAAAAJ&hl=en">Xun Yang</a><sup>3</sup>
  </strong>
  <p align="center">
  <sup>1</sup> Hefei University of Technology &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>2</sup> IAI, Hefei Comprehensive National Science Center <br/>
  <sup>3</sup> USTC &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>4</sup> SJTU &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>5</sup> TeleAI, China Telecom &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>6</sup> Northwestern Polytechnical University<br/>
  <sup>7</sup> United Arab Emirates University &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>8</sup> Anhui Polytechnic University
  </p>
</p>

<br/>

## 📌 Open-Source Roadmap

- [x] Core source code release  
- [x] Pretrained checkpoints (CKPT)  
- [x] Usage documentation and tutorials  
- [x] Rendering tools
- [ ] Integrate with 3DGS projects (e.g., [LAM](https://github.com/aigc3d/LAM))

## 🔥 Highlights

- 🧠 **Causal turn-level formulation** for streaming conversational generation  
- 🔄 **Unified talking & listening modeling** within a single framework  
- 🎧🗣️ **Interleaved multimodal tokens** from both interlocutors  
- 🌊 **Diffusion-based 3D head decoding** for expressive and stochastic motion  
- 📉 **15–30% error reduction** over strong baselines (e.g., DualTalk)

## 🚀 Overview

Human conversation is a continuous exchange of **speech and nonverbal cues**—including head nods, gaze shifts, and subtle expressions.  
Most existing approaches, however, treat **talking-head** and **listening-head** generation as *separate problems*, or rely on *non-causal full-sequence modeling* that is unsuitable for real-time interaction.

We propose a **causal, turn-level framework** for interactive 3D conversational head generation.  
Our method models dialogue as a sequence of **causally linked turns**, where each turn accumulates multimodal context from both participants to produce **coherent, responsive, and humanlike 3D head dynamics**.

<p align="center">
  <img src="assets/overview.svg" alt="Framework Overview" width="90%"/>
</p>

## 🧩 Method: TIMAR

**TIMAR (Turn-level Interleaved Masked AutoRegression)** is the core method proposed in this work.

### 🧱 Key Idea

- Represent conversation as **interleaved audio–visual tokens**:
  - 👤 User speech + user head motion  
  - 🤖 Agent speech + agent head motion  
- Perform:
  - 🔁 **Bidirectional fusion within each turn** (intra-turn alignment)  
  - ⏱️ **Strictly causal reasoning across turns** (inter-turn dependency)

This design mirrors how humans coordinate speaking and listening over time.

### ⚙️ Architecture

<p align="center">
  <img src="assets/method.svg" alt="TIMAR Architecture" width="90%"/>
</p>

**Core components:**
- 🧠 **Turn-Level Causal Attention (TLCA)**  
  - Bidirectional attention inside a turn  
  - Causal masking across turns (no future leakage)  
- 🌊 **Lightweight Diffusion Head**  
  - Predicts continuous 3D head motion  
  - Captures expressive stochasticity beyond deterministic regression  

## 🧪 Experiments

We evaluate our framework on the **interactive 3D conversational head benchmark**, following the DualTalk protocol.

### 📊 Quantitative Results

<details>
<summary>Click to see the results</summary>
<p align="center">
  <img src="assets/quant_results.png" alt="Quantitative Results" width="90%"/>
</p>
</details>

**Results at a glance:**
- ⬇️ **15–30% reduction** in Frechet Distance (FD) and MSE  
- 📈 Improved expressiveness and synchronization (SID ↑)  
- 🌍 Strong generalization on **out-of-distribution conversations**

### 🎭 Qualitative Results

<details>
<summary>Click to see the results</summary>
<p align="center">
  <img src="assets/qual_results.png" alt="Qualitative Results" width="90%"/>
</p>

| Demo | Preview |
|------|---------|
| Demo&nbsp;1 | <video src="https://github.com/user-attachments/assets/f221fc7f-9171-4abc-a13b-c83469e8b731" controls width="320"></video> |
| Demo&nbsp;2 | <video src="https://github.com/user-attachments/assets/dc7cbdae-1be1-492e-ac9e-55ecc27fa35d" controls width="320"></video> |
| Demo&nbsp;3 | <video src="https://github.com/user-attachments/assets/d8b536fb-14e4-4121-ab76-33ebf159e9f2" controls width="320"></video> |

> **Notation**
> 
> - **Agent GT** denotes the ground-truth 3D head motion.  
> - **TIMAR Agent** denotes our generated results.  
> - **DualTalk Agent** denotes the outputs from the DualTalk baseline.
</details>

TIMAR produces:
- Natural listening behavior when the agent is silent  
- Context-aware reactions with longer conversational history  
- Smoother and more stable 3D head motion  

### 🧩 Ablation Studies

<details>
<summary>Click to see the results</summary>
<p align="center">
  <img src="assets/ablation-1.png" alt="Ablation Studies" width="90%"/>
  <img src="assets/ablation-2.png" alt="Ablation Studies" width="90%"/>
</p>
</details>

We analyze the contribution of each design choice:
- ❌ MLP head vs 🌊 diffusion-based head  
- ❌ Full bidirectional attention vs ✅ turn-level causal attention  
- ❌ Encoder–decoder vs ✅ encoder-only backbone  

Each component is critical for causal coherence and generalization.


## 📚 Citation

If you find this work useful, please consider citing:

```bibtex
@article{chen2025timar,
  title={Towards Seamless Interaction: Causal Turn-Level Modeling of Interactive 3D Conversational Head Dynamics},
  author={Chen, Junjie and Wang, Fei and Hunag, Zhihao and Zhou, Qing and Li, Kun and Guo, Dan and Zhang, Linfeng and Yang, Xun},
  journal={arXiv preprint arXiv:2512.15340},
  year={2025}
}
```

## 🙏 Acknowledgements

Our implementation benefits from the publicly available codebases of [MAR](https://github.com/LTH14/mar) and [DualTalk](https://github.com/ZiqiaoPeng/DualTalk). We thank the authors for releasing their repositories, which provided valuable references for implementation details and experimental protocols.

We further acknowledge the broader research community for foundational advances in autoregressive diffusion modeling, multimodal representation learning, and interactive 3D conversational head generation. While our framework introduces a distinct turn-level causal formulation and interleaved modeling strategy, it builds upon and extends these prior technical developments.

