# Consistency Trajectory Models Toy Task

Minimal unofficial implementation of Consistency Trajectory models proposed in [paper_link](https://openreview.net/attachment?id=ymjI8feDTD&name=pdf). 
High chance that some implementation errors remain, since everything is implemented from the paper without code examples. 
Please also check out the original code from the authors to run experiments for high-resolution image generation experiments [ctm](https://github.com/sony/ctm).

---

### Installation

```bash
pip install -e .
```

---

### Consistency Trajectory Models
A new class of generative models close of Diffusion models, that learn to model the trajectory of Probability Flow ODEs directly. Diffusion models learn to predict the denoised action $x_0$ from the noisy action $x_T$ by the current score of the PF ODEs. Consistency Trajectory models learn to predict the trajectory of the ODEs directly and can jump to any point of the trajectory. 
They can be seen as a generalization of consistency models from [Song et al. (2023)](https://arxiv.org/pdf/2303.01469.pdf) and can be trained with the an extended loss function combining score matching objective from diffusion models with a _soft_ consistency loss.

<div style="display:flex">
  <img src="./images/Figure_2_CTM.png" width="75%" />
</div>
<p style="text-align:center"> Overview of different model classes w.r.t. Sampling from the ODE</p>


#### Boundary Conditions

```math
G_{\theta}(x_t, t, s):= \frac{s}{t} x_t + ( 1 - \frac{s}{t})g_{\theta}(x_t, t, s)
```
with the standard Karras et al. (2022) preconditioning functions inside $g_{\theta}(x_t, t, s)$.
The CTM models introduce a novel parameter $s$ that defines the target time step of the ODEs. 


#### Training Objective

The original paper proposes the following training objective consisting of a score matching objective and a consistency loss combined with an additional GAN loss. 
However, since we do not wanna use GAIL style training (yet) we only use the score matching objective and the consistency loss.

```math
\mathcal{L}_{\text{Total}} = \lambda_{\text{SM}} \mathcal{L}_{SM} +\lambda_{\text{CTM}} \mathcal{L}_{CTM} + \lambda_{\text{GAN}} \mathcal{L}_{GAN}
```

The score matching objective is defined as follows:

```math
\mathcal{L}_{SM} = \mathbb{E}_{x_0 \sim p_0} \left[ \left\| x_0 - f_{\theta}(x_t, t=t, s=t) \right\|^2 \right]
```

The soft consistency matching loss is defined as:
```math
\mathcal{L}_{CTM} = \mathbb{E}_{t \in [0, T]}\mathbb{E}_{s \in [0, t]} \mathbb{E}_{tu \in (s, t]} \mathbb{E}_{x_0, p_{0t}(x|x_0)} \left[ d(x_{\text{target}}(x,t,u,s), x_{\text{est}}(x,t,s)) \right]
```
where the $d$ refers to a feature distance in the data space and the two estimates are defined
```math
x_{\text{est}}(x_t, t, s) = G_{\text{sg}(\theta)}(G_{\theta}(x_t, t, s), s, 0)
```
and
```math
x_{\text{target}}(x_t, u, s) = G_{\text{sg}(\theta)}(G_{\text{sg}(\theta)}(\text{Solver}(x_t, t, u;\phi), s, 0))
```
The soft consistency loss tries to enforce local consistency and global consistency for the model, which should enable small jumps for ODE-like sovlers and large jumps for few step generation.

For our application the GAN loss is not usable, thus I didnt implement it yet. 


---


### Toy Task Results 

Here are some first results of CMT trained from scratch without a teacher model. Further experiments with teacher models need to be tested. 

<div style="display:flex">
  <img src="./images/cm_euler_epochs_2000.png" width="45%" />
  <img src="./images/cm_onestep_epochs_2000.png" width="45%" />
</div>
<p style="text-align:center">From left to right: CMT with 10 Euler Steps, Multistep and Single Step prediction with CMT.</p>


Right now the model performs significantly better with 10 steps. Working on improving the single step prediction. Lets see whats possible. However, even the main author of the consistency model paper noted in his recent rebuttal [link](https://openreview.net/forum?id=WNzy9bRDvG), that Consistency models work better on high-dimensional datasets. So it can be, that the same counts for CTM. 


--- 


### Lessons learned

...

---

### To Dos

 - [ ] Implement the new sampling method
 - [ ] Add new toy tasks
 - [ ] Compare teacher and student models vs from scratch training
 - [ ] Find good hyperaparmeters
 - [ ] Work on better single step inference
 - [ ] Add more documentation

---

### Acknowledgement

- the model is based on the paper [Consistency Trajectory Models]([https://openreview.net/attachment?id=ymjI8feDTD&name=pdf](https://arxiv.org/pdf/2310.02279.pdf)https://arxiv.org/pdf/2310.02279.pdf) 

If you found this code helpfull please cite the original paper:

```bash
@article{kim2023consistency,
  title={Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion},
  author={Kim, Dongjun and Lai, Chieh-Hsin and Liao, Wei-Hsiang and Murata, Naoki and Takida, Yuhta and Uesaka, Toshimitsu and He, Yutong and Mitsufuji, Yuki and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.02279},
  year={2023}
}

---

```

---
