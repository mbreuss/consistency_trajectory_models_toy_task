# cmt_toy_task
Minimal unofficial implementation of Consistency Trajectory models proposed in [paper_link](https://openreview.net/attachment?id=ymjI8feDTD&name=pdf).
---

### Installation

```bash

pip install -e .

```

---

### Consistency Trajectory Models

A new class of generative models close of Diffusion models, that learn to model the trajectory of Probability Flow ODEs directly. Diffusion models learn to predict the denoised action $x_0$ from the noisy action $x_T$ by the current score of the PF ODEs. Consistency Trajectory models learn to predict the trajectory of the ODEs directly and can jump to any point of the trajectory. They can be seen as a generalization of consistency models from Song et al. (2024) and can be trained with the an extended loss function combining score matching objective from diffusion models with a _soft_ consistency loss.

---


### Toy Task Results 

Here are some first results of CMT trained from scratch without a teacher model.


<div style="display:flex">
  <img src="./images/cm_euler_epochs_2000.png" width="40%" />
  <img src="./images/cm_onestep_epochs_2000.png" width="40%" />
</div>
<p style="text-align:center">From left to right: CMT with 10 Euler Steps, Multistep and Single Step prediction with CMT.</p>