# Consistency Trajectory Models Toy Task


Minimal unofficial implementation of Consistency Trajectory models proposed in [paper_link](https://openreview.net/attachment?id=ymjI8feDTD&name=pdf).

---

### Installation

```bash
pip install -e .
```

---

### Consistency Trajectory Models

A new class of generative models close of Diffusion models, that learn to model the trajectory of Probability Flow ODEs directly. Diffusion models learn to predict the denoised action $x_0$ from the noisy action $x_T$ by the current score of the PF ODEs. Consistency Trajectory models learn to predict the trajectory of the ODEs directly and can jump to any point of the trajectory. They can be seen as a generalization of consistency models from Song et al. (2024) and can be trained with the an extended loss function combining score matching objective from diffusion models with a _soft_ consistency loss.


#### Boundary Conditions

![Boundary Conditions](https://quicklatex.com/cache3/e9/ql_063cfd6a53d0cc2db965a5efded914e9_l3.png)
with the standard Karras et al. (2022) preconditioning functions inside $g_{\theta}(x_t, t, s)$.
The CTM models introduce a novel parameter $s$ that defines the target time step of the ODEs. 


#### Training Objective

The original paper proposes the following training objective consisting of a score matching objective and a consistency loss combined with an additional GAN loss. 
However, since we do not wanna use GAIL style training (yet) we only use the score matching objective and the consistency loss.

```math
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
```

![Training Objective](https://quicklatex.com/latex3.f/ql_2b646f_5c25cd7f9058c9e928f9a70b9f8cf678_l3.png)


The score matching objective is defined as follows:

![Score Matching Objective](https://quicklatex.com/latex3.f/ql_2b646f_5c25cd7f9058c9e928f9a70b9f8cf678_l3.png)

The consistency loss is defined as follows:

![Consistency Loss](https://quicklatex.com/cache3/ea/ql_85df736cf74762caed890207ebb787ea_l3.png)
where the $d$ refers to a feature distance in the data space and the two estimates are defined
![X_est](https://quicklatex.com/cache3/0d/ql_12316b65a84b503821093518c606a70d_l3.png)
and 
![X_target](https://quicklatex.com/cache3/42/ql_8edf2b45943254b44344eec5ac031342_l3.png)


---


### Toy Task Results 

Here are some first results of CMT trained from scratch without a teacher model.


<div style="display:flex">
  <img src="./images/cm_euler_epochs_2000.png" width="45%" />
  <img src="./images/cm_onestep_epochs_2000.png" width="45%" />
</div>
<p style="text-align:center">From left to right: CMT with 10 Euler Steps, Multistep and Single Step prediction with CMT.</p>


Right now the model performs significantly better with 10 steps. Working on improving the single step prediction. Lets see whats possible. 


--- 


### Lessons learned


None so far :D 


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

- the model is based on the paper [Consistency Trajectory Models](https://openreview.net/attachment?id=ymjI8feDTD&name=pdf)
