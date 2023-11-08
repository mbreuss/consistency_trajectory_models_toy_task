import math 
from functools import partial
import copy 

import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import einops

from .utils import *
from .models.ctm_mlp import ConsistencyTrajectoryNetwork


def ema_eval_wrapper(func):
    def wrapper(self, *args, **kwargs):
        # Swap model parameters with EMA parameters
        model = self.model

        self.model = self.target_model
        
        # Call the original function
        result = func(self, *args, **kwargs)
        
        # Swap the parameters back to the original model
        self.model = model
        return result
    return wrapper


def ema_diffusion_eval_wrapper(func):
    def wrapper(self, *args, **kwargs):
        # Swap model parameters with EMA parameters
        model = self.model
        self.model = self.teacher_model
        # Call the original function
        result = func(self, *args, **kwargs)
        
        # Swap the parameters back to the original model
        self.model = model
        return result
    return wrapper


class ConsistencyTrajectoryModel(nn.Module):

    def __init__(
            self, 
            sampler_type: str,
            sigma_data: float,
            sigma_min: float,
            sigma_max: float,
            conditioned: bool,
            device: str,
            use_teacher: bool = False,
            n_discrete_t: int = 20,
            lr: float = 1e-4,
            rho: int = 7,
            cmt_lambda: float = 1.0,
            diffusion_lambda: float = 1.0,
            gan_lambda: float = 0.0,
            ema_rate: float = 0.999,
            n_sampling_steps: int = 10,
            sigma_sample_density_type: str = 'loglogistic',
    ) -> None:
        super().__init__()
        self.use_gan = False
        self.ema_rate = ema_rate
        self.cmt_lambda = cmt_lambda
        self.diffusion_lambda = diffusion_lambda
        self.gan_lambda = gan_lambda
        self.n_discrete_t = n_discrete_t
        self.model = ConsistencyTrajectoryNetwork(
            x_dim=1,
            hidden_dim=256,
            time_embed_dim=4,
            cond_dim=1,
            cond_mask_prob=0.0,
            num_hidden_layers=4,
            output_dim=1,
            dropout_rate=0.1,
            cond_conditional=conditioned
        ).to(device)
        # we need an ema version of the model for the consistency loss
        self.target_model = copy.deepcopy(self.model)
        # we further can use a teacher model for the solver
        self.use_teacher = use_teacher
        if self.use_teacher:
            self.teacher_model = copy.deepcopy(self.model)
        self.device = device
        self.sampler_type = sampler_type
        # use the score wrapper 
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.n_sampling_steps = n_sampling_steps
        self.sigma_sample_density_type = sigma_sample_density_type
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = 0
        
    def diffusion_wrapper(self, model, x, cond, t, s):
        """
        Performs the diffusion wrapper for the given model, x, cond, and t.
        Based on the conditioning from EDM Karras et al. 2022.

        Args:
            model (torch.nn.Module): The neural network model to be used for the diffusion process.
            x (torch.Tensor): The input tensor to the model.
            cond (torch.Tensor): The conditioning tensor to be used during the diffusion process.
            t (float): The time step for the diffusion process.

        Returns:
            torch.Tensor: The scaled output tensor after applying the diffusion wrapper to the model.
        """
        c_skip = self.sigma_data**2 / (
            t ** 2 + self.sigma_data**2
        )
        c_out = (
            t * self.sigma_data / (t**2 + self.sigma_data**2) ** 0.5
        )
        # these two are not mentioned in the paper but they use it in their code
        c_in = 1 / (t**2 + self.sigma_data**2) ** 0.5
        
        t = 0.25 * torch.log(t + 1e-40)
        c_in = append_dims(c_in, x.ndim)
        c_out = append_dims(c_out, x.ndim)
        c_skip = append_dims(c_skip, x.ndim)

        diffusion_output = model(c_in * x, cond, t, s)
        scaled_output = c_out * diffusion_output + c_skip * x
        
        return scaled_output
    
    def cmt_wrapper(self, model, x, cond, t, s):
        """
        Applies the new cmt wrapper from page 4 of https://openreview.net/attachment?id=ymjI8feDTD&name=pdf

        Args:
            model (torch.nn.Module): The neural network model to be used for the diffusion process.
            x (torch.Tensor): The input tensor to the model.
            cond (torch.Tensor): The conditioning tensor to be used during the diffusion process.
            t (float): The time step for the diffusion process.
            s: (float): the target noise level for the diffusion process.

        Returns:
            torch.Tensor: The scaled output tensor after applying the diffusion wrapper to the model.
        """
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
        if len(s.shape) == 1:
            s = s.unsqueeze(1)
        G_0 = (s / t) * x + (1 - s /t) * self.diffusion_wrapper(model, x, cond, t, s)
        
        return G_0
    
    def _update_ema_weights(self):
        """
        Updates the exponential moving average (EMA) weights of the target model.

        The method performs the following steps:
        1. Gets the state dictionary of the self.model (source model).
        2. Updates the EMA weights for each parameter in the target model by computing the weighted average between 
        the corresponding parameter in the target model and the parameter in the source model, using the EMA rate parameter.
        """
        # Get the state dictionary of the current/source model
        state_dict = self.model.state_dict()
        # Get the state dictionary of the target model
        target_state_dict = self.target_model.state_dict()

        # Iterate over the parameters in the target model state dictionary
        for key in state_dict:
            if key in target_state_dict:
                # Update the EMA weights for each parameter
                target_param_data = target_state_dict[key].data
                model_param_data = state_dict[key].data
                target_state_dict[key].data.copy_((1 - self.ema_rate) * target_param_data + self.ema_rate * model_param_data)

        # You can optionally load the updated state dict into the target model, if necessary
        # self.target_model.load_state_dict(target_state_dict)

    def train_step(self, x, cond):
        """
        Main training step method to compute the loss for the Consistency Trajectory Model.
        The loss consists of three parts: the consistency loss, the diffusion loss, and the GAN loss (optional).
        The first part is similar to Song et al. 23 and the second part is similar to Karras et al. 2022.

        Args:

        Returns:

        """
        self.model.train()
        t_ctm = self.sample_noise_levels(shape=(len(x),), N=self.n_discrete_t, device=self.device)
        noise = torch.randn_like(x)
        # next we sample s in range of [0, t]
        s = torch.rand_like(t_ctm) * t_ctm
        # next we sample u in range of (s, t]
        u = torch.rand_like(t_ctm) * (t_ctm - s) + s
        # get the noise samples
        x_t = x + noise * append_dims(t_ctm, x.ndim)
        # use the solver if we have a teacher model otherwise use the euler method
        solver_target = self.solver(x_t, cond, t_ctm, u)

        # compute the cmt consistency loss
        cmt_loss = self.ctm_loss(x_t, cond, t_ctm, s, u, solver_target)
        
        # compute the diffusion loss
        # sample noise for the diffusion loss from the continuous noise distribution
        t_sm = self.make_sample_density()(shape=(len(x),), device=self.device)
        x_t_sm = x + noise * append_dims(t_sm, x.ndim)
        diffusion_loss = self.diffusion_loss(x, x_t_sm, cond, t_sm)

        # compute the GAN loss if chosen
        # not implemented yet
        if self.use_gan:
            gan_loss = self.gan_loss(x_t, cond, x_t_sm)
        else:
            gan_loss = 0

        # compute the total loss
        loss = self.cmt_lambda * cmt_loss + self.diffusion_lambda * diffusion_loss + self.gan_lambda * gan_loss
        
        # perform the backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update the ema weights
        self._update_ema_weights()
        
        return loss, cmt_loss, diffusion_loss, gan_loss
    
    def sample_noise_levels(self, shape, N, device='cpu'):
        """
        Samples a tensor of the specified shape with noise levels 
        from `N` discretized levels of the noise scheduler.

        Args:
            shape (tuple): Shape of the tensor to sample.
            N (int): Number of discrete noise levels to discretize the scheduler.
            device (str): Device on which to create the noise levels, 'cpu' or 'cuda'.

        Returns:
            torch.Tensor: Tensor containing sampled noise levels.
        """
        # Get the N discretized noise levels
        discretized_sigmas = get_sigmas_exponential(N, self.sigma_min, self.sigma_max, self.device)
        
        # Sample indices from this discretized range
        indices = torch.randint(0, N, size=shape, device=device)
        
        # Use these indices to gather the noise levels from the discretized sigmas
        sampled_sigmas = discretized_sigmas[indices]
        return sampled_sigmas

    def solver(self, x, cond, t, s):
        """
        Eq. (3) in the paper
        """
        if self.use_teacher:
            solver = self.teacher_model
        else:
            solver = self.model

        solver_pred = self.heun_update_step(solver, x, cond, t, s)

        return solver_pred

    
    def eval_step(self, x, cond):
        """
        Eval step method to compute the loss for the action prediction.
        """
        self.model.eval()
        self.target_model.eval()
        x = x.to(self.device)
        cond = cond.to(self.device)
        # next generate the discrete timesteps
        t = [self.sample_discrete_timesteps(i) for i in range(self.t_steps)]
        # compute the loss
        x_T = torch.randn_like(x) * self.sigma_max
        pred_x = self. sample(x_T, cond, t)
        loss = torch.nn.functional.mse_loss(pred_x, x)
        return loss
    
    def ctm_loss(self, x_t, cond, t, s, u, solver_target):
        """
        # TODO add description

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, dim].
            cond (torch.Tensor): Conditioning tensor of shape [batch_size, cond_dim].
            t1 (torch.Tensor): First discrete timestep tensor of shape [batch_size, 1].
            t2 (torch.Tensor): Second discrete timestep tensor of shape [batch_size, 1].

        Returns:
            torch.Tensor: Consistency loss tensor of shape [].
        """

        # compute the cmt prediction: jump from t to s
        ctm_pred = self.cmt_wrapper(self.model, x_t, cond, t, s)

        # compute the cmt target prediction without gradient: jump from u to s
        with torch.no_grad():
            ctm_target = self.cmt_wrapper(self.target_model, solver_target, cond, u, s)

        # transform them into the clean data space by jumping without gradient from s to 0
        # for both predictions and comparing them in the clean data space
        jump_target = einops.repeat(torch.tensor([0]), '1 -> (b 1)', b=len(x_t))
        with torch.no_grad():
            ctm_pred_clean = self.cmt_wrapper(self.model, ctm_pred, cond, s, jump_target)
            ctm_target_clean = self.cmt_wrapper(self.model, ctm_target, cond, s, jump_target)
        
        # compute the cmt loss
        cmt_loss = torch.nn.functional.mse_loss(ctm_pred_clean, ctm_target_clean)

        return cmt_loss


    @torch.no_grad()   
    def heun_update_step(self, model, x, cond, t1, t2):
        """
        Computes a single Heun update step from the Euler sampler with the teacher model

        Parameters:
        x (torch.Tensor): The input tensor.
        t1 (torch.Tensor): The initial timestep.
        t2 (torch.Tensor): The final timestep.
        x0 (torch.Tensor): The ground truth value used to compute the Euler update step.

        Returns:
        torch.Tensor: The output tensor after taking the Euler update step.
        """
        denoised = self.diffusion_wrapper(model, x, cond, t1, t1)
        d = (x - denoised) / append_dims(t1, x.ndim)
        
        
        sample_temp = x + d * append_dims(t2 - t1, x.ndim)
        denoised_2 = self.diffusion_wrapper(model, sample_temp, cond, t2, t2)
        d_2 = (sample_temp - denoised_2) / append_dims(t2, x.ndim)
        d_prime = (d + d_2) / 2
        samples = x + d_prime * append_dims(t2 - t1, x.ndim)
        
        return samples


    def score_matching_loss(self, x, cond, t):
        """
        Computes the diffusion training loss for the given model, input, condition, and time.

        Args:
        - self: the object instance of the model
        - x (torch.Tensor): the input tensor of shape (batch_size, channels, height, width)
        - cond (torch.Tensor): the conditional input tensor of shape (batch_size, cond_dim)
        - t (torch.Tensor): the time step tensor of shape (batch_size,)

        Returns:
        - loss (torch.Tensor): the diffusion training loss tensor of shape ()

        The diffusion training loss is computed based on the following equation from Karras et al. 2022:
        loss = (model_output - target)^2.mean()
        where,
        - noise: a tensor of the same shape as x, containing randomly sampled noise
        - x_1: a tensor of the same shape as x, obtained by adding the noise tensor to x
        - c_skip, c_out, c_in: scaling tensors obtained from the diffusion scalings for the given time step
        - t: a tensor of the same shape as t, obtained by taking the natural logarithm of t and dividing it by 4
        - model_output: the output tensor of the model for the input x_1, condition cond, and time t
        - target: the target tensor for the given input x, scaling tensors c_skip, c_out, c_in, and time t
        """
        noise = torch.randn_like(x)
        x_1 = x + noise * append_dims(t, x.ndim)
        c_skip, c_out, c_in = [append_dims(x, 2) for x in self.get_diffusion_scalings(t)]
        t = torch.log(t) / 4
        model_output = self.diffusion_model(x_1 * c_in, cond, t)
        target = (x - c_skip * x_1) / c_out
        return (model_output - target).pow(2).mean()

    def get_diffusion_scalings(self, sigma):
        """
        Computes the scaling factors for diffusion training at a given time step sigma.

        Args:
        - self: the object instance of the model
        - sigma (float or torch.Tensor): the time step at which to compute the scaling factors
        
        , where self.sigma_data: the data noise level of the diffusion process, set during initialization of the model

        Returns:
        - c_skip (torch.Tensor): the scaling factor for skipping the diffusion model for the given time step sigma
        - c_out (torch.Tensor): the scaling factor for the output of the diffusion model for the given time step sigma
        - c_in (torch.Tensor): the scaling factor for the input of the diffusion model for the given time step sigma

        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in
    
    @staticmethod
    def mean_flat(tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def diffusion_train_step(self,  x, cond, train_step, max_steps):
        """
        Computes the training loss and performs a single update step for the score-based model.

        Args:
        - self: the object instance of the model
        - x (torch.Tensor): the input tensor of shape (batch_size, dim)
        - cond (torch.Tensor): the conditional input tensor of shape (batch_size, cond_dim)

        Returns:
        - loss.item() (float): the scalar value of the training loss for this batch

        """
        self.model.train()
        x = x.to(self.device)
        cond = cond.to(self.device)
        self.optimizer.zero_grad()
        t = self.make_sample_density()(shape=(len(x),), device=self.device)
        x_t = x + torch.randn_like(x) * append_dims(t, x.ndim)
        loss = self.diffusion_loss(x, x_t, cond, t)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    
    def diffusion_loss(self, x, x_t, cond, t):
        """
        Computes the diffusion training loss for the given model, input, condition, and time.

        Args:
        - self: the object instance of the model
        - x (torch.Tensor): the input tensor of shape (batch_size, channels, height, width)
        - cond (torch.Tensor): the conditional input tensor of shape (batch_size, cond_dim)
        - t (torch.Tensor): the time step tensor of shape (batch_size,)

        Returns:
        - loss (torch.Tensor): the diffusion training loss tensor of shape ()

        The diffusion training loss is computed based on the following equation from Karras et al. 2022:
        loss = (model_output - target)^2.mean()
        where,
        - noise: a tensor of the same shape as x, containing randomly sampled noise
        - x_t: a tensor of the same shape as x, obtained by adding the noise tensor to x
        - c_skip, c_out, c_in: scaling tensors obtained from the diffusion scalings for the given time step
        - t: a tensor of the same shape as t, obtained by taking the natural logarithm of t and dividing it by 4
        - model_output: the output tensor of the model for the input x_1, condition cond, and time t
        - target: the target tensor for the given input x, scaling tensors c_skip, c_out, c_in, and time t
        """
        c_skip, c_out, c_in = [append_dims(x, 2) for x in self.get_diffusion_scalings(t)]
        t = torch.log(t) / 4
        model_output = self.model(x_t * c_in, cond, t, t)
        target = (x - c_skip * x_t) / c_out
        return (model_output - target).pow(2).mean()
        
    def update_teacher_model(self):
        self.teacher_model.load_state_dict(self.target_model.state_dict())
        
    def euler_update_step(self, x, t1, t2, x0):
        """
        Computes a single update step from the Euler sampler with a ground truth value.

        Parameters:
        x (torch.Tensor): The input tensor.
        t1 (torch.Tensor): The initial timestep.
        t2 (torch.Tensor): The final timestep.
        x0 (torch.Tensor): The ground truth value used to compute the Euler update step.

        Returns:
        torch.Tensor: The output tensor after taking the Euler update step.
        """
        denoiser = x0

        d = (x - denoiser) / append_dims(t1, x.ndim)
        samples = x + d * append_dims(t2 - t1, x.ndim)

        return samples

    @torch.no_grad()
    @ema_eval_wrapper
    def sample_singlestep(self, x_shape, cond, return_seq=False):
        """
        Samples a single step from the trained consistency model. 
        If return_seq is True, returns a list of sampled tensors, 
        otherwise returns a single tensor. 
        
        Args:
        - x_shape (tuple): the shape of the tensor to be sampled.
        - cond (torch.Tensor or None): the conditional tensor.
        - return_seq (bool, optional): whether to return a list of sampled tensors (default False).
        
        Returns:
        - (torch.Tensor or list): the sampled tensor(s).
        """
        sampled_x = []
        self.model.eval()
        if cond is not None:
            cond = cond.to(self.device)

        x = torch.randn_like(x_shape).to(self.device) * self.sigma_max * 1.5
        sampled_x.append(x)
        x = self.cmt_wrapper(self.model, x, cond, torch.tensor([self.sigma_max]), torch.tensor([0]))
        sampled_x.append(x)
        if return_seq:
            return sampled_x
        else:
            return x
        
    @torch.no_grad()
    @ema_eval_wrapper
    def sample_diffusion_euler(self, x_shape, cond, n_sampling_steps=None, return_seq=False):
        """
        Sample from the pre-trained diffusion model using the Euler method. This method is used for sanity checking 
        the learned diffusion model. It generates a sequence of samples by taking small steps from one sample to the next. 
        At each step, it generates a new noise from a normal distribution and combines it with the previous sample 
        to get the next sample.
        
        Parameters:
        - x_shape (torch.Tensor): Shape of the input tensor to the model.
        - cond (torch.Tensor): Conditional information for the model.
        - n_sampling_steps (int, optional): Number of sampling steps to take. Defaults to None.
        - return_seq (bool, optional): Whether to return the full sequence of samples or just the final one. 
                                        Defaults to False.
                                        
        Returns:
        - x (torch.Tensor or List[torch.Tensor]): Sampled tensor from the model. If `return_seq=True`, it returns
                                                a list of tensors, otherwise it returns a single tensor.
        """
        self.model.eval()
        if cond is not None:
            cond = cond.to(self.device)
        x = torch.randn_like(x_shape).to(self.device) * self.sigma_max * 1.5
        # x = torch.linspace(-4, 4, len(x_shape)).view(len(x_shape), 1).to(self.device)

        sampled_x = []
        if n_sampling_steps is None:
            n_sampling_steps = self.n_sampling_steps
        
        # sample the sequence of timesteps
        sigmas = self.sample_seq_timesteps(N=n_sampling_steps, type='exponential')
        sampled_x.append(x)
        # iterate over the remaining timesteps
        for i in trange(len(sigmas) - 1, disable=True):
            denoised = self.diffusion_wrapper(self.model, x, cond, sigmas[i], sigmas[i])
            x = self.euler_update_step(x, sigmas[i], sigmas[i+1], denoised)
            sampled_x.append(x)
        if return_seq:
            return sampled_x
        else:
            return x

    def sample_seq_timesteps(self, N=100, type='karras'):
        """
        Generates a sequence of N timesteps for the given type.

        Args:
        - self: the object instance of the model
        - N (int): the number of timesteps to generate
        - type (str): the type of sequence to generate, either 'karras', 'linear', or 'exponential'

        Returns:
        - t (torch.Tensor): the generated sequence of timesteps of shape (N,)

        The method generates a sequence of timesteps for the given type using one of the following functions:
        - get_sigmas_karras: a function that generates a sequence of timesteps using the Karras et al. schedule
        - get_sigmas_linear: a function that generates a sequence of timesteps linearly spaced between sigma_min and sigma_max
        - get_sigmas_exponential: a function that generates a sequence of timesteps exponentially spaced between sigma_min and sigma_max
        where,
        - self.sigma_min, self.sigma_max: the minimum and maximum timesteps, set during initialization of the model
        - self.rho: the decay rate for the Karras et al. schedule, set during initialization of the model
        - self.device: the device on which to generate the timesteps, set during initialization of the model

        """
        if type == 'karras':
            t = get_sigmas_karras(N, self.sigma_min, self.sigma_max, self.rho, self.device)
        elif type == 'linear':
            t = get_sigmas_linear(N, self.sigma_min, self.sigma_max, self.device)
        elif type == 'exponential':
            t = get_sigmas_exponential(N, self.sigma_min, self.sigma_max, self.device)
        else:
            raise NotImplementedError('Chosen Scheduler is implemented!')
        return t
    
    def make_sample_density(self):
        """
        Returns a function that generates random timesteps based on the chosen sample density.

        Args:
        - self: the object instance of the model

        Returns:
        - sample_density_fn (callable): a function that generates random timesteps

        The method returns a callable function that generates random timesteps based on the chosen sample density.
        The available sample densities are:
        - 'lognormal': generates random timesteps from a log-normal distribution with mean and standard deviation set
                    during initialization of the model also used in Karras et al. (2022)
        - 'loglogistic': generates random timesteps from a log-logistic distribution with location parameter set to the
                        natural logarithm of the sigma_data parameter and scale and range parameters set during initialization
                        of the model
        - 'loguniform': generates random timesteps from a log-uniform distribution with range parameters set during
                        initialization of the model
        - 'uniform': generates random timesteps from a uniform distribution with range parameters set during initialization
                    of the model
        - 'v-diffusion': generates random timesteps using the Variational Diffusion sampler with range parameters set during
                        initialization of the model
        - 'discrete': generates random timesteps from the noise schedule using the exponential density
        - 'split-lognormal': generates random timesteps from a split log-normal distribution with mean and standard deviation
                            set during initialization of the model
        """
        sd_config = []
        
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(rand_log_normal, loc=loc, scale=scale)
        
        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)
        
        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_uniform, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'uniform':
            return partial(rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.n_sampling_steps, 'exponential')
            return partial(rand_discrete, values=sigmas)
        else:
            raise ValueError('Unknown sample density type')
    
    

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = ((max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho) # [:-1]
    return sigmas.to(device)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]