
from tqdm import tqdm
import torch 

from ctm.ctm import ConsistencyTrajectoryModel
from ctm.toy_tasks.data_generator import DataGenerator
from ctm.visualization.vis_utils import plot_main_figure


"""
Discrete consistency distillation training of the consistency model on a toy task.
We train a diffusion model and the consistency model at the same time and iteratively 
update the weights of the consistency model and the diffusion model.
"""

if __name__ == "__main__":

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    n_sampling_steps = 10
    use_pretraining = True
    cm = ConsistencyTrajectoryModel(
        data_dim=1,
        cond_dim=1,
        sampler_type='ddim',
        lr=1e-4,
        sigma_data=0.5,
        sigma_min=0.05,
        solver_type='heun',
        sigma_max=5,
        n_discrete_t=18,
        conditioned=False,
        diffusion_lambda= 1,
        device=device,
        rho=7,
        ema_rate=0.999,
        use_teacher=use_pretraining,
    )
    train_epochs = 2003
    # chose one of the following toy tasks: 'three_gmm_1D' 'uneven_two_gmm_1D' 'two_gmm_1D' 'single_gaussian_1D'
    data_manager = DataGenerator('two_gmm_1D')
    samples, cond = data_manager.generate_samples(5000)
    samples = samples.reshape(-1, 1).to(device)
    cond = cond.to(device)
    pbar = tqdm(range(train_epochs))
    
    # if not simultanous_training:
    # First pretrain the diffusion model and then train the consistency model
    if use_pretraining:
        for i in range(train_epochs):
            cond = cond.reshape(-1, 1).to(device)        
            diff_loss = cm.diffusion_train_step(samples, cond, i, train_epochs)
            pbar.set_description(f"Step {i}, Diff Loss: {diff_loss:.8f}")
            pbar.update(1)
        
        cm.update_teacher_model()
        
        plot_main_figure(
            data_manager.compute_log_prob, 
            cm, 
            200, 
            train_epochs, 
            sampling_method='euler', 
            n_sampling_steps=n_sampling_steps,
            x_range=[-4, 4], 
            save_path='./plots/'
        )
    
    
    # Train the consistency trajectory model either simultanously with the diffusion model or after pretraining
    for i in range(train_epochs):
        cond = cond.reshape(-1, 1).to(device)        
        loss, cmt_loss, diffusion_loss, gan_loss = cm.train_step(samples, cond)
        
        pbar.set_description(f"Step {i}, Loss: {loss:.8f}, CTM Loss: {cmt_loss:.8f}, Diff Loss: {diffusion_loss:.8f}, GAN Loss: {gan_loss:.8f}")
        pbar.update(1)
    
    # Plotting the results of the training
    # We do this for the one-step and the multi-step sampler to compare the results
    if not use_pretraining:
        plot_main_figure(
                data_manager.compute_log_prob, 
                cm, 
                200, 
                train_epochs, 
                sampling_method='euler', 
                n_sampling_steps=n_sampling_steps,
                x_range=[-4, 4], 
                save_path='./plots/'
            )
    
    plot_main_figure(
        data_manager.compute_log_prob, 
        cm, 
        200, 
        train_epochs, 
        sampling_method='onestep', 
        x_range=[-4, 4], 
        save_path='./plots/ctm'
    )

            
    print('done')