Code for the paper "Coordinated drift of receptive fields during noisy representation learning". See also the priprint https://www.biorxiv.org/content/10.1101/2021.08.30.458264v1.abstract

## Dependence
Some of the scripts depend on ITE (information theoretical estimators) which can be downloaded from https://bitbucket.org/szzoli/ite/src/master/

To plot figures without running the simulations, download the data from https://www.dropbox.com/sh/hgcxraa6kvv7jm3/AACVKWIomrHdyU46iMiQdQl4a?dl=0 and put the `data` folder in the main directory

## Simulations

To run the simulation using the scripts in the folder `simulation`. The resulted data should be stored in the folder `data` 

- `drift_PSP.m` Drift in Principal Subspace Projection (PSP) task, related to Figure 2 
- `PSP_D_Dependency.m` Change the input statistics in PSP and study  its effect on diffusion constant, related to Figure 2
- `ringPlaceModel.m` The nonlinear Hebbian/anti-Hebbian with "ring-shaped" data manifold, related to Figure 3 and Figure 4
- `ring_model_three_phases.m` Simulation of the "ring" model with different noise sources: noise only in forward matrix, noise in recurrent weight and noise in both weight matrices. Related to Figure 4F
- `placeCell1D_slice.m` 1D place cell model, input is draw from 1D grid fields which are slices through 2D grid fields. Related to Figure 5
- `placeCell1D_ExciInhi.m` 1 D place cell model with both excitatory and inhibitory neurons
- `place_cell_learn_forget.m` 1D place cell model with alternating learning and forgetting sessions.
- `placeCell1D_slice_three_phases.m` Different noise source in the 1D place cell model
- `place1D_compare_model_experiment.m` Comparison of experimental data of hippocampal CA1 place cells.
- `placeCell1D_slice_multi_timescale.m` Show that learned representation can be quite stable if there are both fast and slow timescale in the synaptic dynamics
- `placeCells.m` Simulation of the 2D place cell model, related to Figure 5
- `Tmaze.m` Simulation of representational drift of parietal cortex neurons during T-maze task, related to Figure 6
- `comparePSP_PCA_SangerNoise.m` Show that "degeneracy" of objective function is important for observing representational drift in linear networks, we compare PSP with PCA and Sanger's learning rule.



##  Plot the figures

To plot the figures, run the code in the folder `plot`. The script name contains the information about which figure it plots. 



MATLAB version tested: 20220(b)
