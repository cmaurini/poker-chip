# Linear elastic computation

- Launch simulations for getting equivalent stiffness as a function of the aspect ratio
```
mpiexec -n 4 python solvers/poker_chip_linear.py \
  --config-dir config --config-name config_linear --multirun \
  geometry.geometric_dimension=3 geometry.L=1,2,3,4,5,6,7,8,9,10,12,14,16 geometry.H=1.0 geometry.h_div=5 model.kappa=100 model.mu=1 sym=true \
  output_name=Eeq3d-L \
  solver_type=gamg \
  use_iterative_solver=true verbose_solver=true
```
- Postprocess: 
```python scripts/collect_and_plot_results_3d.py```