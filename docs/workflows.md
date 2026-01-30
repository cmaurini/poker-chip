# Parametric Studies with Snakemake

This project uses Snakemake for managing parametric studies with `poker_chip.py`. Snakemake provides:
- **Local execution**: Run multiple simulations in parallel on your machine
- **Cluster execution**: Automatically submit jobs to HPC clusters (Slurm, PBS, etc.)
- **Dependency tracking**: Only re-run simulations that need to be updated
- **Reproducibility**: All parameters and outputs are tracked

## Quick Start

### 1. Run a Single Simulation (default)

```bash
cd /path/to/poker_chip
snakemake -j auto --profile local
```

This runs one simulation with default parameters from `sweeps/default.yml`.

### 2. Run a Parametric Sweep Locally

```bash
# Run with pre-defined sweep (creates 6 simulations)
snakemake -j 4 --profile local --config configfile=sweeps/ell_mesh_convergence.yml

# Use -j auto for all available cores
snakemake -j auto --profile local --config configfile=sweeps/material_parameters.yml
```

### 3. Dry Run (See What Would Execute)

```bash
snakemake -n --config configfile=sweeps/ell_mesh_convergence.yml
```

### 4. Run on HPC Cluster (Slurm)

```bash
# Submit jobs to cluster
snakemake --profile slurm --config configfile=sweeps/ell_mesh_convergence.yml

# Limit max concurrent jobs
snakemake --profile slurm -j 20 --config configfile=sweeps/ell_mesh_convergence.yml
```

## Configuration Files

### Create Your Own Sweep

Create a new file in `sweeps/` directory (e.g., `sweeps/my_study.yml`):

```yaml
# Fixed parameters (single value)
gdim: 2
nsteps: 50

# Parameters to sweep (lists create combinations)
ell:
  - 0.01
  - 0.02
  - 0.05

gamma_mu:
  - 1.5
  - 2.0

# More fixed parameters
e_c: 0.2
h_div: 3.0
mu_0: 0.59
kappa: 500.0
p_cav: 1.475
```

Then run:
```bash
snakemake --profile local --config configfile=sweeps/my_study.yml
```

### Parameter Notes

- **Single value**: `ell: 0.01` - held constant in all simulations
- **List of values**: `ell: [0.01, 0.02, 0.05]` - creates sweep over these values
- **Combinations**: Multiple lists create Cartesian product
  - Example: `ell: [0.01, 0.02]` and `h_div: [2.0, 3.0]` → 2×2 = 4 simulations

Valid parameters (all from command-line args):
- `gdim`, `nsteps`, `ell`, `e_c`, `gamma_mu`, `w_1`, `h_div`, `mu_0`, `kappa`, `p_cav`, `elastic`, `nonlinear_elastic`, `model_dimension`, `L`, `H`, `loadmax`

## Cluster Configuration

### Slurm (HPC Clusters)

Edit `.snakemake/profiles/slurm/config.yaml`:

```yaml
cores: 1
jobs: 10                    # Max concurrent jobs
cluster:
  mkdir -p results/{wildcards.output_dir} &&
  sbatch
    --job-name={rule}-{wildcards.combo_idx}
    --cpus-per-task={threads}
    --mem={resources.mem_mb}M
    --time={resources.time_min}
    --partition=gpu         # Change to your queue
    --gres=gpu:1            # Remove if no GPU available
    --mail-type=FAIL,END
    --mail-user=your.email@example.com
```

Key options:
- `--partition`: Queue name on your cluster
- `--gres=gpu:1`: GPU allocation (adjust or remove)
- `--time`: Maximum runtime per job (minutes)
- `--mem`: Memory per job (MB)
- `--cpus-per-task`: CPU cores per job

### PBS/Torque

Create `.snakemake/profiles/pbs/config.yaml`:

```yaml
cores: 1
jobs: 10
cluster: qsub -N {rule}-{wildcards.combo_idx} -l nodes=1:ppn=1,mem={resources.mem_mb}mb,walltime={resources.time_min}:00
```

## Output Structure

Results are saved to `results/` directory with structure:
```
results/
├── ell0.01_h_div2_mu0.59/
│   ├── ell0.01_h_div2_mu0.59-parameters.yml    (saved configuration)
│   ├── ell0.01_h_div2_mu0.59.log               (execution log)
│   ├── ell0.01_h_div2_mu0.59_data.json         (output data)
│   └── ...                                      (other outputs)
├── ell0.01_h_div3_mu0.59/
│   └── ...
└── ...
```

Each simulation's parameters are saved in `-parameters.yml` files for reproducibility.

## Common Commands

```bash
# Run with progress
snakemake --profile local -p --config configfile=sweeps/my_study.yml

# Run with detailed output
snakemake --profile local -v --config configfile=sweeps/my_study.yml

# Create DAG visualization
snakemake --dag --config configfile=sweeps/my_study.yml | dot -Tpdf > dag.pdf

# Unlock failed runs
snakemake --unlock --config configfile=sweeps/my_study.yml

# Clean up results
snakemake --delete-all-output --config configfile=sweeps/my_study.yml
```

## Troubleshooting

### Cluster jobs not submitting
- Check `.snakemake/profiles/slurm/config.yaml` has correct queue name
- Verify `sbatch` command works: `sbatch --help`
- Add `--dryrun` to see actual cluster commands

### Out of memory
- Increase `mem_mb` in profile config
- Reduce `cores`/`jobs` to run fewer simulations in parallel

### Simulations taking too long
- Increase `time_min` in profile config
- Use `-n` (dry run) to check estimated runtime

### Results not generated
- Check logs: `cat results/*/ell*.log`
- Run single simulation: `python scripts/poker_chip.py --ell 0.01 --gdim 2`

## Advanced Usage

### Rerun only failed simulations
```bash
snakemake --rerun-incomplete --profile local
```

### Force rerun all
```bash
snakemake --forceall --profile local
```

### Use custom configuration
```bash
snakemake --configfile my_config.yml --profile local
```

### Run subset of simulations
```bash
# Only ell=0.01 and ell=0.02
snakemake -j auto --profile local --config configfile=sweeps/ell_mesh_convergence.yml \
  --wildcards combo_idx=0 combo_idx=1
```

## Performance Tips

- **Local execution**: Use `-j auto` to use all cores
- **Cluster execution**: Start with `-j 10-20` to test, then scale up
- **Memory**: Monitor with `top` or cluster monitoring tools
- **I/O**: If disk-bound, reduce `-j` to avoid contention

## References

- Snakemake documentation: https://snakemake.readthedocs.io/
- Cluster execution: https://snakemake.readthedocs.io/en/stable/executing/cluster.html
- Profiles: https://snakemake.readthedocs.io/en/stable/executing/cloud.html#profiles
