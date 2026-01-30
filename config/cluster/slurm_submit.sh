#!/bin/bash
# Slurm submission wrapper for poker_chip parametric studies
# This script handles module loading and environment setup for your cluster
#
# Usage: sbatch slurm_submit.sh

#SBATCH -n 64
#SBATCH -N 2
#SBATCH --time=24:00:00
#SBATCH -J poker-chip-sweep
#SBATCH -o logs/poker-chip-%j.txt
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=YOUR_EMAIL@example.com

# Set output directory in scratch for faster I/O
OUTDIR="/scratch/maurini/poker-chip-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTDIR"
mkdir -p logs

echo "=================================================="
echo "Poker Chip Parametric Study"
echo "=================================================="
echo "Job ID: $SLURM_JOBID"
echo "Output directory: $OUTDIR"
echo "Number of nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_TASKS_PER_NODE"
echo "=================================================="

cd "$SLURM_SUBMIT_DIR"

# Clean environment and load modules
conda deactivate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Load required modules
module load dolfinx/0.9.0.post0/gcc-openmpi-py312

echo "Loaded modules:"
module list

echo "Python version:"
python3 --version

echo "Snakemake version:"
python3 -c "import snakemake; print(f'Snakemake {snakemake.__version__}')"

echo ""
echo "Starting Snakemake workflow..."
echo "=================================================="

# Run Snakemake with Slurm profile
# Specify sweep config via --config configfile=sweeps/YOUR_SWEEP.yml
snakemake \
  --profile slurm \
  --config configfile=sweeps/default.yml \
  --outdir "$OUTDIR" \
  --rerun-incomplete \
  --use-conda \
  --jobs 2

SNAKE_STATUS=$?

echo ""
echo "=================================================="
if [ $SNAKE_STATUS -eq 0 ]; then
  echo "Snakemake completed successfully!"
else
  echo "Snakemake failed with status $SNAKE_STATUS"
fi
echo "Results saved to: $OUTDIR"
echo "=================================================="

exit $SNAKE_STATUS
