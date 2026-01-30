#!/bin/bash
# Direct Slurm submission for poker_chip simulations (MPI version)
# This mimics your existing workflow but uses Snakemake for sweeps
#
# Usage:
#   sbatch poker_chip_direct.sh
#
# Edit SWEEP_CONFIG and other parameters below before submitting

#SBATCH -n 64
#SBATCH -N 2
#SBATCH --time=24:00:00
#SBATCH -J poker-chip-direct
#SBATCH -o logs/poker-chip-direct-%j.txt
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=YOUR_EMAIL@example.com

# Configuration
OUTDIR="/scratch/maurini/poker-chip-$(date +%Y%m%d-%H%M%S)"
SWEEP_CONFIG="sweeps/default.yml"  # Change to your sweep config

# Optional: Use MPI (set to 0 to disable)
USE_MPI=1

mkdir -p "$OUTDIR"
mkdir -p logs

echo "=================================================="
echo "Poker Chip Parametric Study (Direct MPI)"
echo "=================================================="
echo "Job ID: $SLURM_JOBID"
echo "Output directory: $OUTDIR"
echo "Sweep config: $SWEEP_CONFIG"
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
echo ""

# Run with Snakemake
if [ $USE_MPI -eq 1 ]; then
  echo "Running with MPI (Snakemake orchestration)"
  srun --mpi=pmix_v3 snakemake \
    --profile slurm \
    --config configfile="$SWEEP_CONFIG" \
    --outdir "$OUTDIR" \
    --jobs 2
else
  echo "Running without MPI"
  snakemake \
    --profile slurm \
    --config configfile="$SWEEP_CONFIG" \
    --outdir "$OUTDIR" \
    --jobs 2
fi

SWEEP_STATUS=$?

echo ""
echo "=================================================="
if [ $SWEEP_STATUS -eq 0 ]; then
  echo "Parametric study completed successfully!"
else
  echo "Parametric study failed with status $SWEEP_STATUS"
fi
echo "Results saved to: $OUTDIR"
echo "=================================================="

exit $SWEEP_STATUS
