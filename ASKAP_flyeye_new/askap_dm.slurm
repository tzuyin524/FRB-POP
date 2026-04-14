#!/bin/bash
#SBATCH --job-name=askap_frb_chunks
#SBATCH --array=0-168             # 169 jobs (0-168)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G                  # Per job
#SBATCH --time=01:30:00           # Per job
#SBATCH --output=logs/askap_%A_%a.log

source /fred/oz002/psrhome/scripts/psrhome.sh
module load sympy/1.11.1
mkdir -p logs
# Get the script name from the array index
scripts=(run_askap_z*.py)
script="${scripts[$SLURM_ARRAY_TASK_ID]}"

echo "Running: $script"
python "$script"

# After the last one completes, run combiner
if [ $SLURM_ARRAY_TASK_ID -eq $((SLURM_ARRAY_TASK_MAX)) ]; then
    echo "Combining all DM arrays..."
    python combine_all_chunks.py
fi