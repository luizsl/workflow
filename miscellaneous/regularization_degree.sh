#!/bin/bash -l
#

#SBATCH -J evaluate_regularisation
#SBATCH --cpus-per-task 16
#SBATCH --output standard_output_file.%J.out
#SBATCH --error standard_error_file.%J.err
#SBATCH -p cosma
#SBATCH -A durham
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luiz.sl@outlook.com

module purge
# load the modules used to build your program.
module load python/3.10.7
# module load openmpi/3.0.1

# Run the program
python3 regularization_degree.py
