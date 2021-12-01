#SBATCH --nodes=1
#SBATCH --mem=4G
#SBACTH --time=2:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=marwanghalib@cmail.carleton.ca
#SBATCH --mail-type=ALL


python3 lstm_RNN_CV.py