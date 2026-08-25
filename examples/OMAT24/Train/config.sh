export EQUINNX_DIR=/home/theochem/allouche/EquiNNx
export EQUINNX_SRC=$EQUINNX_DIR/src
export EQUINNX_TOOLS=$EQUINNX_DIR/tools
export EQUINNX_SCRIPTS=$EQUINNX_DIR/scripts/Local
export PYTHONPATH=$PYTHONPATH:$EQUINNX_SRC
export PATH=$PATH:$EQUINNX_SRC
export PATH=$PATH:$EQUINNX_SCRIPTS
source /home/allouche/Softwares/anaconda3/etc/profile.d/conda.sh
conda activate jax_cuda13
