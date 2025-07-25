# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
export CONDA_PREFIX=/home/allouche/Softwares/anaconda3
export PATH="/home/allouche/Softwares/anaconda3/bin/":$PATH

__conda_setup="$('/home/allouche/Softwares/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/allouche/Softwares/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/allouche/Softwares/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/allouche/Softwares/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate jax
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.10/site-packages/tensorrt_libs
