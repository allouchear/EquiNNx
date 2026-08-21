__conda_setup="$('/sps/ilm/allouche/Softwares/anaconda3-2023/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sps/ilm/allouche/Softwares/anaconda3-2023/etc/profile.d/conda.sh" ]; then
        . "/sps/ilm/allouche/Softwares/anaconda3-2023/etc/profile.d/conda.sh"
    else
        export PATH="/sps/ilm/allouche/Softwares/anaconda3-2023/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate jax
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sps/ilm/allouche/Softwares/anaconda3-2023/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sps/ilm/allouche/Softwares/anaconda3-2023/envs/jax/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sps/ilm/allouche/Softwares/anaconda3-2023/envs/jax/lib/python3.10/site-packages/tensorrt_bindings
