# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/theochem/allouche/Softwares/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/theochem/allouche/Softwares/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/theochem/allouche/Softwares/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/theochem/allouche/Softwares/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


#export HDF5_USE_FILE_LOCKING=FALSE
conda activate jax
