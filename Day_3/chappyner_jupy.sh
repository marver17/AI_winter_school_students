#!/usr/bin/env bash
# This script is a wrapper for the Chappyner package
# to aid Cineca users to easily connect on web tools
# running on HPC clusters. Details for Chappyner here:
# https://gitlab.hpc.cineca.it/interactive_computing/chappyner

#####        CHANGE HERE       ########
#user= # if not set, the script will prompt for a username
cluster=leonardo
account=tra26_minwinsc # if not set, user default will be used
tool=jupyter-train
batch_options="--ntasks 4 \
               --gres gpu:4 \
               --partition boost_usr_prod \
               --qos  boost_qos_dbg \
               --time 00:30:00 \
               --nodes 1 \
               --exclusive \
               --mem 400GB"
##### DON'T CHANGE AFTER HERE  ########

# Install chappyner in a virtual environment (if not already present)
# The path of the virtual environment is $HOME/.chappyner/venv
if [[ ! -f "$HOME/.chappyner/venv/bin/chappyner" ]]; then
    echo "Chappyner will be installed in a virtual environment in the following path: $HOME/.chappyner/venv"
    read -p "Press Enter to continue..."

    echo "---------------------------"
    mkdir -p $HOME/.chappyner
    python3 -m venv $HOME/.chappyner/venv
    $HOME/.chappyner/venv/bin/pip install chappyner
    echo "---------------------------"
fi

# Ask username if 'user' is not set
if [ -z "${user:-}" ]; then
    read -p "Enter your username on Cineca clusters: " user
fi

# Append account to slurm options if 'account' variable is set
# (slurm default setting is used otherwise)
if [ -n "${account:-}" ]; then
    batch_options="$batch_options --account $account"
fi

# Set default options for chappyner
chappyner_options="--user $user --cluster $cluster --tool $tool"

# Tune options in case of training accounts
if [[ $user =~ ^a([0-9]{2})tr[a-e]([[:alnum:]]{2})$ ]]; then
    chappyner_options="$chappyner_options --no-startup-script --engine socket"
fi

# Start tool via chappyner
$HOME/.chappyner/venv/bin/chappyner $chappyner_options $batch_options
