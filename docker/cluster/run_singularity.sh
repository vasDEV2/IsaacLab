# #!/usr/bin/env bash

# echo "(run_singularity.py): Called on compute node from current isaaclab directory $1 with container profile $2 and arguments ${@:3}"

# #==
# # Helper functions
# #==

# setup_directories() {
#     # Check and create directories
#     for dir in \
#         "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
#         "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
#         "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
#         "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
#         "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
#         "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
#         "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
#         "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
#         if [ ! -d "$dir" ]; then
#             mkdir -p "$dir"
#             echo "Created directory: $dir"
#         fi
#     done
# }


# #==
# # Main
# #==


# # get script directory
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# # load variables to set the Isaac Lab path on the cluster
# source $SCRIPT_DIR/.env.cluster
# source $SCRIPT_DIR/../.env.base

# # make sure that all directories exists in cache directory
# setup_directories
# # copy all cache files
# cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR

# # make sure logs directory exists (in the permanent isaaclab directory)
# mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
# touch "$CLUSTER_ISAACLAB_DIR/logs/.keep"

# # copy the temporary isaaclab directory with the latest changes to the compute node
# cp -r $1 $TMPDIR
# # Get the directory name
# dir_name=$(basename "$1")

# cp -r $CLUSTER_RSL_RL_DIR $TMPDIR/$dir_name/compositional_rsl_rl

# # copy container to the compute node
# tar -xf $CLUSTER_SIF_PATH/$2.tar  -C $TMPDIR

# # execute command in singularity container
# # Use the container's uv environment with the synced checkout.
# mkdir -p $TMPDIR/ctmp
# apptainer exec \
#     -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
#     -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
#     -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
#     -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
#     -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
#     -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
#     -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
#     -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
#     -B $TMPDIR/$dir_name:/workspace/isaaclab:rw \
#     -B $CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/logs:rw \
#     -B $TMPDIR/ctmp:/tmp:rw \
#     --nv --writable --containall $TMPDIR/$2.sif \
#     bash -c "cd /workspace/isaaclab && ls -d wandb 2>/dev/null; /opt/isaaclab-venv/bin/python -c "import wandb, wandb.util; print(wandb.__version__, wandb.__file__, hasattr(wandb.util,'generate_id'))" ; && export WANDB_API_KEY=${WANDB_API_KEY} && export WANDB_MODE=${WANDB_MODE} && export ISAACLAB_PATH=/workspace/isaaclab && export WARP_CACHE_PATH=/tmp/warp_cache && export PYTHONFAULTHANDLER=1 && mkdir -p /tmp/warp_cache && cd /workspace/isaaclab && ln -sfn ${DOCKER_ISAACSIM_ROOT_PATH} _isaac_sim && uv pip install --no-deps -e /workspace/isaaclab/compositional_rsl_rl && uv run --no-sync isaaclab -p ${CLUSTER_PYTHON_EXECUTABLE} ${*:3}"

# # copy resulting cache files back to host
# rsync -azPv $TMPDIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..

# # if defined, remove the temporary isaaclab directory pushed when the job was submitted
# if $REMOVE_CODE_COPY_AFTER_JOB; then
#     rm -rf $1
# fi

# echo "(run_singularity.py): Return"

#!/usr/bin/env bash

echo "(run_singularity.py): Called on compute node from current isaaclab directory $1 with container profile $2 and arguments ${@:3}"

#==
# Helper functions
#==

setup_directories() {
    # Check and create directories
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}


#==
# Main
#==


# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the Isaac Lab path on the cluster
source $SCRIPT_DIR/.env.cluster
source $SCRIPT_DIR/../.env.base

# make sure that all directories exists in cache directory
setup_directories
# copy all cache files
cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR

# make sure logs directory exists (in the permanent isaaclab directory)
mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
touch "$CLUSTER_ISAACLAB_DIR/logs/.keep"

# copy the temporary isaaclab directory with the latest changes to the compute node
cp -r $1 $TMPDIR
# Get the directory name
dir_name=$(basename "$1")

cp -r $CLUSTER_RSL_RL_DIR $TMPDIR/$dir_name/compositional_rsl_rl

# copy container to the compute node
tar -xf $CLUSTER_SIF_PATH/$2.tar  -C $TMPDIR

# execute command in singularity container
# Use the container's uv environment with the synced checkout.
mkdir -p $TMPDIR/ctmp
apptainer exec \
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B $TMPDIR/$dir_name:/workspace/isaaclab:rw \
    -B $CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/logs:rw \
    -B $TMPDIR/ctmp:/tmp:rw \
    --nv --writable --containall $TMPDIR/$2.sif \
        bash -c "cd /workspace/isaaclab && export UV_CACHE_DIR=/tmp/uv_cache UV_LINK_MODE=copy && mkdir -p /tmp/uv_cache && uv pip install "wandb==0.19.11" --python /opt/isaaclab-venv/bin/python && (ls -d wandb 2>/dev/null || true) && /opt/isaaclab-venv/bin/python -c 'import wandb, wandb.util; print(wandb.__version__, wandb.__file__, hasattr(wandb.util, \"generate_id\"))' && export WANDB_API_KEY=${WANDB_API_KEY} && export WANDB_MODE=${WANDB_MODE} && export ISAACLAB_PATH=/workspace/isaaclab && export WARP_CACHE_PATH=/tmp/warp_cache && export PYTHONFAULTHANDLER=1 && mkdir -p /tmp/warp_cache && ln -sfn ${DOCKER_ISAACSIM_ROOT_PATH} _isaac_sim && uv pip install --no-deps -e /workspace/isaaclab/compositional_rsl_rl --python /opt/isaaclab-venv/bin/python && uv run --no-sync isaaclab -p ${CLUSTER_PYTHON_EXECUTABLE} ${*:3}"
# copy resulting cache files back to host
rsync -azPv $TMPDIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..

# if defined, remove the temporary isaaclab directory pushed when the job was submitted
if $REMOVE_CODE_COPY_AFTER_JOB; then
    rm -rf $1
fi

echo "(run_singularity.py): Return"
-