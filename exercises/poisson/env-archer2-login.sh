

OPENMP_OFFLOAD_COURSE_ROOT="${HOME/home/work}/.openmp_offload_course"

run_gpu()
    {
        srun --export=PATH --time=01:00:00 --partition=gpu --qos=gpu-shd --gpus=1 $1
    }

if [ ! -d "$OPENMP_OFFLOAD_COURSE_ROOT/python_env" ]; then
    echo "Python testing virtual environment not found, creating it now..."
    python3 -m venv "$OPENMP_OFFLOAD_COURSE_ROOT/python_env"
    source "$OPENMP_OFFLOAD_COURSE_ROOT/python_env/bin/activate"
    pip install -r requirements.txt
    source "$OPENMP_OFFLOAD_COURSE_ROOT/python_env/bin/activate"
else
    source "$OPENMP_OFFLOAD_COURSE_ROOT/python_env/bin/activate"
fi

