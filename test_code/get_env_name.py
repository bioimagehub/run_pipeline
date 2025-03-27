import os

def get_conda_env_name():
    conda_env = os.getenv('CONDA_DEFAULT_ENV')
    if conda_env:
        return conda_env
    return 'Not running in a Conda environment'

print("Getting Conda Environment Name...")
env_name = get_conda_env_name()
print(f"Conda Environment Name: {env_name}")
