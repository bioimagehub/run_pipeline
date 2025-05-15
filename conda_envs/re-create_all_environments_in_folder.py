import os
import subprocess
import argparse

def install_conda_envs_from_yml(folder_path, reinstall=False):
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".yml"):
                env_name = extract_env_name(os.path.join(folder_path, file_name))
                if env_name:
                    print(f"Processing Environment: {env_name}")
                    if reinstall:
                        # Remove existing conda environment if it exists
                        remove_existing_env(env_name)
                    # Create the new conda environment
                    create_conda_env(os.path.join(folder_path, file_name))
    elif os.path.isfile(folder_path) and folder_path.endswith('.yml'):
        env_name = extract_env_name(folder_path)
        if env_name:
            print(f"Processing Environment: {env_name}")
            if reinstall:
                # Remove existing conda environment if it exists
                remove_existing_env(env_name)
            # Create the new conda environment
            create_conda_env(folder_path)

def extract_env_name(yml_file_path):
    """Extract the environment name from the YAML file by searching for the 'name' key."""
    with open(yml_file_path, 'r') as file:
        for line in file:
            if line.strip().startswith('name:'):
                return line.split(':', 1)[1].strip()  # Get the value after 'name:'
    return None  # Return None if no name is found

def remove_existing_env(env_name):
    """Remove an existing conda environment."""
    try:
        print(f"Removing existing environment: {env_name}...")
        subprocess.run(["conda", "env", "remove", "--name", env_name, "--yes"], check=True)
        print(f"Environment {env_name} removed.")
    except subprocess.CalledProcessError:
        print(f"No existing environment {env_name} found. Continuing...")

def create_conda_env(yml_file_path):
    """Create a conda environment from the YAML file."""
    try:
        print(f"Creating environment from {yml_file_path}...")
        subprocess.run(["conda", "env", "create", "-f", yml_file_path], check=True)
        print("Environment created successfully.")
    except subprocess.CalledProcessError as e:
        if reinstall:
            print(f"Failed to create environment from {yml_file_path}: {str(e)}")
        else:
            print(f"Found {yml_file_path} environment. Skipping. Use --reinstall to replace")
    print("")

def delete_environments(folder_path):
    """Delete all environments defined in the YAML files under the given path."""
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".yml"):
                env_name = extract_env_name(os.path.join(folder_path, file_name))
                if env_name:
                    remove_existing_env(env_name)
    elif os.path.isfile(folder_path) and folder_path.endswith('.yml'):
        env_name = extract_env_name(folder_path)
        if env_name:
            remove_existing_env(env_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install Conda environments from YAML files.")
    parser.add_argument("environment_paths", type=str, nargs='?', 
                        help="Path to a directory with .yml files or a single .yml file (optional).")
    parser.add_argument("--reinstall", action="store_true", 
                        help="If passed, remove existing environments before installing.")
    parser.add_argument("--delete", action="store_true", 
                        help="If passed, remove all existing environments defined in .yml files.")

    args = parser.parse_args()
    
    # Get the path of the provided argument or default to the current directory
    environment_path = args.environment_paths if args.environment_paths else os.getcwd()

    if args.delete:
        print("Deleting environments...")
        delete_environments(environment_path)
    else:
        reinstall = args.reinstall
        install_conda_envs_from_yml(environment_path, reinstall)
