import os
import sys
import subprocess
import venv

def run_command(command, cwd=None):
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(command)}")
        print(result.stdout)
        print(result.stderr)
        return False
    return True

def main():
    print("=== Setting up GNN XAI Robustness Environment ===")
    
    # 1. Create virtual environment
    venv_dir = os.path.join(os.getcwd(), ".venv")
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment in {venv_dir}...")
        venv.create(venv_dir, with_pip=True)
    else:
        print("Virtual environment already exists.")
        
    # Determine the python/pip paths inside virtualenv
    if os.name == 'nt': # Windows
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        venv_pip = os.path.join(venv_dir, "Scripts", "pip.exe")
    else: # Unix/macOS
        venv_python = os.path.join(venv_dir, "bin", "python")
        venv_pip = os.path.join(venv_dir, "bin", "pip")
        
    # 2. Upgrade pip
    print("Upgrading pip inside virtual environment...")
    if not run_command([venv_python, "-m", "pip", "install", "--upgrade", "pip"]):
        sys.exit(1)
        
    # 3. Install packages
    print("Installing requirements from requirements.txt...")
    if not run_command([venv_pip, "install", "-r", "requirements.txt"]):
        print("Standard pip install failed. Retrying with --no-cache-dir...")
        if not run_command([venv_pip, "install", "--no-cache-dir", "-r", "requirements.txt"]):
            sys.exit(1)
            
    # 4. Verify installation
    print("Verifying installation...")
    verification_code = """
import sys
try:
    import torch
    print(f"torch version: {torch.__version__}")
    import torch_geometric
    print(f"torch_geometric version: {torch_geometric.__version__}")
    print("Success: Dependencies loaded successfully!")
except Exception as e:
    print(f"Verification Failed: {e}", file=sys.stderr)
    sys.exit(1)
"""
    if run_command([venv_python, "-c", verification_code]):
        print("\n=== Setup Complete! ===")
        print(f"To activate the virtual environment, run:")
        if os.name == 'nt':
            print(f"  .venv\\Scripts\\activate")
        else:
            print(f"  source .venv/bin/activate")
    else:
        print("\n=== Setup Failed verification check ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
