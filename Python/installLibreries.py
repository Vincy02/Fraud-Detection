import subprocess
import sys
import importlib.metadata

try:
    import tqdm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    import tqdm

installed_packages = [i.metadata["Name"].lower() for i in importlib.metadata.distributions()]

# Funzione per installare un package specifico
def install(package):
    if package.lower() not in installed_packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

# Funzione per installare le librerie necessarie
def installPackages():
    packages = []
    with open("requirements.txt", "r") as file:
        packages = file.readlines()
    packages = [s.strip() for s in packages]
    print("Checking & Installing requirements")
    for package in tqdm.tqdm(packages):
        install(package)
    print("\nRequirements installed & checked\n")