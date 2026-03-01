from setuptools import setup, find_packages
import os

def find_package_data(package_dir):
    paths = []
    for root, dirs, files in os.walk(package_dir):
        for file in files:
            if file.endswith('.so') or file.endswith('.pyd') or file.endswith('.py'):
                paths.append(os.path.relpath(os.path.join(root, file), package_dir))
    return paths

setup(
    name="jaxlib",
    version="0.3.22",
    packages=find_packages(),
    package_data={
        "jaxlib": find_package_data("jaxlib"),
        "": ["*.so", "*.pyd", "*.py"],
    },
    include_package_data=True,
)