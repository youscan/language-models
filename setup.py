from setuptools import find_packages, setup

setup(
    name="src",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
)
