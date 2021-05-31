from setuptools import find_packages, setup

setup(
    name="language-modelling",
    version="0.2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
)
