from setuptools import setup

setup(
    name="physix",
    version="0.1",
    packages=["physix"],
    install_required=["pandas", "numpy", "scipy", "uncertainties"],
    package_data={"physix": ["data/*"]}
)
