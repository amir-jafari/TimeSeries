from setuptools import setup, find_packages

setup (
    author="S, J, H",
    description="Complete package for Times Series Analysis",
    name="TimeSeriesSRC",
    version="0.1.0",
    packages=find_packages(include=["TimeSeriesSCR", "TimeSeriesSRC.basefunctions", "TimeSeriesSRC.TimeSeries", "TimesSeriesSRC.Model"]),
    install_requires=["NumPy", "matplotlib","scipy"]

)