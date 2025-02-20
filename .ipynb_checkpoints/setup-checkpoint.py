from setuptools import setup, find_packages

with open("requirements.txt", 'r') as ifile:
    requirements = ifile.read().splitlines()

nb_requirements = [
    'nbconvert>=6.1.0',
    'nbformat>=5.1.3',
    'notebook>=6.4.11',
    'jupyter>=7.0.0',
    'jupyterlab>=3.4.3',
    'ipython>=7.27.0',
]

setup(
    name="Squidiff",
    version="1.0.0",
    description="Diffusion model-based generative framework designed to predict transcriptomic changes across diverse cell types in response to a wide range of environmental changes.",
    authors=["Siyu He"],
    url="https://Squidiff.readthedocs.io",
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],

    dependency_links=[
        'https://girder.github.io/large_image_wheels'
    ]

    # extras_require={
    #    'notebooks': nb_requirements,
    #    'dev': open('dev-requirements.txt').read().splitlines(),
    # }
)