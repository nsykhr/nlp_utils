import setuptools

with open('readme.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="nlp-utils",
    version="0.1.0",
    author="Nikita Sykhrannov",
    author_email="Sykhrannov.N.E@omega.sbrf.ru",
    description="A collection of NLP models for the LegalTech solutions team.",
    long_description=long_description,
    long_description_conttype="text/markdown",
    packages=setuptools.find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
    entry_points={'console_scripts': ["components=components.run:run"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    zip_safe=False,
)
