import setuptools

with open("README_PYPI.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="boston-housing-prediction",
    version="0.2.2a0",
    author="Lupos",
    author_email="buisnessgithublupos@gmail.com",
	py_modules=["misc_libary", "polynomial_regression_libary", "linear_regression_libary"],
	scripts=["boston-housing-main.py"],
    description="Predict housing prices in boston.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LuposX/BostonHousingPrediction",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
		'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
	keywords="regression meachine-learning housing_prices_boston learning_purpose",
	license="MIT",
	install_requires=requirements,
    python_requires='>=3.5',
)
