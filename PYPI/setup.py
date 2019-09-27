import setuptools

with open("README_PYPI.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
	# Information about project
    name="boston_housing_prediction",
    version="0.2.2a0",
    author="Lupos",
    author_email="buisnessgithublupos@gmail.com",
	url="https://github.com/LuposX/BostonHousingPrediction",
	license="MIT",
	keywords="regression meachine-learning housing_prices_boston learning_purpose",
	
	# Description
	description="Predict housing prices in boston.",
    long_description=long_description,
    long_description_content_type="text/markdown",
	
	# What is in our module/scripts
	# py_modules=["misc_libary", "polynomial_regression_libary", "linear_regression_libary"],
	scripts=["boston_housing_prediction/__main__.py"],
	packages=["boston_housing_prediction"], 
    entry_points = {
        'console_scripts': [
            'boston_housing_prediction = boston_housing_prediction.__main__'
        ]
    },

	# Dependence and required stuff
	install_requires=requirements,
    python_requires='>=3.5',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
		'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
)
