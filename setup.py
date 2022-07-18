from setuptools import setup, find_packages
from distutils.util import convert_path

# Get version
main_ns = {}
version_path = convert_path("chemist/version.py")
with open(version_path) as handle:
	exec(handle.read(), main_ns)

# Set up
setup(
	name="chemist",
	version=main_ns['__version__'],
	description="VAE base reinforcement learning molecular generator .",
	url="https://github.com/bpmunson/chemist",
	author="Brenton Munson",
	author_email="bpmunson@eng.ucsd.edu",
    liscense="MIT",
	classifiers=[
		# How mature is this project? Common values are
		#   3 - Alpha
		#   4 - Beta
		#   5 - Production/Stable
		'Development Status :: 4 - Beta',

		# Indicate who your project is intended for
		'Intended Audience :: Developers',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Bio-Informatics',

		# Pick your license as you wish (should match "license" above)
		'License :: OSI Approved :: MIT License',

		# Specify the Python versions you support here. In particular, ensure
		# that you indicate whether you support Python 2, Python 3 or both.
		'Programming Language :: Python :: 3.6',
	], 
	entry_points={
		'console_scripts': [
			'chemist=chemist.run:main',
		]
	},
	keywords='',
	packages=find_packages(),
  	install_requires=['pandas>=1.0.3','numpy>=1.18.1','rdkit>=2019.09.3','torch>=1.4.0','joblib>=0.14.1','scikit-learn>=0.22.1'],
	include_package_data=True,
)


