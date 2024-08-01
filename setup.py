import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='gofi',
                 version='0.0.1',
                 description='Open-source implementation of the occlusion-supporting '
                             'goal recognition and motion planning algorithm GOFI '
                             'from the paper: Interpretable Goal Recognition in the '
                             'Presence of Occluded Factors for Autonomous Vehicles',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 author='Balint Gyevnar',
                 author_email='balint.gyevnar@ed.ac.uk',
                 url='https://github.com/uoe-agents/GOFI',
                 packages=setuptools.find_packages(exclude=["scenarios", "scripts"]),
                 install_requires=requirements
                 )
