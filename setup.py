from setuptools import setup, find_packages

setup(
    name='vispyx',
    version='0.1.0',
    description='Paquete Python para procesamiento general de imágenes y video',
    author='Tu Nombre',
    author_email='tu@email.com',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'scikit-image',
        'matplotlib'
    ],
    entry_points={
        'console_scripts': [
            'vispyx=vispyx.cli:main',
        ],
    },
    python_requires='>=3.7',
)
