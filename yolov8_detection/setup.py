from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'yolov8_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        ('share/' + package_name, ['package.xml']),

        (
            os.path.join('share', package_name, 'models'),
            glob('models/*')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='strpicket',
    maintainer_email='ejohns.ipod@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'segmentation_test = yolov8_detection.segmentation_test:main'
        ],
    },
)