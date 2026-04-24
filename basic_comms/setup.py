from setuptools import find_packages, setup

package_name = 'basic_comms'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='strpicket',
    maintainer_email='strpicket@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'talker = basic_comms.talker:main',
            'listener = basic_comms.listener:main',
		    'center =  basic_comms.centerAruco:main',
            'waypoints = basic_comms.waypoints:main',
            'perception = basic_comms.perception:main',
            'poseEstimation = basic_comms.poseEstimation:main',
            'monteCarlo = basic_comms.monteCarlo:main',
        ],
    },
)
