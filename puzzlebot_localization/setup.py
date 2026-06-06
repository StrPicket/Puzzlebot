from setuptools import find_packages, setup

package_name = 'puzzlebot_localization'

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
            'monte_carlo      = puzzlebot_localization.monteCarlo:main',
            'odometry         = puzzlebot_localization.odometry:main',
            'particle_filter  = puzzlebot_localization.particle_filter:main',
            'pose_estimation  = puzzlebot_localization.poseEstimation:main',
            'soloArucos       = puzzlebot_localization.soloArucos:main',
            'soloKalman       = puzzlebot_localization.soloKalman:main',
        ],
    },
)
