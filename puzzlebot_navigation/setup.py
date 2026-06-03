from setuptools import find_packages, setup

package_name = 'puzzlebot_navigation'

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
            'centerQR           = puzzlebot_navigation.centerQR:main',
            'waypoints           = puzzlebot_navigation.waypoints:main',
            'semantic_planner    = puzzlebot_navigation.semantic_planner:main',
            'bug2_monitor        = puzzlebot_navigation.bug2_monitor:main',
            'waypoint_controller = puzzlebot_navigation.waypoint_controller:main',
            'mission_manager     = puzzlebot_navigation.mission_manager:main',
        ],
    },
)
