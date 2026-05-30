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
            'center_aruco        = puzzlebot_navigation.centerAruco:main',
            'center_qr           = puzzlebot_navigation.centerQR:main',
            'waypoints           = puzzlebot_navigation.waypoints:main',
            'waypoint_controller = puzzlebot_navigation.waypoint_controller:main',
        ],
    },
)
