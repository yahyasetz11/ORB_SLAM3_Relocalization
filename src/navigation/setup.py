from setuptools import find_packages, setup

package_name = 'navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            ['navigation/config/locations.yaml',
             'navigation/config/navigation_params.yaml']),
        ('share/' + package_name + '/launch',
            ['launch/test_navigation.launch.py',
             'launch/nav.launch.py']),
    ],
    install_requires=['setuptools', 'numpy', 'opencv-python', 'cv_bridge'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'navigation = navigation.main:main',
            'navigation_ui = navigation.ui.map_ui_node:main',
            'fake_data_publisher = navigation.test.fake_data_publisher:main',
            'map_publisher = navigation.map_constructor.map_publisher:main',
        ],
    },
)