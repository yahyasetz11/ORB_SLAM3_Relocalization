from setuptools import find_packages, setup

package_name = 'yolo_bbox'

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
    maintainer='orange',
    maintainer_email='orange@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_node = yolo_bbox.camera_node:main',
            'bbox_reciever = yolo_bbox.bbox_reciever:main',
            'yolo_bbox_image = yolo_bbox.yolo_bbox_image:main',
            'yolo_bbox_video = yolo_bbox.yolo_bbox_video:main',
            'yolo_bbox_webcam = yolo_bbox.yolo_bbox_webcam:main',
            'yolo_bbox_receiver = yolo_bbox.yolo_bbox_receiver:main'
        ],
    },
)
