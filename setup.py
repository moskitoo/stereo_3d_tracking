from setuptools import setup

package_name = 'stereo_3d_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=['stereo_3d_tracking'],
    install_requires=['setuptools', 'opencv-python', 'cv-bridge'],
    data_files=[],
    entry_points={
    'console_scripts': [
        'image_publisher_node = stereo_3d_tracking.image_publisher_node:main',
        'calibration_publisher_node = stereo_3d_tracking.calibration_publisher_node:main'
    ],
    },
    script_dir='scripts',
    install_scripts='scripts',
)