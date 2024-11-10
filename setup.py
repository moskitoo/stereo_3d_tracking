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
            'image_publisher_node = stereo_3d_tracking.frame_publisher:main',
        ],
    },
    script_dir='scripts',  # Example if you have custom scripts to install
    install_scripts='scripts',  # Example if you are installing scripts
)
