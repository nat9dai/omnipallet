import glob

from setuptools import setup

package_name = "stock_drone"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob.glob("launch/*.py")),
        ("share/" + package_name + "/config", glob.glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Nuthasith Gerdpratoom",
    maintainer_email="nuthasith.g@gmail.com",
    description="Warehouse stock-counting quadrotor nodes",
    license="MIT",
    entry_points={
        "console_scripts": [
            "mission = stock_drone.mission_node:main",
        ],
    },
)
