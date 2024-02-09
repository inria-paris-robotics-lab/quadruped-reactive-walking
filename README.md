# qrw

Implementation of a reactive walking controller for quadruped robots. Architecture mainly in Python with some parts in C++ with bindings to Python.

## Dependencies

This package requires Python 3.8 and above and a C++14 compliant compiler.

### Common dependencies

* [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
* [Crocoddyl](https://github.com/loco-3d/crocoddyl)
* [example-robot-data](https://github.com/Gepetto/example-robot-data) | apt: `sudo apt install robotpkg-example-robot-data`
* Install Scipy, Numpy, Matplotlib, IPython
* Bullet: `pip install --user pybullet`
* [`yaml-cpp`](https://github.com/jbeder/yaml-cpp) | apt: `sudo apt install libyaml-cpp-dev`

### Dependencies for real system

* Install package that handles the gamepad: `pip install --user inputs`
* [odri_control_interface](https://github.com/open-dynamic-robot-initiative/odri_control_interface)

## Installation
### For running locally - No _Motion Server_

1. Clone this repo
```bash
mkdir ~/qrw_ws/
cd ~/qrw_ws
git clone --recursive https://github.com/inria-paris-robotics-lab/quadruped-reactive-walking.git
```

2. Create conda environment.
(It is recommended to use `mamba` instead of `conda` for faster/better dependencies solving)
```bash
mamba env create -f quadruped-reactive-walking/environment.yaml
mamba activate qrw
```

3. Download dependencies
(Some dependencies are not available on conda, or not with adequate versions)
```bash
vcs import --recursive < quadruped-reactive-walking/git-deps.yaml
```

4. Build
```bash
for dir in ndcurves quadruped-reactive-walking ; do # loop over each directory
    mkdir -p $dir/build
    cd $dir/build
    cmake .. -DCMAKE_BUILD_TYPE=Release             \
             -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX   \
             -DCMAKE_PREFIX_PATH=$CONDA_PREFIX      \
             -DBUILD_TESTING=OFF                    `# For faster build`             \
             -DCMAKE_CXX_COMPILER_LAUNCHER='ccache' `# For faster build`             \
             -DBUILD_PYTHON_INTERFACE=ON            `# Generate py bindings`         \
             -DPYTHON_EXECUTABLE=$(which python)    `# Generate propper py bindings` \
             -DGENERATE_PYTHON_STUBS=OFF
    make install -j
    cd ../../
done
```

### With ROS - for _Motion Server_ support

1. Clone this repo
```bash
mkdir ~/qrw_catkin_ws/src
cd ~qrw_catkin_ws
git clone --recursive https://github.com/inria-paris-robotics-lab/quadruped-reactive-walking.git src/quadruped-reactive-walking
```

2. Create conda environment.
(It is recommended to use `mamba` instead of `conda` for faster/better dependencies solving)
```bash
mamba env create -f src/quadruped-reactive-walking/ros-environment.yaml
mamba activate qrw-ros
```

3. Download dependencies
(ROS imposes fixed versions for many packages. Thus most of _qrw_ dependencies need to be built manually, because they are not available on conda for this particular set sub dependencies.)
```bash
vcs import --recursive < src/quadruped-reactive-walking/ros-git-deps.yaml
```

4. Build

(Note for extra performances, export the following variable before building `export CXXFLAGS="$CXXFLAGS -march=native"`, but might cause Segmentation Faults with certain boost versions)

```bash
catkin build --cmake-args -DBUILD_TESTING=OFF                    `# For faster build`               \
                          -DBUILD_BENCHMARK=OFF                  `# For faster build`               \
                          -DBUILD_BENCHMARKS=OFF                 `# For faster build (aligator)`    \
                          -DBUILD_EXAMPLES=OFF                   `# For faster build`               \
                          -DCMAKE_CXX_COMPILER_LAUNCHER='ccache' `# For faster build`               \
                          -DBUILD_WITH_MULTITHREADS=ON           `# Enable parallelization (croc)`  \
                          -DBUILD_WITH_OPENMP_SUPPORT=ON         `# Enable parallelization (algtr)` \
                          -DBUILD_CROCODDYL_COMPAT=ON            `# Aligator compatibility flag`    \
                          -DBUILD_WITH_COLLISION_SUPPORT=ON      `# Pinocchio flag`                 \
                          -DBUILD_PYTHON_INTERFACE=ON            `# Generate py bindings`           \
                          -DPYTHON_EXECUTABLE=$(which python)    `# Generate propper py bindings`   \
                          -DBUILD_WITH_ROS_SUPPORT=ON            `# Generate QRW custom ros msgs`   \
                          -DGENERATE_PYTHON_STUBS=OFF            \
                          -DCMAKE_CXX_FLAGS="$CXXFLAGS -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION" `# for compatibility between old boost (from ros) and modern c++`
```

5. Source worskpace (Needs to be repeated for every new terminal)
```bash
source ~/qrw_catkin_ws/devel/setupb.bash # Linux users
source ~/qrw_catkin_ws/devel/setupb.zsh  # Mac users
```

## Run the simulation

* After installing the package, run `python -m quadruped_reactive_walking.main_solo12_control`

## Tune the simulation

* In `main_solo12_control.py`, you can change some of the parameters defined at the beginning of the `control_loop` function or by passing command-line arguments.
* To see which CLI arguments are available, run

  ```bash
  python -m quadruped_reactive_walking.main_solo12_control --help
  ```

* Set `env_id` to 1 to load obstacles and stairs.
* Set `use_flat_plane` to False to load a ground with lots of small bumps.
* If you have a gamepad you can control the robot with two joysticks by turning `predefined_vel` to False in `main_solo12_control.py`. Velocity limits with the joystick are defined in `Joystick.py` by `self.VxScale` (maximul lateral velocity), `self.VyScale` (maximum forward velocity) and `self.vYawScale` (maximum yaw velocity).
* If `predefined_vel = True` the robot follows the reference velocity pattern. Velocity patterns are defined in walk_parameters, you can modify them or add new ones. Each profile defines forward, lateral and yaw velocities that should be reached at the associated loop iterations (in `self.k_switch`). There is an automatic interpolation between milestones to have a smooth reference velocity command.
