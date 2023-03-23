# qrw

Implementation of a reactive walking controller for quadruped robots. Architecture mainly in Python with some parts in C++ with bindings to Python.

## Dependencies

### Common dependencies

* [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
* [Crocoddyl](https://github.com/loco-3d/crocoddyl)
* [example-robot-data](https://github.com/Gepetto/example-robot-data) | apt: `sudo apt install robotpkg-example-robot-data`
* Install Scipy, Numpy, Matplotlib, IPython
* [Sobec](https://github.com/MeMory-of-MOtion/sobec)
* Bullet: `pip install --user pybullet`
* [`yaml-cpp`](https://github.com/jbeder/yaml-cpp) | apt: `sudo apt install libyaml-cpp-dev`

### Dependencies for real system

* Install package that handles the gamepad: `pip install --user inputs`
* [odri_control_interface](https://github.com/open-dynamic-robot-initiative/odri_control_interface)

## Compilation instructions

```bash
# Initialize cmake submodules:
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=your/install/prefix
# Compile Python bindings:
cd build && make install
```

## Run the simulation

* Run `python -m quadruped_reactive_walking.main_solo12_control` while being in the `scripts` folder
* Sometimes the parallel process that runs the MPC does not terminate properly so it will keep running in the background forever, you can manually end all python processes with `pkill -9 python3`

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
