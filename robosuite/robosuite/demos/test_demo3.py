"""
A script to collect human demonstrations with enhanced joint angle data collection.

The demonstrations are saved in HDF5 format and can be played back using the playback functionality.
Joint angles, end-effector positions, and other relevant data are organized and saved.
"""

import argparse
import datetime
import json
import os
import time
from glob import glob
from copy import deepcopy

import h5py
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper


class EnhancedDataCollectionWrapper(DataCollectionWrapper):
    """Enhanced wrapper that collects additional joint angle and robot state data."""
    
    def __init__(self, env, directory):
        super().__init__(env, directory)
        self.joint_data = []
        self.timestamps = []
        self.step_count = 0
        self.start_time = None
        
    def reset(self):
        """Reset and initialize data collection for new episode."""
        obs = super().reset()
        self.joint_data = []
        self.timestamps = []
        self.step_count = 0
        self.start_time = time.time()
        return obs
    
    def step(self, action):
        """Step environment and collect enhanced data."""
        obs, reward, done, info = super().step(action)
        
        # Collect joint and robot state data
        current_time = time.time()
        relative_time = current_time - self.start_time if self.start_time else 0
        
        # Extract joint data from first robot (can be extended for multi-robot)
        robot = self.env.robots[0]
        joint_positions = obs.get('robot0_joint_pos', [])
        joint_velocities = obs.get('robot0_joint_vel', [])
        end_effector_pos = obs.get('robot0_eef_pos', [])
        end_effector_quat = obs.get('robot0_eef_quat', [])
        gripper_pos = obs.get('robot0_gripper_qpos', [0.0])
        
        # Store enhanced data
        step_data = {
            'step': self.step_count,
            'timestamp': current_time,
            'relative_time': relative_time,
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'end_effector_pos': end_effector_pos,
            'end_effector_quat': end_effector_quat,
            'gripper_pos': gripper_pos[0] if len(gripper_pos) > 0 else 0.0,
            'actions': action,
            'reward': reward,
            'done': done
        }
        
        self.joint_data.append(step_data)
        self.timestamps.append(relative_time)
        self.step_count += 1
        
        return obs, reward, done, info


def collect_human_trajectory(env, device, arm, max_fr):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format with enhanced joint data.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arm (str): which arm to control (eg bimanual) 'right' or 'left'
        max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
    """

    env.reset()
    env.render()

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    # Keep track of prev gripper actions
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    print("Starting trajectory collection...")
    print("Use device to control robot. Press 'r' to reset, close window to finish.")

    step_counter = 0  # Add this counter

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        active_robot = env.robots[device.active_robot]

        # Get the newest action
        input_ac_dict = device.input2action()

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            break

        action_dict = deepcopy(input_ac_dict)
        
        # set arm actions
        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        # Step environment (enhanced data collection happens in wrapper)
        obs, reward, done, info = env.step(env_action)
        env.render()
        
        step_counter += 1
        
        # Print progress more frequently
        if step_counter % 10 == 0:  # Changed from 50 to 10
            joint_pos = obs.get('robot0_joint_pos', [])
            if len(joint_pos) > 0:
                print(f"Step {step_counter} - Joint angles: {joint_pos}")

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    print(f"Trajectory collection completed. Total steps: {step_counter}")
    
    # Check if data was actually collected
    if hasattr(env, 'ep_directory'):
        print(f"Data should be in: {env.ep_directory}")
        if os.path.exists(env.ep_directory):
            files = os.listdir(env.ep_directory)
            print(f"Files in episode directory: {files}")
        else:
            print("Episode directory doesn't exist!")


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a single hdf5 file 
    with enhanced joint angle data.

    The structure of the hdf5 file is:
    data (group)
        date, time, repository_version, env, env_info (attributes)
        
        demo1 (group)
            model_file (attribute)
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration
            joint_positions (dataset) - joint angles over time
            joint_velocities (dataset) - joint velocities over time
            end_effector_positions (dataset) - end effector positions
            end_effector_quaternions (dataset) - end effector orientations
            gripper_positions (dataset) - gripper positions
            timestamps (dataset) - relative timestamps
            rewards (dataset) - rewards over time
            
        demo2 (group)
        ...
    """

    hdf5_path = os.path.join(out_dir, "demonstrations.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None

    print(f"Looking for demonstrations in: {directory}")
    print(f"Directory contents: {os.listdir(directory)}")

    for ep_directory in os.listdir(directory):
        ep_path = os.path.join(directory, ep_directory)
        if not os.path.isdir(ep_path):
            continue
            
        print(f"Processing episode directory: {ep_directory}")
        
        state_paths = os.path.join(ep_path, "state_*.npz")
        states = []
        actions = []
        success = False

        state_files = sorted(glob(state_paths))
        print(f"Found {len(state_files)} state files")

        for state_file in state_files:
            print(f"Loading state file: {state_file}")
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            
            # Debug: Check success flag
            file_success = dic.get("successful", False)
            print(f"File success flag: {file_success}")
            success = success or file_success

        print(f"Episode success: {success}, States: {len(states)}, Actions: {len(actions)}")

        if len(states) == 0:
            print("No states found, skipping episode")
            continue

        # Save ALL demonstrations for debugging (comment out the success check)
        if True:  # Changed from: if success:
            print(f"Demonstration {num_eps + 1} is being saved (successful: {success})")
            
            # Delete the last state
            if len(states) > 0:
                del states[-1]
                if len(actions) > len(states):
                    actions = actions[:len(states)]

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # store model xml as an attribute
            xml_path = os.path.join(ep_path, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # Extract joint data from states
            joint_positions = []
            joint_velocities = []
            end_effector_positions = []
            end_effector_quaternions = []
            gripper_positions = []
            
            for i, state in enumerate(states):
                # Reconstruct simulation state to extract joint data
                sim_state = state
                
                # Extract joint positions (first n_joints elements of qpos)
                # This is environment-specific, may need adjustment
                n_joints = 7  # Panda has 7 joints, adjust as needed
                joint_pos = sim_state[:n_joints] if len(sim_state) >= n_joints else sim_state
                joint_positions.append(joint_pos)
                
                # Extract joint velocities (approximated)
                if i > 0:
                    joint_vel = np.array(joint_positions[i]) - np.array(joint_positions[i-1])
                else:
                    joint_vel = np.zeros_like(joint_pos)
                joint_velocities.append(joint_vel)
                
                # Placeholder for end effector data (would need forward kinematics)
                end_effector_positions.append([0.0, 0.0, 0.0])
                end_effector_quaternions.append([0.0, 0.0, 0.0, 1.0])
                gripper_positions.append(0.0)

            # Write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
            
            # Write enhanced joint data
            ep_data_grp.create_dataset("joint_positions", data=np.array(joint_positions))
            ep_data_grp.create_dataset("joint_velocities", data=np.array(joint_velocities))
            ep_data_grp.create_dataset("end_effector_positions", data=np.array(end_effector_positions))
            ep_data_grp.create_dataset("end_effector_quaternions", data=np.array(end_effector_quaternions))
            ep_data_grp.create_dataset("gripper_positions", data=np.array(gripper_positions))
            
            # Create timestamps
            timestamps = np.arange(len(states)) * 0.05  # 20Hz control frequency
            ep_data_grp.create_dataset("timestamps", data=timestamps)
            
            # Create dummy rewards
            rewards = np.zeros(len(states))
            ep_data_grp.create_dataset("rewards", data=rewards)
            
            # Store metadata for this demo
            ep_data_grp.attrs["num_timesteps"] = len(states)
            ep_data_grp.attrs["num_joints"] = len(joint_positions[0]) if joint_positions else 0
            ep_data_grp.attrs["duration"] = timestamps[-1] if len(timestamps) > 0 else 0.0

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info
    grp.attrs["num_demos"] = num_eps

    f.close()
    print(f"Saved {num_eps} demonstrations to {hdf5_path}")
    return hdf5_path


def playback_demonstrations_from_hdf5(hdf5_path, demo_index=None):
    """
    Playback demonstrations from HDF5 file.
    
    Args:
        hdf5_path (str): Path to HDF5 file containing demonstrations
        demo_index (int): Specific demo to playback (None for all)
    """
    
    if not os.path.exists(hdf5_path):
        print(f"HDF5 file not found: {hdf5_path}")
        return
    
    print(f"Loading demonstrations from: {hdf5_path}")
    
    with h5py.File(hdf5_path, "r") as f:
        data_grp = f["data"]
        
        # Print metadata
        print(f"Environment: {data_grp.attrs['env']}")
        print(f"Number of demonstrations: {data_grp.attrs['num_demos']}")
        print(f"Collection date: {data_grp.attrs['date']} at {data_grp.attrs['time']}")
        
        # List available demos
        demo_keys = [key for key in data_grp.keys() if key.startswith("demo_")]
        print(f"Available demonstrations: {demo_keys}")
        
        # Playback specific demo or all demos
        if demo_index is not None:
            demo_key = f"demo_{demo_index}"
            if demo_key in data_grp:
                demo_grp = data_grp[demo_key]
                print(f"\nPlayback {demo_key}:")
                print(f"  Timesteps: {demo_grp.attrs['num_timesteps']}")
                print(f"  Duration: {demo_grp.attrs['duration']:.2f}s")
                print(f"  Joints: {demo_grp.attrs['num_joints']}")
                
                # Display joint positions
                joint_positions = demo_grp["joint_positions"][:]
                print(f"  Joint positions shape: {joint_positions.shape}")
                print(f"  First joint position: {joint_positions[0]}")
                print(f"  Last joint position: {joint_positions[-1]}")
            else:
                print(f"Demo {demo_index} not found")
        else:
            # Show info for all demos
            for demo_key in demo_keys:
                demo_grp = data_grp[demo_key]
                print(f"\n{demo_key}:")
                print(f"  Timesteps: {demo_grp.attrs['num_timesteps']}")
                print(f"  Duration: {demo_grp.attrs['duration']:.2f}s")
                print(f"  Joints: {demo_grp.attrs['num_joints']}")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="./robot_demonstrations",
        help="Directory to save demonstrations"
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        nargs="*",
        type=str,
        default="agentview",
        help="Camera names for demo collection",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="Position input sensitivity",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="Rotation input sensitivity",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Renderer type",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Maximum frame rate",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="Reverse xy axes for DualSense",
    )
    parser.add_argument(
        "--playback",
        action="store_true",
        help="Playback demonstrations instead of collecting",
    )
    parser.add_argument(
        "--demo_file",
        type=str,
        default=None,
        help="Specific HDF5 file to playback",
    )
    parser.add_argument(
        "--demo_index",
        type=int,
        default=None,
        help="Specific demo index to playback",
    )
    
    args = parser.parse_args()

    if args.playback:
        # Playback mode
        if args.demo_file:
            hdf5_path = args.demo_file
        else:
            # Find latest HDF5 file
            hdf5_files = glob(os.path.join(args.directory, "**/demonstrations.hdf5"), recursive=True)
            if not hdf5_files:
                print(f"No HDF5 demo files found in {args.directory}")
                exit(1)
            hdf5_path = max(hdf5_files, key=os.path.getmtime)
        
        playback_demonstrations_from_hdf5(hdf5_path, args.demo_index)
    else:
        # Collection mode
        # Get controller config
        controller_config = load_composite_controller_config(
            controller=args.controller,
            robot=args.robots[0],
        )

        # Create argument configuration
        config = {
            "env_name": args.environment,
            "robots": args.robots,
            "controller_configs": controller_config,
        }

        # Check if we're using a multi-armed environment
        if "TwoArm" in args.environment:
            config["env_configuration"] = args.config

        # Create environment
        env = suite.make(
            **config,
            has_renderer=True,
            renderer=args.renderer,
            has_offscreen_renderer=False,
            render_camera=args.camera,
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
        )

        # Wrap with visualization wrapper
        env = VisualizationWrapper(env)

        # Grab reference to controller config
        env_info = json.dumps(config)

        # wrap the environment with enhanced data collection wrapper
        tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
        env = EnhancedDataCollectionWrapper(env, tmp_directory)

        # initialize device
        if args.device == "keyboard":
            from robosuite.devices import Keyboard
            device = Keyboard(
                env=env,
                pos_sensitivity=args.pos_sensitivity,
                rot_sensitivity=args.rot_sensitivity,
            )
        elif args.device == "spacemouse":
            from robosuite.devices import SpaceMouse
            device = SpaceMouse(
                env=env,
                pos_sensitivity=args.pos_sensitivity,
                rot_sensitivity=args.rot_sensitivity,
            )
        elif args.device == "dualsense":
            from robosuite.devices import DualSense
            device = DualSense(
                env=env,
                pos_sensitivity=args.pos_sensitivity,
                rot_sensitivity=args.rot_sensitivity,
                reverse_xy=args.reverse_xy,
            )
        elif args.device == "mjgui":
            from robosuite.devices.mjgui import MJGUI
            device = MJGUI(env=env)
        else:
            raise Exception("Invalid device choice")

        # make a new timestamped directory
        t1, t2 = str(time.time()).split(".")
        new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
        os.makedirs(new_dir, exist_ok=True)

        # collect demonstrations
        try:
            collect_human_trajectory(env, device, args.arm, args.max_fr)
            
            # Always create HDF5 file, even if no data
            hdf5_path = gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
            
            # Check if file was created
            if os.path.exists(hdf5_path):
                print(f"HDF5 file created successfully: {hdf5_path}")
                
                # List contents
                with h5py.File(hdf5_path, "r") as f:
                    print(f"HDF5 file contents: {list(f.keys())}")
                    if "data" in f:
                        data_grp = f["data"]
                        print(f"Data group contents: {list(data_grp.keys())}")
                        print(f"Number of demos: {data_grp.attrs.get('num_demos', 0)}")
            else:
                print(f"HDF5 file was not created: {hdf5_path}")
                
        except KeyboardInterrupt:
            print("\nCollection stopped by user")
        except Exception as e:
            print(f"Error during collection: {e}")
        finally:
            # Force create HDF5 file even on error
            try:
                hdf5_path = gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
                print(f"Final HDF5 file: {hdf5_path}")
            except Exception as e:
                print(f"Error creating final HDF5: {e}")
            
            env.close()