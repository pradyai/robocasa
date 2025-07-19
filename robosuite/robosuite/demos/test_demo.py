"""
Teleoperate robot with keyboard or other devices while collecting trajectory data.

This script combines device control with data collection, allowing you to:
1. Control the robot with keyboard, spacemouse, or other devices
2. Record trajectory data including joint angles, actions, and states
3. Play back recorded trajectories

Example usage:
    # Record data with keyboard control
    $ python demo_device_control_with_data_collection.py --environment Lift --device keyboard --directory /tmp/trajectories

    # Play back recorded data
    $ python demo_device_control_with_data_collection.py --environment Lift --playback --directory /tmp/trajectories
"""

import argparse
import os
import time
from glob import glob
from copy import deepcopy

import numpy as np

import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper, DataCollectionWrapper


def collect_teleoperation_trajectory(env, device, args):
    """
    Collect trajectory data using teleoperation device control.
    
    Args:
        env: The robosuite environment wrapped with DataCollectionWrapper
        device: The input device (keyboard, spacemouse, etc.)
        args: Command line arguments
    """
    print("Starting teleoperation data collection...")
    print("Use your device to control the robot. Press 'r' to reset environment.")
    print("Close the window or press Ctrl+C to stop recording.")
    
    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    
    episode_count = 0
    
    while True:
        # Reset the environment
        obs = env.reset()
        episode_count += 1
        print(f"\n--- Episode {episode_count} ---")
        print(f"Data will be saved to: {env.ep_directory}")
        
        # Setup rendering
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)
        env.render()
        
        # Initialize variables that should be maintained between resets
        last_grasp = 0
        step_count = 0
        
        # Initialize device control
        device.start_control()
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]
        
        # Loop until we get a reset from the input or the task completes
        while True:
            start = time.time()
            
            # Set active robot
            active_robot = env.robots[device.active_robot]
            
            # Get the newest action
            input_ac_dict = device.input2action()
            
            # If action is none, then this is a reset so we should break
            if input_ac_dict is None:
                break
            
            action_dict = deepcopy(input_ac_dict)
            
            # Set arm actions
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
            
            # Step the environment (this automatically records data due to DataCollectionWrapper)
            obs, reward, done, info = env.step(env_action)
            env.render()
            
            step_count += 1
            if step_count % 100 == 0:
                print(f"Step {step_count} - Joint angles: {obs['robot0_joint_pos']}")
            
            # Limit frame rate if necessary
            if args.max_fr is not None:
                elapsed = time.time() - start
                diff = 1 / args.max_fr - elapsed
                if diff > 0:
                    time.sleep(diff)
        
        print(f"Episode {episode_count} completed with {step_count} steps")


def playback_trajectory(env, ep_dir, args):
    """
    Playback data from a recorded episode.
    
    Args:
        env: The robosuite environment
        ep_dir: The path to the directory containing data for an episode
        args: Command line arguments
    """
    print(f"Playing back trajectory from: {ep_dir}")
    
    # First reload the model from the xml
    xml_path = os.path.join(ep_dir, "model.xml")
    with open(xml_path, "r") as f:
        env.reset_from_xml_string(f.read())
    
    # Load and display joint angle data
    joint_data_path = os.path.join(ep_dir, "joint_data.npz")
    if os.path.exists(joint_data_path):
        joint_data = np.load(joint_data_path)
        print(f"Joint data shape: {joint_data['joint_positions'].shape}")
        print(f"Action data shape: {joint_data['actions'].shape}")
    
    state_paths = os.path.join(ep_dir, "state_*.npz")
    
    # Read states back, load them one by one, and render
    t = 0
    for state_file in sorted(glob(state_paths)):
        print(f"Loading state file: {state_file}")
        dic = np.load(state_file)
        states = dic["states"]
        
        for i, state in enumerate(states):
            start = time.time()
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env.viewer.update()
            env.render()
            
            # Print joint angles every 50 steps
            if t % 50 == 0:
                joint_pos = env.sim.data.qpos[:env.robots[0].dof]
                print(f"Step {t} - Joint angles: {joint_pos}")
            
            t += 1
            
            if args.max_fr is not None:
                elapsed = time.time() - start
                diff = 1 / args.max_fr - elapsed
                if diff > 0:
                    time.sleep(diff)
    
    print(f"Playback completed. Total steps: {t}")
    env.close()


def setup_device(args, env):
    """Setup the input device based on command line arguments."""
    if args.device == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
        env.viewer.add_keypress_callback(device.on_press)
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
        raise Exception("Invalid device choice: choose either 'keyboard', 'dualsense', 'spacemouse', or 'mjgui'.")
    
    return device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--switch-on-grasp",
        action="store_true",
        help="Switch gripper control on gripper action",
    )
    parser.add_argument(
        "--toggle-camera-on-grasp",
        action="store_true",
        help="Switch camera angle on gripper action",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic or json file path",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simulation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense Only) Reverse the effect of the x and y axes of the joystick",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="/tmp/robosuite_trajectories",
        help="Directory to save/load trajectory data",
    )
    parser.add_argument(
        "--playback",
        action="store_true",
        help="Playback recorded trajectories instead of recording new ones",
    )
    parser.add_argument(
        "--episode_dir",
        type=str,
        default=None,
        help="Specific episode directory to playback (if not specified, will use latest)",
    )
    
    args = parser.parse_args()
    
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
    else:
        args.config = None
    
    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )
    
    if args.playback:
        # Playback mode
        env = VisualizationWrapper(env, indicator_configs=None)
        
        # Determine which episode to playback
        if args.episode_dir:
            ep_dir = args.episode_dir
        else:
            # Find the latest episode in the directory
            episode_dirs = glob(os.path.join(args.directory, "ep_*"))
            if not episode_dirs:
                print(f"No episodes found in {args.directory}")
                exit(1)
            ep_dir = max(episode_dirs, key=os.path.getmtime)
        
        print(f"Playing back episode from: {ep_dir}")
        playback_trajectory(env, ep_dir, args)
    
    else:
        # Recording mode
        # Wrap environment with data collection wrapper
        env = DataCollectionWrapper(env, args.directory)
        
        # Wrap with visualization wrapper
        env = VisualizationWrapper(env, indicator_configs=None)
        
        # Setup device
        device = setup_device(args, env)
        
        # Start teleoperation with data collection
        try:
            collect_teleoperation_trajectory(env, device, args)
        except KeyboardInterrupt:
            print("\nData collection stopped by user.")
        finally:
            env.close()