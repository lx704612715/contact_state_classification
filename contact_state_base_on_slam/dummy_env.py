import numpy as np
import pybullet as p
import os, sys
script_path = os.path.realpath(os.path.dirname(__name__))
os.chdir(script_path)
sys.path.append("./SimulationFrameworkPublic")
from SimulationFrameworkPublic.classic_framework.pybullet.PyBulletRobot import PyBulletRobot
from SimulationFrameworkPublic.classic_framework.pybullet.PyBulletScene import PyBulletScene as Scene
from SimulationFrameworkPublic.classic_framework.interface.Logger import RobotPlotFlags
from SimulationFrameworkPublic.classic_framework.pybullet.pb_utils.pybullet_scene_object import PyBulletObject

if __name__ == '__main__':

    duck = PyBulletObject(urdf_name='cuboid',
                          object_name='cubic',
                          position=[6.7e-01, -0.1, 0.91],
                          orientation=[np.pi / 2, 0, 0],
                          data_dir=None)
    # load duck

    object_list = [duck]
    scene = Scene(object_list=object_list)

    PyBulletRobot = PyBulletRobot(p, scene, gravity_comp=True)
    PyBulletRobot.use_inv_dyn = False

    init_pos = PyBulletRobot.current_c_pos
    init_or = PyBulletRobot.current_c_quat
    init_joint_pos = PyBulletRobot.current_j_pos

    PyBulletRobot.startLogging()
    # duration = 4
    duration = 2

    PyBulletRobot.ctrl_duration = duration
    PyBulletRobot.set_gripper_width = 0.04

    # move to the position 10cm above the object
    desired_cart_pos_1 = np.array([0.64669174, -0.1, 0.05])
    # desired_quat_1 = [0.01806359,  0.91860348, -0.38889658, -0.06782891]
    desired_quat_1 = [0, 1, 0, 0]  # we use w,x,y,z. where pybullet uses x,y,z,w (we just have to swap the positions)

    PyBulletRobot.gotoCartPositionAndQuat(desiredPos=desired_cart_pos_1, desiredQuat=desired_quat_1, duration=duration)
    # there is no gripper controller. The desired gripper width will be executed right after the next controller
    # starts
    PyBulletRobot.set_gripper_width = 0.0

    # close the gripper and lift up the object
    desired_cart_pos_2 = np.array([6.5e-01, -0.1, 1.51])
    PyBulletRobot.gotoCartPositionAndQuat(desiredPos=desired_cart_pos_2, desiredQuat=desired_quat_1, duration=duration)

    # get camera image
    robot_id = PyBulletRobot.robot_id
    client_id = scene.physics_client_id
    scene.inhand_cam.get_image(cam_id=robot_id, client_id=client_id)
    # scene.cage_cam.get_image(cam_id=robot_id, client_id=client_id)
    print("show camera image")

    PyBulletRobot.stopLogging()
    PyBulletRobot.logger.plot(RobotPlotFlags.END_EFFECTOR | RobotPlotFlags.JOINTS)
