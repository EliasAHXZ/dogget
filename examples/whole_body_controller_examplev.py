"""Example of whole body controller on A1 robot."""
# from asyncio.windows_events import NULL
import select
from cgitb import reset
import os
import inspect
from re import T
from select import select
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import socket
from absl import app
from absl import flags
from absl import logging
from datetime import datetime
import numpy as np
import os
import scipy.interpolate
import time

import pybullet_data
from pybullet_utils import bullet_client
import pybullet  # pytype:disable=import-error

from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
#from mpc_controller import torque_stance_leg_controller
#import mpc_osqp
from mpc_controller import torque_stance_leg_controller_quadprog as torque_stance_leg_controller




from motion_imitation.robots import a1
from motion_imitation.robots import robot_config
from motion_imitation.robots.gamepad import gamepad_reader

flags.DEFINE_string("logdir", None, "where to log trajectories.")
flags.DEFINE_bool("use_gamepad", False,
                  "whether to use gamepad to provide control input.")
flags.DEFINE_bool("use_real_robot", False,
                  "whether to use real robot or simulation")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_float("max_time_secs", 300., "maximum time to run the robot.")
FLAGS = flags.FLAGS




_NUM_SIMULATION_ITERATION_STEPS = 300
_MAX_TIME_SECONDS = 30.

_STANCE_DURATION_SECONDS = [
    0.3
] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).

# def Standing
# _DUTY_FACTOR = [1.] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0., 0., 0.]
 
# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#  )

# def Tripod
#  _DUTY_FACTOR = [.8] * 4
#  _INIT_PHASE_FULL_CYCLE = [0., 0.25, 0.5, 0.]

#  _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.SWING,
#   )

# def Trotting
# _DUTY_FACTOR = [0.6] * 4
# _INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.SWING,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.SWING,
# )


def _generate_example_linear_angular_speed(t):
  """Creates an example speed profile based on time for demo purpose."""
  vx = 0.2
  vy = 0.35
  wz = 0.03

  time_points = (0,5,10,15,20,25,30)
  speed_points = ((vx, 0, 0, 0), (vx, 0, 0, 0),(vx, 0, 0, 0),(vx, 0, 0, 0), (0, 0, 0, 0),(0, 0, 0, 0), (0, 0, 0, 0))

  speed = scipy.interpolate.interp1d(time_points,
                                     speed_points,
                                     kind="previous",
                                     fill_value="extrapolate",
                                     axis=0)(t)

  return speed[0:3], speed[3], False

def _setup_controller1(robot):
  """Demonstrates how to create a locomotion controller."""
  desired_speed = (0, 0)
  desired_twisting_speed = 0

  gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
      robot,
      stance_duration=_STANCE_DURATION_SECONDS,
      duty_factor= [0.6] * 4,
      initial_leg_phase= [0.9, 0, 0, 0.9],
      initial_leg_state= (
       gait_generator_lib.LegState.SWING,
       gait_generator_lib.LegState.STANCE,
       gait_generator_lib.LegState.STANCE,
       gait_generator_lib.LegState.SWING,
        )            )
#  b'$J\x08\x00\x08\x00\x08\x00\x08\x00\x08\x00\x08\x00\x01\x00\x00\x03\xe8\x00\x03\xe8\x00\x03\xe8\x00$J\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
#  b'$J\x08\x00\x08\x00\x08\x00\x08\x00\x08\x00\x08\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00&p\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
  window_size = 20 if not FLAGS.use_real_robot else 1
  state_estimator = com_velocity_estimator.COMVelocityEstimator(
      robot, window_size=window_size)
  sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_height=robot.MPC_BODY_HEIGHT,
      foot_clearance=0.01)

  st_controller = torque_stance_leg_controller.TorqueStanceLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_body_height=robot.MPC_BODY_HEIGHT
      #,qp_solver = mpc_osqp.QPOASES #or mpc_osqp.OSQP
      )

  controller = locomotion_controller.LocomotionController(
      robot=robot,
      gait_generator=gait_generator,
      state_estimator=state_estimator,
      swing_leg_controller=sw_controller,
      stance_leg_controller=st_controller,
      clock=robot.GetTimeSinceReset)
  return controller

def _setup_controller0(robot):
  """Demonstrates how to create a locomotion controller."""
  desired_speed = (0, 0)
  desired_twisting_speed = 0

  gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
      robot,
      stance_duration=_STANCE_DURATION_SECONDS,
      duty_factor= [1.] * 4,
      initial_leg_phase= [0, 0, 0, 0],
      initial_leg_state= (
       gait_generator_lib.LegState.STANCE,
       gait_generator_lib.LegState.STANCE,
       gait_generator_lib.LegState.STANCE,
       gait_generator_lib.LegState.STANCE,
        )            )
  window_size = 20 if not FLAGS.use_real_robot else 1
  state_estimator = com_velocity_estimator.COMVelocityEstimator(
      robot, window_size=window_size)
  sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_height=robot.MPC_BODY_HEIGHT,
      foot_clearance=0.01)

  st_controller = torque_stance_leg_controller.TorqueStanceLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_body_height=robot.MPC_BODY_HEIGHT
      #,qp_solver = mpc_osqp.QPOASES #or mpc_osqp.OSQP
      )

  controller = locomotion_controller.LocomotionController(
      robot=robot,
      gait_generator=gait_generator,
      state_estimator=state_estimator,
      swing_leg_controller=sw_controller,
      stance_leg_controller=st_controller,
      clock=robot.GetTimeSinceReset)
  return controller


# def _setup_controller(robot):
#   """Demonstrates how to create a locomotion controller."""
#   desired_speed = (0, 0)
#   desired_twisting_speed = 0

#   gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
#       robot,
#       stance_duration=_STANCE_DURATION_SECONDS,
#       duty_factor=_DUTY_FACTOR,
#       initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
#       initial_leg_state=_INIT_LEG_STATE)
#   window_size = 20 if not FLAGS.use_real_robot else 1
#   state_estimator = com_velocity_estimator.COMVelocityEstimator(
#       robot, window_size=window_size)
#   sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
#       robot,
#       gait_generator,
#       state_estimator,
#       desired_speed=desired_speed,
#       desired_twisting_speed=desired_twisting_speed,
#       desired_height=robot.MPC_BODY_HEIGHT,
#       foot_clearance=0.01)

#   st_controller = torque_stance_leg_controller.TorqueStanceLegController(
#       robot,
#       gait_generator,
#       state_estimator,
#       desired_speed=desired_speed,
#       desired_twisting_speed=desired_twisting_speed,
#       desired_body_height=robot.MPC_BODY_HEIGHT
#       #,qp_solver = mpc_osqp.QPOASES #or mpc_osqp.OSQP
#       )

#   controller = locomotion_controller.LocomotionController(
#       robot=robot,
#       gait_generator=gait_generator,
#       state_estimator=state_estimator,
#       swing_leg_controller=sw_controller,
#       stance_leg_controller=st_controller,
#       clock=robot.GetTimeSinceReset)
#   return controller


def _update_controller_params(controller, lin_speed, ang_speed):
  controller.swing_leg_controller.desired_speed = lin_speed
  controller.swing_leg_controller.desired_twisting_speed = ang_speed
  controller.stance_leg_controller.desired_speed = lin_speed
  controller.stance_leg_controller.desired_twisting_speed = ang_speed



def _speed0():
  vx = 0.2
  vy = 0.35
  wz = 0.03



  time_points = (0,5)
  speed_points = ((0, 0, 0, 0), (0, 0, 0, 0))

  speed = scipy.interpolate.interp1d(time_points,
                                     speed_points,
                                     kind="previous",
                                     fill_value="extrapolate",
                                     axis=0)(1)

  return speed[0:3], speed[3], False


def _speed1():

  vx = 0.2
  vy = 0.35
  wz = 0.03

  time_points = (0,5)
  speed_points = ((vx, 0, 0, 0), (vx, 0, 0, 0))

  speed = scipy.interpolate.interp1d(time_points,
                                     speed_points,
                                     kind="previous",
                                     fill_value="extrapolate",
                                     axis=0)(1)

  return speed[0:3], speed[3], False
 
def _speedmin1():

  vx = 0.2
  vy = 0.35
  wz = 0.03

  time_points = (0,5)
  speed_points = ((-vx, 0, 0, 0), (-vx, 0, 0, 0))

  speed = scipy.interpolate.interp1d(time_points,
                                     speed_points,
                                     kind="previous",
                                     fill_value="extrapolate",
                                     axis=0)(1)

  return speed[0:3], speed[3], False


def _speed2():


  vx = 0.2
  vy = 0.4
  wz = 0.03

  time_points = (0,5)
  speed_points = ((0, 0, 0, vy), (0, 0, 0, vy))

  speed = scipy.interpolate.interp1d(time_points,
                                     speed_points,
                                     kind="previous",
                                     fill_value="extrapolate",
                                     axis=0)(1)

  return speed[0:3], speed[3], False  

def _speedmin2():


  vx = 0.2
  vy = 0.4
  wz = 0.03

  time_points = (0,5)
  speed_points = ((0, 0, 0, -vy), (0, 0, 0, -vy))

  speed = scipy.interpolate.interp1d(time_points,
                                     speed_points,
                                     kind="previous",
                                     fill_value="extrapolate",
                                     axis=0)(1)

  return speed[0:3], speed[3], False

def _speed3():


  vx = 0.2
  vy = 0.4
  wz = 0.03

  time_points = (0,5)
  speed_points = ((0, vx, 0, 0), (0, vx, 0, 0))

  speed = scipy.interpolate.interp1d(time_points,
                                     speed_points,
                                     kind="previous",
                                     fill_value="extrapolate",
                                     axis=0)(1)

  return speed[0:3], speed[3], False   

def _speedmin3():


  vx = 0.2
  vy = 0.4
  wz = 0.03

  time_points = (0,5)
  speed_points = ((0, -vx, 0, 0), (0, -vx, 0, 0))

  speed = scipy.interpolate.interp1d(time_points,
                                     speed_points,
                                     kind="previous",
                                     fill_value="extrapolate",
                                     axis=0)(1)

  return speed[0:3], speed[3], False   

def switch_speed(recv_data):
  # if recv_data[15] == 0 :      

  #   speed = _speed0()
  if recv_data[24] == recv_data[25] == recv_data[26] ==0 :
    speed = _speed0()
  elif  recv_data[24] == 1 :
    speed = _speed1()
  elif  recv_data[24] == 2 :
    speed = _speedmin1()
  elif  recv_data[25] == 1 :
    speed = _speed3()
  elif  recv_data[25] == 2 :
    speed = _speedmin3()
  elif recv_data[26] == 1 :
    speed = _speed2()
  elif recv_data[26] == 2  :
    speed = _speedmin2() 
  return speed

def switch_ctrl(recv_data , robot):
  # if recv_data[15] == 0 :      

  #  ctrler =_setup_controller0(robot)
  if recv_data[24] == recv_data[25] == recv_data[26] == 0 :

   ctrler =_setup_controller0(robot)
  else :
   ctrler =_setup_controller1(robot)
  return ctrler

def get_bcc(inputArr ):
    bcc = None
    for i in range(len(inputArr)-1):
        if i:
            bcc ^= inputArr[i]
        else:
            bcc = inputArr[i] ^ 0
    return bcc
# def socket1(tcp_server_socket):
#     new_client_socket, client_addr = tcp_server_socket.accept()

#     recv_data = new_client_socket.recv(1024)
#     if new_client_socket!=NULL:
#      t=1
#     else : t=0
#     return t

def main(argv):
  """Runs the locomotion controller example."""
  del argv # unused

  tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  tcp_server_socket.bind(("", 7925))
  tcp_server_socket.listen(128)

  # new_client_socket, client_addr = tcp_server_socket.accept()
  # recv_data= [0 for x in range (0,1024)]
  # recv_data[15]=0
  # recv_data[25]=2
  # recv_data[26]=1


  # Construct simulator
  if FLAGS.show_gui and not FLAGS.use_real_robot:
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
  else:
    p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
  p.setPhysicsEngineParameter(numSolverIterations=30)
  p.setTimeStep(0.001)
  p.setGravity(0, 0, -9.8)
  p.setPhysicsEngineParameter(enableConeFriction=0)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  p.loadURDF("plane.urdf")

  # Construct robot class:
  if FLAGS.use_real_robot:
    from motion_imitation.robots import a1_robot
    robot = a1_robot.A1Robot(
        pybullet_client=p,
        motor_control_mode=robot_config.MotorControlMode.HYBRID,
        enable_action_interpolation=False,
        time_step=0.002,
        action_repeat=1)
  else:
    robot = a1.A1(p,
                  motor_control_mode=robot_config.MotorControlMode.HYBRID,
                  enable_action_interpolation=False,
                  reset_time=2,
                  time_step=0.002,
                  action_repeat=1)
  if FLAGS.use_gamepad:
    gamepad = gamepad_reader.Gamepad()
    command_function = gamepad.get_command
  else:
    command_function = _generate_example_linear_angular_speed

  if FLAGS.logdir:
    logdir = os.path.join(FLAGS.logdir,
                          datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(logdir)
  start_time = robot.GetTimeSinceReset()
  current_time = start_time
  com_vels, imu_rates, actions = [], [], []
  # recv_data = new_client_socket.recv(1024)
  # for i in range(10000):
  # # while len(recv_data)<10:
  #   #time.sleep(0.0008) #on some fast computer, works better with sleep on real A1?
  #   start_time_robot = current_time
  #   start_time_wall = time.time()

  #   if i/3000 == 0:
  #     controller = _setup_controller0(robot)
  #     print("0")
  #     controller.reset()
  #     lin_speed, ang_speed, e_stop = _speed0()
  #   if e_stop:
  #     logging.info("E-stop kicked, exiting...")
  #     break
  #   _update_controller_params(controller, lin_speed, ang_speed)
  #   controller.update()
  #   hybrid_action, _ = controller.get_action()
  #   com_vels.append(np.array(robot.GetBaseVelocity()).copy())
  #   imu_rates.append(np.array(robot.GetBaseRollPitchYawRate()).copy())
  #   actions.append(hybrid_action)
  #   robot.Step(hybrid_action)
  #   current_time = robot.GetTimeSinceReset()



  #   if not FLAGS.use_real_robot:
  #     expected_duration = current_time - start_time_robot
  #     actual_duration = time.time() - start_time_wall
  #     if actual_duration < expected_duration:
  #       time.sleep(expected_duration - actual_duration)
    #print("actual_duration=", actual_duration)


  while current_time - start_time < FLAGS.max_time_secs:
   # for j in range(120000):
    #time.sleep(0.0008) #on some fast computer, works better with sleep on real A1?
    # new_client_socket, client_addr = tcp_server_socket.accept()

    # recv_data = new_client_socket.recv(1024)

    start_time_robot = current_time
    start_time_wall = time.time()
    # for i in range(2000):
    #   controller = _setup_controller0(robot)   
    #   controller.reset()
    #   lin_speed, ang_speed, e_stop = _speed0()
    #   _update_controller_params(controller, lin_speed, ang_speed)
    #   controller.update()
    #   hybrid_action, _ = controller.get_action()
    #   com_vels.append(np.array(robot.GetBaseVelocity()).copy())
    #   imu_rates.append(np.array(robot.GetBaseRollPitchYawRate()).copy())
    #   actions.append(hybrid_action)
    #   robot.Step(hybrid_action)
    #   current_time = robot.GetTimeSinceReset()
     
    #   if not FLAGS.use_real_robot:
    #     expected_duration = current_time - start_time_robot
    #     actual_duration = time.time() - start_time_wall
    #     if actual_duration < expected_duration:
    #       time.sleep(expected_duration - actual_duration)
    
    new_client_socket, client_addr = tcp_server_socket.accept()

    recv_data = new_client_socket.recv(1024)

    ready1 = select.select([new_client_socket], [], [], 0)
    print("%d"%ready1)

    # Updates the controller behavior parameters.      
    # print("Waiting to be connected...")



    # controller = _setup_controller0(robot)   
    # controller.reset()
    # lin_speed, ang_speed, e_stop = _speed0()
    # if e_stop:
    #   logging.info("E-stop kicked, exiting...")
    #   break
    # _update_controller_params(controller, lin_speed, ang_speed)
    # controller.update()
    # hybrid_action, _ = controller.get_action()
    # com_vels.append(np.array(robot.GetBaseVelocity()).copy())
    # imu_rates.append(np.array(robot.GetBaseRollPitchYawRate()).copy())
    # actions.append(hybrid_action)
    # robot.Step(hybrid_action)
    # current_time = robot.GetTimeSinceReset()
    # controller = _setup_controller0(robot)   
    # controller.reset()
    # lin_speed, ang_speed, e_stop = _speed0()
    # if e_stop:
    #   logging.info("E-stop kicked, exiting...")
    #   break
    # _update_controller_params(controller, lin_speed, ang_speed)
    # controller.update()
    # hybrid_action, _ = controller.get_action()
    # com_vels.append(np.array(robot.GetBaseVelocity()).copy())
    # imu_rates.append(np.array(robot.GetBaseRollPitchYawRate()).copy())
    # actions.append(hybrid_action)
    # robot.Step(hybrid_action)
    # current_time = robot.GetTimeSinceReset()
    # recv_data = new_client_socket.recv(1024)
    # recv_data[0]=36 
    # recv_data[1]=74  
    # recv_data[15]=1
    # recv_data[25]=1
    # recv_data[26]=1
    # if recv_data[0]==36 and recv_data[1]==74 and get_bcc(recv_data)== recv_data[45]:
    # if recv_data[0]==36 and recv_data[1]==74 :
    # k=socket1(tcp_server_socket)
    if len(recv_data)>10:
      for i in range(6000):
        if i/1000 == 0:
          controller = switch_ctrl(recv_data , robot)

          controller.reset()
          print(" %s" % recv_data)
          print("%s %s %s"% (recv_data[24],recv_data[25],recv_data[26]))
          lin_speed, ang_speed, e_stop = switch_speed(recv_data)
          
        if i/4000 == 1:
          controller = _setup_controller0(robot)   
          controller.reset()
          lin_speed, ang_speed, e_stop = _speed0()
        if e_stop:
          logging.info("E-stop kicked, exiting...")
          break
        _update_controller_params(controller, lin_speed, ang_speed)
        controller.update()
        hybrid_action, _ = controller.get_action()
        com_vels.append(np.array(robot.GetBaseVelocity()).copy())
        imu_rates.append(np.array(robot.GetBaseRollPitchYawRate()).copy())
        actions.append(hybrid_action)
        robot.Step(hybrid_action)
        current_time = robot.GetTimeSinceReset()

        if not FLAGS.use_real_robot:
          expected_duration = current_time - start_time_robot
          actual_duration = time.time() - start_time_wall
          if actual_duration < expected_duration:
            time.sleep(expected_duration - actual_duration)
  if FLAGS.use_gamepad:
    gamepad.stop()
  if FLAGS.logdir:
    np.savez(os.path.join(logdir, 'action.npz'),
              action=actions,
              com_vels=com_vels,
              imu_rates=imu_rates)
    logging.info("logged to: {}".format(logdir))

        # print("%f" % lin_speed[0])
      # else: 
      #   controller = _setup_controller0(robot)   
      #   controller.reset()
      #   lin_speed, ang_speed, e_stop = _speed0()
      


    # if i/3000 == 0:
    #   controller = _setup_controller1(robot)

    #   controller.reset()
    #   lin_speed, ang_speed, e_stop = _speed1()
    # if i/3000 == 1:
    #   controller = _setup_controller1(robot)

    #   controller.reset()
    #   lin_speed, ang_speed, e_stop = _speed2()
    # if i/3000 == 2:
    #   controller = _setup_controller1(robot)

    #   controller.reset()
    #   lin_speed, ang_speed, e_stop = _speed1()
    # if i/3000 == 3:
    #   controller = _setup_controller0(robot)

    #   controller.reset()
    #   lin_speed, ang_speed, e_stop = _speed0()

    # if recv_data[15] == 0 :      
    #   controller = _setup_controller0(robot)

    #   controller.reset()
    #   lin_speed, ang_speed, e_stop = _speed0()
 
    # elif recv_data[25] == recv_data[26] == recv_data[27] ==0 :
    #   controller = _setup_controller0(robot)

    #   controller.reset()
    #   lin_speed, ang_speed, e_stop = _speed0()
    # elif  recv_data[25] == 1 :
    #   controller = _setup_controller1(robot)
 
    #   controller.reset()
    #   lin_speed, ang_speed, e_stop = _speed1()
    # elif  recv_data[25] == 2 :
    #   controller = _setup_controller1(robot)

    #   controller.reset()
    #   lin_speed, ang_speed, e_stop = _speedmin1()
    # elif recv_data[27] == 1 :
    #   controller = _setup_controller1(robot)

    #   controller.reset()
    #   lin_speed, ang_speed, e_stop = _speed2()
    # elif recv_data[27] == 2 :
    #   controller = _setup_controller1(robot)

    #   controller.reset() 
    #   lin_speed, ang_speed, e_stop = _speedmin2() 
    # i = i + 1

    # print("%f" % lin_speed[0])
    # if e_stop:
    #   logging.info("E-stop kicked, exiting...")
    #   break
    # _update_controller_params(controller, lin_speed, ang_speed)
    # controller.update()
    # hybrid_action, _ = controller.get_action()
    # com_vels.append(np.array(robot.GetBaseVelocity()).copy())
    # imu_rates.append(np.array(robot.GetBaseRollPitchYawRate()).copy())
    # actions.append(hybrid_action)
    # robot.Step(hybrid_action)
    # current_time = robot.GetTimeSinceReset()

    # if not FLAGS.use_real_robot:
    #   expected_duration = current_time - start_time_robot
    #   actual_duration = time.time() - start_time_wall
    #   if actual_duration < expected_duration:
    #     time.sleep(expected_duration - actual_duration)
    #print("actual_duration=", actual_duration)
    
  # if FLAGS.use_gamepad:
  #   gamepad.stop()

  # if FLAGS.logdir:
  #   np.savez(os.path.join(logdir, 'action.npz'),
  #            action=actions,
  #            com_vels=com_vels,
  #            imu_rates=imu_rates)
  #   logging.info("logged to: {}".format(logdir))


if __name__ == "__main__":
  app.run(main)
