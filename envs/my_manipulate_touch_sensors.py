import os
import numpy as np
import mujoco_py

from gym import utils, error, spaces

from gym.envs.robotics import rotations
from gym.envs.robotics.hand import manipulate
from gym.envs.robotics.utils import robot_get_obs
from gym.wrappers import TimeLimit

# path of the xml for environment
MANIPULATE_SHARE_XML = os.path.join(os.getcwd(), 'envs', 'shared_control_block_sensors.xml')

'''
achieved_goal the current pos and the force
target_goal   the target pos and the force constrain
obs           all joint poses and touch sensors
'''

FINGERTIP_SITE_NAMES = [
    'robot0:S_fftip',
    'robot0:S_mftip',
    'robot0:S_rftip',
    'robot0:S_lftip',
    'robot0:S_thtip',
]

SENSOR_SITE_NAME = [
    'robot0:ST_Tch_fftip',
    'robot0:ST_Tch_mftip',
    'robot0:ST_Tch_rftip',
    'robot0:ST_Tch_lftip',
    'robot0:ST_Tch_thtip',
]

DEFAULT_INITIAL_QPOS= {
    'robot0:WRJ1': -0.16514339750464327,
    'robot0:WRJ0': -0.31973286565062153,
    'robot0:FFJ3': 0.14340512546557435,
    'robot0:FFJ2': 0.32028208333591573,
    'robot0:FFJ1': 0.7126053607727917,
    'robot0:FFJ0': 0.6705281001412586,
    'robot0:MFJ3': 0.000246444303701037,
    'robot0:MFJ2': 0.3152655251085491,
    'robot0:MFJ1': 0.7659800313729842,
    'robot0:MFJ0': 0.7323156897425923,
    'robot0:RFJ3': 0.00038520700007378114,
    'robot0:RFJ2': 0.36743546201985233,
    'robot0:RFJ1': 0.7119514095008576,
    'robot0:RFJ0': 0.6699446327514138,
    'robot0:LFJ4': 0.0525442258033891,
    'robot0:LFJ3': -0.13615534724474673,
    'robot0:LFJ2': 0.39872030433433003,
    'robot0:LFJ1': 0.7415570009679252,
    'robot0:LFJ0': 0.704096378652974,
    'robot0:THJ4': 0.003673823825070126,
    'robot0:THJ3': 0.5506291436028695,
    'robot0:THJ2': -0.014515151997119306,
    'robot0:THJ1': -0.0015229223564485414,
    'robot0:THJ0': -0.7894883021600622,
}

def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat /= np.linalg.norm(quat)
    return quat

class ManipulateTouchSensorsEnv(manipulate.ManipulateEnv):
    def __init__(
        self, model_path, target_position, target_rotation,
        target_position_range, reward_type, initial_qpos=DEFAULT_INITIAL_QPOS,
        randomize_initial_position=True, randomize_initial_rotation=True,
        distance_threshold=0.05, rotation_threshold=0.1, n_substeps=20, relative_control=False,
        ignore_z_target_rotation=False, touch_visualisation="on_touch", touch_get_obs="sensordata",
    ):
        """Initializes a new Hand manipulation environment with touch sensors.

        Args:
            touch_visualisation (string): how touch sensor sites are visualised
                - "on_touch": shows touch sensor sites only when touch values > 0
                - "always": always shows touch sensor sites
                - "off" or else: does not show touch sensor sites
            touch_get_obs (string): touch sensor readings
                - "boolean": returns 1 if touch sensor reading != 0.0 else 0
                - "sensordata": returns original touch sensor readings from self.sim.data.sensordata[id]
                - "log": returns log(x+1) touch sensor readings from self.sim.data.sensordata[id]
                - "off" or else: does not add touch sensor readings to the observation

        """

        self.touch_visualisation = touch_visualisation
        self.touch_get_obs = touch_get_obs
        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []
        self.touch_color = [1, 0, 0, 0.5]
        self.notouch_color = [0, 0.5, 0, 0.2]

        manipulate.ManipulateEnv.__init__(
            self, model_path, target_position, target_rotation,
            target_position_range, reward_type, initial_qpos=initial_qpos,
            randomize_initial_position=randomize_initial_position, randomize_initial_rotation=randomize_initial_rotation,
            distance_threshold=distance_threshold, rotation_threshold=rotation_threshold, n_substeps=n_substeps, relative_control=relative_control,
            ignore_z_target_rotation=ignore_z_target_rotation,
        )

        for k, v in self.sim.model._sensor_name2id.items():  # get touch sensor site names and their ids
            if 'robot0:TS_' in k:
                self._touch_sensor_id_site_id.append((v, self.sim.model._site_name2id[k.replace('robot0:TS_', 'robot0:T_')]))
                self._touch_sensor_id.append(v)

        if self.touch_visualisation == 'off':  # set touch sensors rgba values
            for _, site_id in self._touch_sensor_id_site_id:
                self.sim.model.site_rgba[site_id][3] = 0.0
        elif self.touch_visualisation == 'always':
            pass

        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))



    # ManipulateEnv methods
    # ----------------------------
    def _get_achieved_goal(self):
        # Goal from HandReachEnv
        goal = [self.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES]
        fingertips_goal = np.array(goal).flatten()
        # Goal from ManipulateEnv
        force_goal = []
        for sensor_name in SENSOR_SITE_NAME:
            sensor_force = self.sim.data.sensordata[self.sim.model._sensor_name2id[sensor_name]]
            force_goal.append([sensor_force])
        force_goal = np.array(force_goal).flatten()
        fingertips_goal = np.concatenate([fingertips_goal,force_goal])

        return fingertips_goal
        # return object_qpos

    # GoalEnv methods
    # ----------------------------
    def compute_reward(self, achieved_goal, goal, info):
        if self.reward_type == 'sparse':
            finger_success = self._is_success(achieved_goal, goal)
            return (finger_success - 1.)
        else:
            d = self.goal_distance(achieved_goal, goal)
            force = np.array([0.0])
            if force.any() > 10:
                d_force = -100
            elif force.any() > 1:
                d_force = 1
            else:
                d_force = 0
            return -d

    # RobotEnv methods
    # ----------------------------

    def _env_setup(self, initial_qpos): # From ManipulateEnv
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

        self.initial_goal = self._get_achieved_goal().copy()
        self.palm_xpos = self.sim.data.body_xpos[self.sim.model.body_name2id('robot0:palm')].copy()


    def _sample_goal(self):

        finger_names = [name for name in FINGERTIP_SITE_NAMES]

        meeting_pos = self.palm_xpos + np.array([0.0, -0.09, 0.05])

        finger_goal = self.initial_goal[:-5].copy().reshape(-1, 3)
        for finger_name in finger_names:
            finger_idx = FINGERTIP_SITE_NAMES.index(finger_name)
            offset_direction = (meeting_pos - finger_goal[finger_idx])
            offset_direction /= np.linalg.norm(offset_direction)
            finger_goal[finger_idx] =  meeting_pos - self.np_random.uniform(0.02, 0.1) * offset_direction # TODO:设定合适的距离


        if self.np_random.uniform() < 0.1:
            finger_goal = self.initial_goal[:-5].copy()

        force_goal = np.array([10]*5).astype(np.float32).copy()
        finger_goal = np.concatenate([finger_goal.flatten(),force_goal])

        return finger_goal.flatten()


    # From HandReachEnv
    def _is_success(self, achieved_goal, desired_goal):
        force_threshold = 0.1
        if len(achieved_goal.shape) > 1:
            achieved_goal_pos = achieved_goal[:,:-5]
            desired_goal_pos = desired_goal[:,:-5]
            achieved_goal_force = achieved_goal[:,-5:]
            desired_goal_force = desired_goal[:,-5:]
            m,n = achieved_goal_pos.shape
            mask = achieved_goal_force > force_threshold
            for i in range(m):
                for j in range(n):
                    if mask[i][j//3]:
                        achieved_goal_pos[i][j] = desired_goal_pos[i][j]

        else:
            achieved_goal_pos = achieved_goal[:-5]
            desired_goal_pos = desired_goal[:-5]
            achieved_goal_force = achieved_goal[-5:]
            desired_goal_force = desired_goal[-5:]
            mask = achieved_goal_force > force_threshold
            m = 1
            n = achieved_goal_pos.shape
            for j in range(n[0]):
                if mask[j//3]:
                    achieved_goal_pos[j] = desired_goal_pos[j]

        d = self.goal_distance(achieved_goal_pos,desired_goal_pos) < self.distance_threshold
        safty = np.logical_and.reduce(achieved_goal_force < desired_goal_force, axis=-1)

        return np.logical_and.reduce([d, safty]).astype(np.float32)

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)


    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        initial_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)
        initial_qpos = None

        # Randomization initial rotation.
        if self.randomize_initial_rotation:
            if self.target_rotation == 'z':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0., 0., 1.])
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == 'parallel':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0., 0., 1.])
                z_quat = quat_from_angle_and_axis(angle, axis)
                parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
                offset_quat = rotations.quat_mul(z_quat, parallel_quat)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation in ['xyz', 'ignore']:
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = self.np_random.uniform(-1., 1., size=3)
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == 'fixed':
                pass
            else:
                raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != 'fixed':
                if self.np_random.uniform() < 0.3:
                    initial_pos = np.ones_like(initial_pos) * 10.0
                else:
                    initial_pos += self.np_random.normal(size=3, scale=0.005)

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])
        self.sim.data.set_joint_qpos('object:joint', initial_qpos)

        def is_on_palm():
            self.sim.forward()
            cube_middle_idx = self.sim.model.site_name2id('object:center')
            cube_middle_pos = self.sim.data.site_xpos[cube_middle_idx]
            is_on_palm = (cube_middle_pos[2] > 0.04)
            return is_on_palm

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(20))
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False
        return is_on_palm()

    def _render_callback(self): # From ManipulateEnv
        if self.touch_visualisation == 'on_touch':
            for touch_sensor_id, site_id in self._touch_sensor_id_site_id:
                if self.sim.data.sensordata[touch_sensor_id] != 0.0:
                    self.sim.model.site_rgba[site_id] = self.touch_color
                else:
                    self.sim.model.site_rgba[site_id] = self.notouch_color
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        goal = self.goal[:-5].reshape(5, 3)
        for finger_idx in range(5):
            site_name = 'target{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = goal[finger_idx] - sites_offset[site_id]
        achieved_goal = self._get_achieved_goal()[:-5].reshape(5, 3)
        for finger_idx in range(5):
            site_name = 'finger{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = achieved_goal[finger_idx] - sites_offset[site_id]
        self.sim.forward()


    def _get_obs_obj_nosensor(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        object_qvel = self.sim.data.get_joint_qvel('object:joint')
        achieved_goal = self._get_achieved_goal().ravel()  # this contains the object position + rotation
        observation = np.concatenate([robot_qpos, robot_qvel, object_qvel, achieved_goal])
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.ravel().copy(),
        }

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        achieved_goal = self._get_achieved_goal().ravel()
        touch_values = []  # get touch sensor readings. if there is one, set value to 1
        if self.touch_get_obs == 'sensordata':
            touch_values = self.sim.data.sensordata[self._touch_sensor_id]
        elif self.touch_get_obs == 'boolean':
            touch_values = self.sim.data.sensordata[self._touch_sensor_id] > 0.0
        elif self.touch_get_obs == 'log':
            touch_values = np.log(self.sim.data.sensordata[self._touch_sensor_id] + 1.0)
        observation = np.concatenate([robot_qpos, robot_qvel, touch_values, achieved_goal])
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _get_obs_obj_sensor(self):
        robot_qpos, robot_qvel = manipulate.robot_get_obs(self.sim)
        object_qvel = self.sim.data.get_joint_qvel('object:joint')
        achieved_goal = self._get_achieved_goal().ravel()  # this contains the object position + rotation
        touch_values = []  # get touch sensor readings. if there is one, set value to 1
        if self.touch_get_obs == 'sensordata':
            touch_values = self.sim.data.sensordata[self._touch_sensor_id]
        elif self.touch_get_obs == 'boolean':
            touch_values = self.sim.data.sensordata[self._touch_sensor_id] > 0.0
        elif self.touch_get_obs == 'log':
            touch_values = np.log(self.sim.data.sensordata[self._touch_sensor_id] + 1.0)
        observation = np.concatenate([robot_qpos, robot_qvel, object_qvel, touch_values, achieved_goal])

        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.ravel().copy(),
        }


class SharedBlockTouchSensorsEnv(ManipulateTouchSensorsEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='ignore', touch_get_obs='sensordata', reward_type='dense'):
        utils.EzPickle.__init__(self, target_position, target_rotation, touch_get_obs, reward_type)
        ManipulateTouchSensorsEnv.__init__(self,
            model_path=MANIPULATE_SHARE_XML,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type)

class SharedBlockTouchSensorsEnvSparse(ManipulateTouchSensorsEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='ignore', touch_get_obs='sensordata', reward_type='sparse'):
        utils.EzPickle.__init__(self, target_position, target_rotation, touch_get_obs, reward_type)
        ManipulateTouchSensorsEnv.__init__(self,
            model_path=MANIPULATE_SHARE_XML,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type)