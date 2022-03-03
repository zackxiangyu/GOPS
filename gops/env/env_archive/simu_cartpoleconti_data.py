from gym import spaces
import gym
from gops.env.resources.simu_cartpole import cartpole
import numpy as np

class SimuCartpoleconti(gym.Env):

    def __init__(self, **kwargs):
        self._physics = cartpole.model_wrapper()
        self.is_adversary = kwargs.get('is_adversary', True)

        self.action_space = spaces.Box(np.array(self._physics.get_param()['a_min']).reshape(-1), np.array(self._physics.get_param()['a_max']).reshape(-1))
        self.observation_space = spaces.Box(np.array(self._physics.get_param()['x_min']).reshape(-1), np.array(self._physics.get_param()['x_max']).reshape(-1))
        self.adv_action_space = spaces.Box(np.array(self._physics.get_param()['adva_min']).reshape(-1), np.array(self._physics.get_param()['adva_max']).reshape(-1))
        self.adv_action_dim = self.adv_action_space.shape[0]
        self.reset()

    def step(self, action, adv_action=None):
        if self.is_adversary==False:
            if adv_action is not None:
                raise ValueError('Adversary training setting is wrong')
            else:
                adv_action = np.array([0.] * self.adv_action_dim)
        else:
            if adv_action is None:
                raise ValueError('Adversary training setting is wrong')
        state, isdone, reward = self._step_physics({'Action': action.astype(np.float64), 'AdverAction': adv_action.astype(np.float64)})
        self.cstep += 1
        info = {'TimeLimit.truncated': self.cstep > 200}
        return state, reward, isdone, info

    def reset(self):
        self._physics.terminate()
        self._physics = cartpole.model_wrapper()

        # randomized initiate
        state = np.random.uniform(low=[-0.05], high=[0.05], size=(4,))
        param = self._physics.get_param()
        param.update(list(zip(('x_ini'), state)))
        self._physics.set_param(param)
        self._physics.initialize()
        self.cstep = 0
        return state

    def render(self):
        pass

    def close(self):
        self._physics.renderterminate()

    def _step_physics(self, action):
        return self._physics.step(action)


if __name__ == "__main__":
    import gym
    import numpy as np

    env = SimuCartpoleconti()
    s = env.reset()
    for i in range(50):
        a = np.ones([1])*2
        sp, r, d, _ = env.step(a, 1)
        print(s, a, r, d)
        s = sp