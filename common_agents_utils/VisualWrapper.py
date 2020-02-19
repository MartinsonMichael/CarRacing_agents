import cv2
import numpy as np
from threading import Event, Thread
import gym
import time
from xvfbwrapper import Xvfb

from envs.common_envs_utils.visualizer import save_as_mp4


class VisualWrapper(gym.Wrapper):

    def __init__(self, env):
        self.width = 400
        self.height = 400

        gym.Wrapper.__init__(self, env)
        self.env = env.unwrapped

        """
        start new thread to deal with getting raw image
        """
        self.renderer = RenderThread(env)
        self.renderer.start()
        self.ims = []

    def _pre_process(self, frame):
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # frame = np.expand_dims(frame, -1)
        self.ims.append(frame)

    def on_done(self):
        save_as_mp4(np.array(self.ims), f'animations/_{time.time()}.mp4')

    def step(self, ac):
        state, reward, done, info = self.env.step(ac)
        self.renderer.begin_render()  # move screen one step

        self._pre_process(self.renderer.get_screen())
        if done:
            self.on_done()

        return state, reward, done, info

    def reset(self, **kwargs):
        try:
            self.renderer.stop()
            self.renderer.join()
        except:
            pass
        state = self.env.reset()
        self.renderer.begin_render()
        self._pre_process(self.renderer.get_screen())
        return state

    def close(self):
        self.renderer.stop()  # terminate the threads
        self.renderer.join()  # collect the dead threads and notice all threads are safely terminated
        if self.env:
            return self.env.close()


class RenderThread(Thread):
    """
    Original Code:
        https://github.com/tqjxlm/Simple-DQN-Pytorch/blob/master/Pytorch-DQN-CartPole-Raw-Pixels.ipynb

    Data:
        - Observation: 3 x 400 x 600

    Usage:
        1. call env.step() or env.reset() to update env state
        2. call begin_render() to schedule a rendering task (non-blocking)
        3. call get_screen() to get the lastest scheduled result (block main thread if rendering not done)

    Sample Code:

    ```python
        # A simple test
        env = gym.make('CartPole-v0').unwrapped
        renderer = RenderThread(env)
        renderer.start()
        env.reset()
        renderer.begin_render()
        for i in range(100):
            screen = renderer.get_screen() # Render the screen
            env.step(env.action_space.sample()) # Select and perform an action
            renderer.begin_render()
            print(screen)
            print(screen.shape)
        renderer.stop()
        renderer.join()
        env.close()
    ```
    """

    def __init__(self, env):
        super(RenderThread, self).__init__(target=self.render)
        self._stop_event = Event()
        self._state_event = Event()
        self._render_event = Event()
        self.env = env

        self.vdisplay = Xvfb()

    def stop(self):
        """
        Stops the threads

        :return:
        """
        self.vdisplay.stop()
        self._stop_event.set()
        self._state_event.set()

    def stopped(self):
        """
        Check if the thread has been stopped

        :return:
        """
        return self._stop_event.is_set()

    def begin_render(self):
        """
        Start rendering the screen

        :return:
        """
        self.vdisplay.start()
        self._state_event.set()

    def get_screen(self):
        """
        get and output the screen image

        :return:
        """
        self._render_event.wait()
        self._render_event.clear()
        return self.screen

    def render(self):
        while not self.stopped():
            self._state_event.wait()
            self._state_event.clear()
            self.screen = self.env.render(mode='rgb_array')
            self._render_event.set()
