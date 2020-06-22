import os
import time

import numpy as np
from matplotlib import animation
from IPython.display import display, HTML
import datetime
import matplotlib.pyplot as plt


def episode_visualizer(env, action_picker, name='test', folder='save_animation_folder', image_processor=None):
    folder_full_path = os.path.join(folder, name)
    if not os.path.exists(folder_full_path):
        os.makedirs(folder_full_path)

    if image_processor is None:
        image_processor = lambda x: np.array(x)
    elif image_processor == 'swap_channels':
        image_processor = lambda x: np.transpose(np.array(x), (2, 1, 0))

    state = env.reset(visualize_next_episode=True)
    im_array = [np.array(env.get_true_picture()).astype(np.uint8)]
    total_reward = 0.0
    step_num = 0
    while True:
        action = action_picker(state)
        new_state, reward, done, info = env.step(action)
        im_array.append(np.array(env.get_true_picture()).astype(np.uint8))
        state = new_state
        total_reward += reward
        step_num += 1
        if done or step_num > 300:
            break
    plot_sequence_images(im_array, need_disaply=False, need_save=os.path.join(
        folder_full_path, f'R_{total_reward}__Step_{step_num}__Time_{datetime.datetime.now()}_.mp4'
    ))


# f'./save_animation_folder/{datetime.datetime.now()}.mp4'

def save_as_mp4(image_array, save_path, logger, save_to_wandb: bool) -> None:
    dpi = 72.0
    xpixels, ypixels = image_array[0].shape[:2]
    fig = plt.figure(figsize=(ypixels / dpi, xpixels / dpi), dpi=dpi)
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(image_array),
        interval=33,
        repeat_delay=1,
        repeat=True
    )
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    anim.save(save_path)

    if save_to_wandb:
        assert logger is not None
        time.sleep(10)
        logger.log_video(save_path)


def plot_sequence_images(image_array, need_disaply=False, need_save=None):
    ''' Display images sequence as an animation in jupyter notebook

    Args:
        image_array(numpy.ndarray): image_array.shape equal to (num_images, height, width, num_channels)
    '''
    dpi = 72.0
    xpixels, ypixels = image_array[0].shape[:2]
    fig = plt.figure(figsize=(ypixels / dpi, xpixels / dpi), dpi=dpi)
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(image_array),
        interval=33,
        repeat_delay=1,
        repeat=True
    )
    if need_save is not None:
        anim.save(need_save)

    if need_disaply:
        display(HTML(anim.to_html5_video()))
