## Repo of experiments with CarIntersect env

This repo contains the implementation of RL agents: SAC, TD3, PPO, and wrapper for chainerRL.Rainbow.

The environment itself is in env folder. Open it for a more detailed description of the environment.


## Intro
Control architectures for mobile robotic systems and self-driving vehicles currently allow us to solve basic tasks for planning and self-driving in complex urban environments.
Often the methods used are based on pre-defined scenarios and rules of behavior, which significantly reduces the degree of autonomy of such systems.
One of the promising areas for increasing the degree of autonomy is the use of machine learning methods for automatically generating generalized object recognition procedures, including dynamic ones, in the external environment, and generating actions to achieve certain goals.


## This work
In this project, we consider the task of learning an intelligent agent that simulates a self-driving car that performs the task of passing through the road intersection.
As a basic statement of the problem, we consider a realistic scenario of using data from the agent's sensors (images from cameras within the field of view, laser rangefinders, etc.), data coming from video surveillance cameras located in complex and loaded transport areas, in particular at road intersections.


## Methods examples

| | | |
|:---:|:---:|:---:|
| ![](media/TD3_fisrt_sucecc_rotate_R__15.0_Time__289_.mp4.gif) | ![](media/Rainbow_image_vector_full_rotate_R_27.0_Time_456_1586555257.733288.mp4.gif) | ![](media/PPO_with_bots_R_5.5_Time_298_1588081821.2236094.mp4.gif) |
| TD3 on track Full rotate | Rainbow on track Full rotate | PPO on track Small rotation with bots |
| | | |
| ![](media/TD3_line_vectors_semiFAIL_R_7.5_Time_287_1588119921.2851553.mp4.gif) | ![](media/Rainbow_image_line_semiFAIL_R_4.0_Time_312_1588043862.5293024.mp4.gif) | ![](media/PPO_Image_FAIL_R_0.0_Time_240_1587995861.7774887.mp4.gif) |
| TD3 on track Line | Rainbow with image as state on track Line  | PPO with image state on track Line |


## Charts

We made three series of experiments. 

### State as a vector

| Small rotation | Medium rotation | Line | Full rotation |
|:---:|:---:|:---:|:---:|
| ![](media/vector_small_rotation_track.svg) | ![](media/vector_medium_rotation_track.svg) | ![](media/vector_line_track.svg) | ![](media/vector_full_rotation_track.svg) | 




