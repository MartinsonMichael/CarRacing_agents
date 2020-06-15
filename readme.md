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

| | |
|---|---|
| TD3 on track full rotate | ![](media/TD3_fisrt_sucecc_rotate_R__15.0_Time__289_.mp4.gif) | 
| Rainbow on track full rotate | ![](media/Rainbow_image_vector_full_rotate_R_27.0_Time_456_1586555257.733288.mp4.gif) |
| PPO on track small rotation with bots | ![](media/PPO_with_bots_R_5.5_Time_298_1588081821.2236094.mp4.gif) |

 
 



