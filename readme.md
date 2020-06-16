## Navigating Autonomous Vehicle at the Road Intersection with Reinforcement Learning

Michael Martinson, Alexey Skrynnik, Aleksandr I. Panov

## Abstract
In this paper, we consider the problem of controlling an intelligent agent that simulates the behavior of an unmanned car when passing a road intersection together with other vehicles.
We consider the case of using smart city systems, which allow the agent to get full information about what is happening at the intersection in the form of video frames from surveillance cameras.
The paper proposes the implementation of a control system based on a trainable behavior generation module.
The agent's model is implemented using reinforcement learning (RL) methods.
In our work, we analyze various RL methods (PPO, Rainbow, TD3), and variants of the computer vision subsystem of the agent.
Also, we present our results of the best implementation of the agent when driving together with other participants in compliance with traffic rules.


## This work
In this project, we consider the task of learning an agent that simulates a self-driving car that performs the task of passing through the road intersection.
As a basic statement of the problem, we consider a realistic scenario of using data from the agent's sensors (images from cameras within the field of view, laser rangefinders, etc.), data coming from video surveillance cameras located in complex and loaded transport areas, in particular at road intersections.


## Env CarIntersect

Our environment - CarIntersect simulate four way crossroad with some physics and bot cars.
Technical description in env folder [link](env/).


## Methods examples

| | | |
|:---:|:---:|:---:|
| ![](media/TD3_fisrt_sucecc_rotate_R__15.0_Time__289_.mp4.gif) | ![](media/Rainbow_image_vector_full_rotate_R_27.0_Time_456_1586555257.733288.mp4.gif) | ![](media/PPO_with_bots_R_5.5_Time_298_1588081821.2236094.mp4.gif) |
| TD3 on track Full rotate | Rainbow on track Full rotate | PPO on track Small rotation with bots |
| | | |
| ![](media/TD3_line_vectors_semiFAIL_R_7.5_Time_287_1588119921.2851553.mp4.gif) | ![](media/Rainbow_image_line_semiFAIL_R_4.0_Time_312_1588043862.5293024.mp4.gif) | ![](media/PPO_Image_FAIL_R_0.0_Time_240_1587995861.7774887.mp4.gif) |
| TD3 on track Line | Rainbow with image as state on track Line  | PPO with image state on track Line |



## Convergence 

### State as a vector

| Small rotation | Medium rotation | Line | Full rotation |
|:---:|:---:|:---:|:---:|
| <img src="media/vector_small_rotation_track.svg" width="200" height="100"/> | <img src="media/vector_medium_rotation_track.svg" width="200" height="100"/> | <img src="media/vector_line_track.svg" width="200" height="100"/> | <img src="media/vector_full_rotation_track.svg" width="200" height="100"/> |
 


### State as an image

| Small rotation | Medium rotation | Line | Full rotation |
|:---:|:---:|:---:|:---:|
| <img src="media/image_small_rotation.svg" width="200" height="100"/> | <img src="media/image_med_rotation.svg" width="200" height="100"/> | <img src="media/image_line.svg" width="200" height="100"/> | <img src="media/image_full_rotation.svg" width="200" height="100"/> | 


### With bot cars

| Vector state | Image state |
|:---:|:---:|
| <img src="media/bots_vector.svg" width="200" height="100"/> | <img src="media/bots_image.svg" width="200" height="100"/> | 
