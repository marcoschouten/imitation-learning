# Imitation Learning for Skill Transfer in Human-Robot Teleoperation.

This research will develop a remote control system based on human hand input gestures to enable a faraway robot to replicate human motion with zero latency and high accuracy.
More specifically, to develop an imitation learning system to reproduce skills on robot manipulators with minimal inputs using the Learning by Demonstration (LbD) framework. Human-like input demonstrations are retrieved through a Kinect motion sensor. The experimental results confirm the effectiveness and efficiency of the proposed method when comparing the inferred trajectory to the ground truth.


# Contribution
1. Implement a Gaussian Mixture Model (GMM) to characterize the task trajectory and generate the underlying task trajectory.
2. Build an encoder-decoder system to encode the generated trajectory using minimal inputs.


Overall Systems Design:
[design.jpg](https://postimg.cc/2bbNTvwL)

Input:
[input.jpg](https://postimg.cc/8FnVhwhV)

Groud Truth:
[1.jpg](https://postimg.cc/ZC1BLFxp)

Gaussian Components:
[2.jpg](https://postimg.cc/v4hymGTT)

Inference:
[3.jpg](https://postimg.cc/5Xb11XxD)

Encoder:
[4.jpg](https://postimg.cc/SJWqw6pW)

Decoder:
[5.jpg](https://postimg.cc/qzf4bmWJ)

Task Dictionary:
[6.jpg](https://postimg.cc/V50zFyGZ)


