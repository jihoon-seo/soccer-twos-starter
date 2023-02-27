import soccer_twos
import onnxruntime as rt
import numpy as np

env = soccer_twos.make(render=True)
print("Observation Space: ", env.observation_space.shape) # (336,)
print("Action Space: ", env.action_space.shape) # (3,)

team0_reward = 0
team1_reward = 0
obs = env.reset()
print("obs[0].ndim: {}".format(obs[0].ndim)) # for debug
print("obs[0].shape: {}".format(obs[0].shape)) # for debug
print("obs[1].ndim: {}".format(obs[1].ndim)) # for debug
print("obs[1].shape: {}".format(obs[1].shape)) # for debug
print("obs[2].ndim: {}".format(obs[2].ndim)) # for debug
print("obs[2].shape: {}".format(obs[2].shape)) # for debug
print("obs[3].ndim: {}".format(obs[3].ndim)) # for debug
print("obs[3].shape: {}".format(obs[3].shape)) # for debug

sess = rt.InferenceSession("./trained_models/SoccerTwos.onnx")

"""
[SoccerTwos.onnx]

[Inputs]
- vector_observation: 336
- action_masks: 9

[Outputs]
- version_number: 1
- memory_size: 1
- discrete_actions: 9
- discrete_action_output_shape: 1
- action: 9
- is_continuous_control: 1
- action_output_shape: 1
"""

while True:
    
    # Team 1
    X_test1 = np.vstack((obs[0], obs[1]))
#     print("X_test1.ndim: {}".format(X_test1.ndim)) # for debug
#     print("X_test1.shape: {}".format(X_test1.shape)) # for debug

    action_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
#     print("action_mask.ndim: {}".format(action_mask.ndim)) # for debug
#     print("action_mask.shape: {}".format(action_mask.shape)) # for debug

    action_mask = np.vstack((action_mask, action_mask))
#     print("action_mask.ndim: {}".format(action_mask.ndim)) # for debug
#     print("action_mask.shape: {}".format(action_mask.shape)) # for debug

    pred1 = sess.run(["discrete_actions"], {
            "vector_observation": X_test1.astype(np.float32),
            "action_masks": action_mask.astype(np.float32),
            })[0]
#     print("pred1.ndim: {}".format(pred1.ndim)) # for debug
#     print("pred1.shape: {}".format(pred1.shape)) # for debug

    # Team 2
    X_test2 = np.vstack((obs[2], obs[3]))
#     print("X_test2.ndim: {}".format(X_test2.ndim)) # for debug
#     print("X_test2.shape: {}".format(X_test2.shape)) # for debug

    pred2 = sess.run(["discrete_actions"], {
            "vector_observation": X_test2.astype(np.float32),
            "action_masks": action_mask.astype(np.float32),
            })[0]
#     print("pred2.ndim: {}".format(pred2.ndim)) # for debug
#     print("pred2.shape: {}".format(pred2.shape)) # for debug

    actions = {
            0: np.array((np.argmax(pred1[0][:3]), np.argmax(pred1[0][3:6]), np.argmax(pred1[0][6:9]))), # Team 1, Agent 1
            1: np.array((np.argmax(pred1[1][:3]), np.argmax(pred1[1][3:6]), np.argmax(pred1[1][6:9]))), # Team 1, Agent 2
            2: np.array((np.argmax(pred2[0][:3]), np.argmax(pred2[0][3:6]), np.argmax(pred2[0][6:9]))), # Team 2, Agent 1
            3: np.array((np.argmax(pred2[1][:3]), np.argmax(pred2[1][3:6]), np.argmax(pred2[1][6:9]))), # Team 2, Agent 2
        }
#     actions = np.array(actions).reshape((-1, self.action_size))

    obs, reward, done, info = env.step(actions)

    team0_reward += reward[0] + reward[1]
    team1_reward += reward[2] + reward[3]
    if done["__all__"]:
        print("Total Reward: ", team0_reward, " x ", team1_reward)
        team0_reward = 0
        team1_reward = 0
        env.reset()
