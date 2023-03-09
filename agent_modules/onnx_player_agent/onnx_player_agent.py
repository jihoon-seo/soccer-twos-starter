from typing import Dict

import numpy as np
import onnxruntime as rt

import soccer_twos

class OnnxPlayerAgent(soccer_twos.AgentInterface):
    def __init__(self, env):
        super().__init__()
        self.name: str = "onnx_Player"
        
        self.sess1 = rt.InferenceSession("./trained_models/SoccerTwos.onnx")
        self.sess2 = rt.InferenceSession("./trained_models/SoccerTwos.onnx")
        # self.input_name = self.sess.get_inputs()[0].name
        # self.label_name = self.sess.get_outputs()[0].name

        self.action_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        # print("action_mask.ndim: {}".format(action_mask.ndim)) # for debug
        # print("action_mask.shape: {}".format(action_mask.shape)) # for debug

        self.action_mask = np.vstack((self.action_mask)).transpose()
        # print("action_mask.ndim: {}".format(action_mask.ndim)) # for debug
        # print("action_mask.shape: {}".format(action_mask.shape)) # for debug

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
                e.g., {0: obs[0], 1: obs[1]}
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
                e.g., {0: action_member1, 1: action_member2}
        """

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

        # print("observation[0].ndim: {}".format(observation[0].ndim)) # for debug
        # print("observation[0].shape: {}".format(observation[0].shape)) # for debug

        X_test1 = np.vstack((observation[0])).transpose()
        X_test2 = np.vstack((observation[1])).transpose()
        # print("X_test1.ndim: {}".format(X_test1.ndim)) # for debug
        # print("X_test1.shape: {}".format(X_test1.shape)) # for debug

        pred1 = self.sess1.run(["discrete_actions"], {
            "vector_observation": X_test1.astype(np.float32),
            "action_masks": self.action_mask.astype(np.float32),
            })[0]
        # print("pred1.ndim: {}".format(pred1.ndim)) # for debug
        # print("pred1.shape: {}".format(pred1.shape)) # for debug

        pred2 = self.sess2.run(["discrete_actions"], {
            "vector_observation": X_test2.astype(np.float32),
            "action_masks": self.action_mask.astype(np.float32),
            })[0]

        return {
            0: np.array((np.argmax(pred1[0][:3]), np.argmax(pred1[0][3:6]), np.argmax(pred1[0][6:9]))),
            1: np.array((np.argmax(pred2[0][:3]), np.argmax(pred2[0][3:6]), np.argmax(pred2[0][6:9]))),
            }
