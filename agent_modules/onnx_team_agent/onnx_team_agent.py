from typing import Dict

import numpy as np
import onnxruntime as rt

import soccer_twos

class OnnxTeamAgent(soccer_twos.AgentInterface):
    def __init__(self, env):
        super().__init__()
        self.name: str = "onnx_Team"
        
        self.sess = rt.InferenceSession("./trained_models/SoccerTwos.onnx")
        # self.input_name = self.sess.get_inputs()[0].name
        # self.label_name = self.sess.get_outputs()[0].name

        self.action_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        # print("action_mask.ndim: {}".format(action_mask.ndim)) # for debug
        # print("action_mask.shape: {}".format(action_mask.shape)) # for debug

        self.action_mask = np.vstack((self.action_mask, self.action_mask))
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
        # print("observation[1].ndim: {}".format(observation[1].ndim)) # for debug
        # print("observation[1].shape: {}".format(observation[1].shape)) # for debug

        # X_test = np.concatenate((observation[0], observation[1]))
        # X_test = np.array([[observation[0]], [observation[1]]])
        X_test = np.vstack((observation[0], observation[1]))
        # print("X_test.ndim: {}".format(X_test.ndim)) # for debug
        # print("X_test.shape: {}".format(X_test.shape)) # for debug

        pred = self.sess.run(["discrete_actions"], {
            "vector_observation": X_test.astype(np.float32),
            "action_masks": self.action_mask.astype(np.float32),
            })[0]
        # print("pred.ndim: {}".format(pred.ndim)) # for debug
        # print("pred.shape: {}".format(pred.shape)) # for debug

        return {
            0: np.array((np.argmax(pred[0][:3]), np.argmax(pred[0][3:6]), np.argmax(pred[0][6:9]))),
            1: np.array((np.argmax(pred[1][:3]), np.argmax(pred[1][3:6]), np.argmax(pred[1][6:9]))),
            }
