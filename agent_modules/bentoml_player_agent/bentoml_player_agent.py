from typing import Dict

import numpy as np

import soccer_twos
import requests

class BentomlPlayerAgent(soccer_twos.AgentInterface):
    def __init__(self, env):
        super().__init__()
        self.name: str = "bentoml_Player"
        
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

        X_test1 = observation[0]
        X_test2 = observation[1]

        response1 = requests.post("http://127.0.0.1:3000/predict", json=X_test1.tolist())
        # print("response1.text: ", response1.text) # for debug

        response2 = requests.post("http://127.0.0.1:3000/predict", json=X_test2.tolist())
        # print("response2.text: ", response2.text) # for debug

        result1_array = np.matrix(response1.text).A[0]
        # print("result1_array: ", result1_array) # for debug

        result2_array = np.matrix(response2.text).A[0]
        # print("result2_array: ", result2_array) # for debug

        return {
            0: result1_array,
            1: result2_array,
            }

