import copy
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class AdversarialAgent(BaseAgent):

    def __init__(self):

        self.instruction = "Let's think step by step, and put your final answer within \\boxed{}."
        self._trajectory = Trajectory()
        self.messages = []

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        if not isinstance(observation, dict):
            return

        messages = observation.get("messages")
        # SingleTurnEnvironment returns {} after the terminal step. In that case we
        # keep the existing conversation, which already includes the assistant reply.
        if messages is None:
            return
        if not isinstance(messages, list):
            raise TypeError("'messages' in observation must be a list.")
        self.messages = copy.deepcopy(messages)

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's internal state based on the model's response.
        """
        self.messages.append({"role": "assistant", "content": response})
        new_step = Step(chat_completions=copy.deepcopy(self.chat_completions))
        self.trajectory.steps.append(new_step)

        return Action(action=response)

    def reset(self):
        """Reset agent state for new episode."""
        self._trajectory = Trajectory()
        self.messages = []

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return conversation history for model interaction."""
        # remove thinking from assistant messages if not accumulate_thinking except the last one
        messages = copy.deepcopy(self.messages)
        return messages

    @property
    def trajectory(self) -> Trajectory:
        """Return complete interaction trajectory."""
        return self._trajectory

    def get_current_state(self) -> Step:
        """Returns the current step/state of the agent."""
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]
