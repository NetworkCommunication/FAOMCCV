import numpy as np
import torch
from torch.autograd import Variable
from sklearn.neighbors import NearestNeighbors

from DRL.agents.memory.memory import NStepReplayBuffer
from DRL.agents.pdqn import PDQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PDQNNStepAgent(PDQNAgent):
    """
    P-DQN agent using mixed n-step return targets
    """

    NAME = "P-DQN N-Step Agent"

    def __init__(self,
                 *args,
                 beta=0.2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.replay_memory = NStepReplayBuffer(self.replay_memory_size, self.seed)

    def __str__(self):
        desc = super().__str__()
        desc += "Beta: {}\n".format(self.beta)
        # print("haha",type(self.beta))
        return desc

    def _optimize_td_loss(self):

        if len(self.replay_memory) < self.batch_size:
            return

        def knn_sample(memory, batch_size, n):
            experiences = memory.buffer
            states = np.array([experience[0] for experience in experiences])
            knn = NearestNeighbors(n_neighbors=int(n)).fit(states)
            random_idx = np.random.randint(len(states))
            center_state = states[random_idx]
            distances, indices = knn.kneighbors([center_state])
            nearest_samples = [experiences[idx] for idx in indices[0]]

            furthest_indices = np.argsort(distances[0])[-(batch_size - int(n)):]
            furthest_samples = [experiences[idx] for idx in furthest_indices]

            return nearest_samples + furthest_samples

        # Sample a batch from replay memory using KNN
        n = self.batch_size / 2
        experiences = knn_sample(self.replay_memory, self.batch_size, n)
        states, actions, rewards, next_states, terminals, n_step_returns = zip(*experiences)
        states = torch.tensor(states).to(device)
        actions_combined = torch.tensor(actions).to(device)  # make sure to separate actions and action-parameters
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.tensor(rewards).to(device).squeeze()
        next_states = torch.tensor(next_states).to(device)
        terminals = torch.tensor(np.float32(terminals)).to(device).squeeze()
        n_step_returns = torch.tensor(n_step_returns).to(device)
        # ---------------------- optimise critic ----------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()

            off_policy_target = rewards + (1 - terminals) * self.gamma * Qprime
            on_policy_target = n_step_returns.squeeze()
            target = self.beta * on_policy_target + (1. - self.beta) * off_policy_target

        q_values = self.actor(states, action_parameters)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze().float()
        y_expected = target.float()
        loss_Q = self.loss_func(y_predicted, y_expected)
        # print("loss_Q", type(loss_Q))
        self.actor_optimiser.zero_grad()
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()

        # ---------------------- optimise actor ----------------------
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True
        Q_val = self.actor(states, action_params)
        Q_loss = torch.mean(torch.sum(Q_val, 1))
        self.actor.zero_grad()
        Q_loss.backward()
        from copy import deepcopy
        delta_a = deepcopy(action_params.grad.data)
        # step 2
        action_params = self.actor_param(Variable(states))
        delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)

        out = -torch.mul(delta_a, action_params)
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape).to(device))
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)

        self.actor_param_optimiser.step()
