from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import pdb
import numpy as np

from cs224r.infrastructure import pytorch_util as ptu

class IQLCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        self.optimizer_spec = optimizer_spec
        network_initializer = hparams['q_func']
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)

        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.mse_loss = nn.MSELoss()
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)

        # TODO define value function
        # HINT: see Q_net definition above and optimizer below
        # HINT: Define using same hparams as Q_net, but adjust output dimensions
        ### YOUR CODE HERE ###
        self.v_net = network_initializer(self.ob_dim, 1)
        self.v_net.to(ptu.device)
        ### YOUR CODE HERE ###

        self.v_optimizer = self.optimizer_spec.constructor(
            self.v_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        
        self.v_learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.v_optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.iql_expectile = hparams['iql_expectile']

    def expectile_loss(self, diff):
        # TODO: Implement the expectile loss given the difference between q and v
        # HINT: self.iql_expectile provides the \tau value as described 
        # in the problem statement.
        # HINT: You can return a tensor with same dimensionality as diff and 
        # aggregate it later
        ### YOUR CODE HERE ###
        tau = self.iql_expectile
        # Since tau < 1, |tau - 1| = 1 - tau
        loss = torch.where(diff <= 0, (1 - tau) * diff**2, tau * diff**2).mean()
        return loss
        ### YOUR CODE HERE ###


    def update_v(self, ob_no, ac_na):
        """
        Update value function using expectile loss
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)

        # TODO: Compute loss for v_net
        # HINT: use target q network to train V
        # HINT: Use self.expectile_loss as defined above, 
        # passing in the difference between the computed targets and predictions
        ### YOUR CODE HERE ###
        # ac_na has a shape of (batch_size)
        # For each element in batch get q_net_target(ob_no)[i, ac_na[i]]
        # Then subtract v_net(ob_no)[i] do this without gather
        diff = self.q_net_target(ob_no).gather(1, ac_na.unsqueeze(1)) - self.v_net(ob_no)
        value_loss = self.expectile_loss(diff)
        ### YOUR CODE HERE ###
        

        self.v_optimizer.zero_grad()
        value_loss.backward()
        utils.clip_grad_value_(self.v_net.parameters(), self.grad_norm_clipping)
        self.v_optimizer.step()
        
        self.v_learning_rate_scheduler.step()

        return {'Training V Loss': ptu.to_numpy(value_loss)}



    def update_q(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
        Use target v network to train Q
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)
        
        
        # TODO: Compute loss for updating Q_net parameters
        # HINT: Note that if the next state is terminal, 
        # its target reward value needs to be adjusted.
        ### YOUR CODE HERE ###
        
        # Step 1: Compute the target values
        target_values = self.v_net(next_ob_no)
        target_values = target_values.reshape(-1)
        # debug(target_values)

        # Step 2: Compute the target q values
        # debug(reward_n)
        # debug(terminal_n)
        target_q_values = reward_n + self.gamma * target_values * (1 - terminal_n)
        target_q_values = target_q_values.reshape(-1, 1)
        # debug(target_q_values)

        # Step 3: Compute the q values
        q_values = self.q_net(ob_no).gather(1, ac_na.unsqueeze(1))
        # debug(q_values)

        # Step 4: Compute the loss
        loss = self.mse_loss(q_values, target_q_values)
        # debug(loss)
        # print("")

        ### YOUR CODE HERE ###
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        
        self.learning_rate_scheduler.step()

        return {'Training Q Loss': ptu.to_numpy(loss)}

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
