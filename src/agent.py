import random
from collections import deque

import numpy as np
import torch

from utilities.debug import DebugType
from utilities.utils import t, nan_in_model, dict_with_default, \
    td_values, mc_values, obs_to_state

import sys
import gym
import torch
import pylab
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from utilities.prioritized_memory import Memory
import math
import pickle
from utilities import utils


class AWRAgent:

    with open("bc_policy.pkl", "rb") as file:
        beta_policy = pickle.load(file)

    name = "awr"

    @staticmethod
    def train(models, environment, hyper_ps, debug_type, writer, beta_policy=beta_policy):
        assert len(models) == 2, "AWR needs exactly two models to function properly."
        actor, critic = models

        # replay buffer
        sample_mod = dict_with_default(hyper_ps, 'sample_mod', 10)
        max_buffer_size = hyper_ps['replay_size']
        states = deque(maxlen=max_buffer_size)
        actions = deque(maxlen=max_buffer_size)
        rewards = deque(maxlen=max_buffer_size)
        dones = deque(maxlen=max_buffer_size)
        replay_fill_threshold = dict_with_default(hyper_ps, 'replay_fill_threshold', 0.)
        random_exploration = dict_with_default(hyper_ps, 'random_exploration', False)

        # learning time setup
        max_epoch_count = hyper_ps['max_epochs']
        epoch = 0
        pre_training_epochs = 0
        max_pre_epochs = 150

        memory_size = 100000
        # create prioritized replay memory using SumTree
        memory = Memory(memory_size)

        ntraj = 200
        max_path_length = 1000
        paths = utils.sample_n_trajectories(environment, beta_policy, ntraj, max_path_length)
        observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards = AWRAgent.convert_listofrollouts(paths)

        for i in range(len(observations)):
            memory = AWRAgent.append_sample(observations[i], actions[i], np.array(concatenated_rewards[i]), next_observations[i], np.array(terminals[i]), memory)
        
        memory_batch_size = 10000

        # algorithm specifics
        beta = hyper_ps['beta']
        critic_steps_start = hyper_ps['critic_steps_start']
        critic_steps_end = hyper_ps['critic_steps_end']
        actor_steps_start = hyper_ps['actor_steps_start']
        actor_steps_end = hyper_ps['actor_steps_end']
        batch_size = hyper_ps['batch_size']
        # batch_size = 64
        max_advantage_weight = hyper_ps['max_advantage_weight']
        min_log_pi = hyper_ps['min_log_pi']

        # debug helper field
        debug_full = debug_type == DebugType.FULL

        # critic pre-training
        critic_threshold = hyper_ps['critic_threshold']
        critic_suffices_required = hyper_ps['critic_suffices_required']
        critic_suffices_count = 0
        critic_suffices = False

        # evaluation
        validation_epoch_mod = dict_with_default(hyper_ps, 'validation_epoch_mod', 30)
        test_iterations = hyper_ps['test_iterations']

        AWRAgent.compute_validation_return(
            actor,
            environment,
            hyper_ps,
            debug_type,
            test_iterations,
            epoch,
            writer
        )

        # mini_batch, idxs, is_weights = memory.sample(batch_size)

        while epoch < max_epoch_count + pre_training_epochs:

            mini_batch, idxs, is_weights = memory.sample(memory_batch_size)
            # print(mini_batch)
            mini_batch = np.array(mini_batch).transpose()

            states = np.vstack(mini_batch[0])
            actions = np.vstack(mini_batch[1]) # removed list()
            rewards = list(mini_batch[2]) # removed list()
            next_states = np.vstack(mini_batch[3])
            dones = mini_batch[4]

            # bool to binary
            dones = dones.astype(int)

            # Q function of current state
            states = torch.Tensor(states)

            # one-hot encoding
            a = torch.Tensor(actions)

            rhos = AWRAgent.get_policy_density(actor, a, states)-AWRAgent.get_policy_density(beta_policy, a, states, 1)
            # print(rhos)
            rhos = np.exp(rhos.detach().numpy())

            # memory = AWRAgent.append_sample()
            for i in range(memory_batch_size):
                idx = idxs[i]
                memory.update(idx, rhos[i])
            
            mini_batch, idxs, is_weights = memory.sample(memory_batch_size)
            # print(mini_batch)
            mini_batch = np.array(mini_batch).transpose()

            states = np.vstack(mini_batch[0])
            actions = np.vstack(mini_batch[1]) # removed list()
            rewards = list(mini_batch[2]) # removed list()
            next_states = np.vstack(mini_batch[3])
            dones = mini_batch[4]

            # bool to binary
            dones = dones.astype(int)

            # Q function of current state
            states = torch.Tensor(states)

            # one-hot encoding
            a = torch.Tensor(actions)

            print(f"\nEpoch: {epoch}")

            # set actor and critic update steps
            epoch_percentage = ((epoch - pre_training_epochs) / max_epoch_count)
            critic_steps = critic_steps_start + int((critic_steps_end - critic_steps_start) * epoch_percentage)
            actor_steps = actor_steps_start + int((actor_steps_end - actor_steps_start) * epoch_percentage)

            dq_states = states
            states = np.array(states)
            dq_actions = actions
            actions = np.array(actions)
            dq_rewards = rewards
            rewards = np.array(rewards)
            dq_next_states = next_states
            next_states = np.array(next_states)

            print("Reached here")

            # training the critic
            avg_loss = 0.
            state_values = np.array(critic.evaluate(t(states)).squeeze(1).cpu())
            next_state_values = np.array(critic.evaluate(t(next_states)).squeeze(1).cpu())

            tds = td_values((states, rewards, dones), state_values, hyper_ps, next_state_values)
            for _ in range(critic_steps):
                indices = random.sample(range(len(states)), batch_size)
                ins = t(states[indices])
                tars = t(tds[indices])

                outs = critic(ins)
                loss = critic.update(states, actions, None, rewards, dones, actor, tars)
                loss = loss['Critic Training Loss']
                avg_loss += loss
            avg_loss /= critic_steps
            print(f"average critic loss: {avg_loss}")

            if nan_in_model(critic):
                print("NaN values in critic\nstopping training")
                break

            writer.add_scalar('critic_loss', avg_loss, epoch)

            if avg_loss <= critic_threshold:
                critic_suffices_count += 1
            else:
                critic_suffices_count = 0

            if critic_suffices_count >= critic_suffices_required:
                critic_suffices = True

            if critic_suffices:
                # training the actor
                avg_loss = 0.
                state_values = np.array(critic.evaluate(t(states)).squeeze(1).cpu())
                next_state_values = np.array(critic.evaluate(t(next_states)).squeeze(1).cpu())
                returns = td_values((states, rewards, dones), state_values, hyper_ps, next_state_values)
                advantages = returns - state_values
                for _ in range(actor_steps):
                    indices = random.sample(range(len(states)), batch_size)

                    advantage_weights = np.exp(advantages[indices] / beta)
                    advantage_weights = t(np.minimum(advantage_weights, max_advantage_weight))

                    normal, _ = actor(t(states[indices]))
                    log_pis = normal.log_prob(t(actions[indices]))
                    log_pis = torch.clamp(log_pis, min=min_log_pi)

                    losses = -log_pis * advantage_weights
                    losses = losses / batch_size  # normalise wrt the batch size
                    actor.backward(losses)

                    mean_loss = torch.sum(losses)
                    avg_loss += mean_loss
                avg_loss /= actor_steps
                print(f"average actor loss: {avg_loss}")

                if nan_in_model(actor):
                    print("NaN values in actor\nstopping training")
                    break

                writer.add_scalar('actor_loss', avg_loss, epoch)
            else:
                pre_training_epochs += 1
                if pre_training_epochs > max_pre_epochs:
                    print("critic couldn't be trained in appropriate time\nstopping training")
                    break

            epoch += 1

            if critic_suffices and epoch % validation_epoch_mod == 0:
                print("avg_return: ", AWRAgent.eval_episode(actor, environment))

            states = dq_states
            actions = dq_actions
            rewards = dq_rewards
            next_states = dq_next_states

        return actor, critic

    @staticmethod
    def compute_validation_return(actor, env, hyper_ps, debug_type, iterations, epoch, writer):
        print("computing average return")
        sample_return = AWRAgent.validation_return(actor, env, hyper_ps, debug_type, iterations)
        writer.add_scalar('return', sample_return, epoch)
        print(f"return: {sample_return}")

    @staticmethod
    def validation_return(actor, env, hyper_ps, debug_type, iterations):
        sample_return = 0.
        for _ in range(iterations):
            s, a, r, d = [], [], [], []
            AWRAgent.sample_from_env(
                actor,
                env,
                debug_type != DebugType.NONE,
                exploration=False,
                replay_buffers=(s, a, r, d)
            )
            mcs = mc_values(r, hyper_ps)
            sample_return += np.mean(mcs)

        sample_return /= iterations
        return sample_return
    
    def eval_step(actor_model, state):
        state = t(state)
        with torch.no_grad():
            _, a = actor_model.forward(state)
        return a.cpu().numpy()
    
    def eval_episode(actor_model, env, timeout=10000, log_traj=False):
        mean_avg_return = 0
        for i in range(5):
            ep_traj = []
            state = env.reset()
            total_rewards = 0
            ep_steps = 0
            done = False
            while True:
                action = AWRAgent.eval_step(actor_model, state)
                last_state = state
                state, reward, done, _ = env.step(action)
                total_rewards += reward
                ep_steps += 1
                if done or ep_steps == timeout:
                    break
            mean_avg_return += total_rewards

        return mean_avg_return/5
    
    # save sample (error,<s,a,r,s'>) to the replay memory
    @staticmethod
    def append_sample(state, action, reward, next_state, done, memory):
        out = np.array([state, action, reward, next_state, done], dtype=object)

        memory.add(1, out)
        return memory

    @staticmethod
    def sample_from_env(actor_model, env, debug, exploration, replay_buffers):
        states, actions, rewards, dones = replay_buffers
        obs = env.reset()
        state = obs_to_state(obs)
        done = False

        if debug:
            env.render()

        while not done:
            if exploration:
                action = t(env.action_space.sample())
            else:
                normal, action = actor_model.evaluate(state)
            # print(action.shape)
            res = env.step(action.cpu().numpy())

            reward = res[1]
            done = res[2]
            states.append(np.array(state.cpu()))
            actions.append(np.array(action.cpu()))
            rewards.append(reward)
            dones.append(done)

            state = obs_to_state(res[0])

            if debug:
                env.render()

    @staticmethod
    def test(models, environment, hyper_ps, debug_type):
        actor, _ = models
        return AWRAgent.validation_return(actor, environment, hyper_ps, debug_type, hyper_ps['test_iterations'])
    
    @staticmethod
    def get_policy_density(policy_net, action, state, behave = 0):
        if behave == 1:
            # print(policy_net.forward(state).log_prob(action))
            return policy_net.forward(state).log_prob(action)
        normal, act = policy_net.forward(state)
        # print(act)
        # print(normal.log_prob(action))
        return normal.log_prob(action).view(-1)


    @staticmethod
    def convert_listofrollouts(paths):
        """
            Take a list of rollout dictionaries
            and return separate arrays,
            where each array is a concatenation of that array from across the rollouts
        """
        observations = np.concatenate([path["observation"] for path in paths])
        actions = np.concatenate([path["action"] for path in paths])
        next_observations = np.concatenate([path["next_observation"] for path in paths])
        terminals = np.concatenate([path["terminal"] for path in paths])
        concatenated_rewards = np.concatenate([path["reward"] for path in paths])
        unconcatenated_rewards = [path["reward"] for path in paths]

        return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards
            
