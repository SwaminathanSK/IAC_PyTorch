import numpy as np
import torch
import time
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def t(v, dtype=None, device=device, requires_grad=False):
    """Shortcut for tensor instantiation with device."""
    return torch.tensor(v, device=device, dtype=dtype, requires_grad=requires_grad)


def mc_values(rewards, hyper_ps):
    """
    Gives a list of MC estimates for a given list of samples from an RL environment.
    The MC estimator is used for this computation.

    :param rewards: The rewards that were obtained while exploring the RL environment.
    :param hyper_ps: The hyper-parameters to be used.

    :return: The MC estimates.
    """

    mcs = np.zeros(shape=(len(rewards),))
    discount_factor = dict_with_default(hyper_ps, 'discount_factor', .9)

    for i, reward in enumerate(rewards):
        ret = reward
        gamma = 1.

        for j in range(i + 1, len(rewards)):
            gamma *= discount_factor
            ret += gamma * rewards[j]

        mcs[i] = ret

    return mcs


def td_values(replay_buffers, state_values, hyper_ps):
    """
    Gives a list of TD estimates for a given list of samples from an RL environment.
    The TD(λ) estimator is used for this computation.

    :param replay_buffers: The replay buffers filled by exploring the RL environment.
    Includes: states, rewards, "final state?"s.
    :param state_values: The currently estimated state values.
    :param hyper_ps: The hyper-parameters to be used.

    :return: The TD estimates.
    """

    states, rewards, dones = replay_buffers
    sample_count = len(states)
    tds = np.zeros(shape=(sample_count,))

    discount_factor = dict_with_default(hyper_ps, 'discount_factor', .9)
    alpha = dict_with_default(hyper_ps, 'alpha', .95)
    lam = dict_with_default(hyper_ps, 'lambda', .95)

    val = 0.
    for i in range(sample_count - 1, -1, -1):
        state_value = state_values[i]
        next_value = 0. if dones[i] else state_values[i + 1]

        error = rewards[i] + discount_factor * next_value - state_value
        val = alpha * error + discount_factor * lam * (1 - dones[i]) * val

        tds[i] = val + state_value

    return tds


def critic_inputs(trajectories, next_states=False):
    """
    Extracts the relevant inputs for the V-critic from the given list of trajectories.

    :param trajectories: The trajectories from which the information should be taken.
    :param next_states: Extract the next-state entries from the samples instead of the current states.

    :return: The extracted information in the form of a batched tensor.
    """

    return torch.cat([(tr.next_state if next_states else tr.state).flatten().unsqueeze(0) for tr in trajectories]).to(device)


def nan_in_model(model):
    """
    Checks if the given model holds any parameters that contain NaN values.

    :param model: The model to be checked for NaN entries.

    :return: Whether the model contain NaN parameters.
    """

    for p in model.parameters():
        p_nan = torch.isnan(p.data).flatten().tolist()
        if any(p_nan):
            return True

    return False


def dict_with_default(dict, key, default):
    """
    Returns the value contained in the given dictionary for the given key, if it exists.
    Otherwise, returns the given default value.

    :param dict: The dictionary from which the value should be read.
    :param key: The key to look for in the dictionary.
    :param default: The fallback value, in case the dictionary doesn't contain the desired key.

    :return: The value read from the dictionary, if it exists. The default value otherwise.
    """

    if key in dict:
        return dict[key]
    else:
        return default


def xavier_init(m):
    """
    Xavier normal initialisation for layer m.

    :param m: The layer to have its weight and bias initialised.
    """

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def kaiming_init(m):
    """
    Kaiming normal initialisation for layer m.

    :param m: The layer to have its weight and bias initialised.
    """

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def obs_to_state(observation):
    """
    Converts a given observation into a state tensor.
    Necessary as a catch-all for MuJoCo environments.

    :param observation: The observation received from the environment.

    :return: The state tensor.
    """

    if type(observation) is dict:
        state = state_from_mujoco(observation)
    else:
        state = observation

    return t(state).float()


def state_from_mujoco(observation):
    """
    Converts the observation parts returned by a MuJoCo environment into a single vector of values.

    :param observation: The observation containing the relevant parts.

    :return: A single vector containing all the observation information.
    """

    ag = observation['achieved_goal']
    dg = observation['desired_goal']
    obs = observation['observation']

    return np.concatenate([ag, dg, obs])



############################################
############################################

def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)['observation']

    # predicted
    ob = np.expand_dims(true_states[0],0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac,0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states

def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def mean_squared_error(a, b):
    return np.mean((a-b)**2)

############################################
############################################

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        if render:
            # pdb.set_trace()
            if 'rgb_array' in render_mode:
                if hasattr(env.unwrapped, 'sim'):
                    if 'track' in env.unwrapped.model.camera_names:
                        image_obs.append(env.unwrapped.sim.render(camera_name='track', height=500, width=500)[::-1])
                    else:
                        image_obs.append(env.unwrapped.sim.render(height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)
        obs.append(ob)
        if policy:
            ac = policy.get_action(ob)
            ac = ac[0]
        else:
            ac = env.action_space.sample()
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done or steps > max_path_length:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect rollouts using policy
        until we have collected min_timesteps_per_batch steps
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        #collect rollout
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)

        #count steps
        timesteps_this_batch += get_pathlength(path)
        print('At timestep:    ', timesteps_this_batch, '/', min_timesteps_per_batch, end='\r')

    return paths, timesteps_this_batch

def sample_uniform_trajectories(env, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect rollouts using policy
        until we have collected min_timesteps_per_batch steps
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        #collect rollout
        path = sample_trajectory(env, None, max_path_length, render, render_mode)
        paths.append(path)

        #count steps
        timesteps_this_batch += get_pathlength(path)
        print('At timestep:    ', timesteps_this_batch, '/', min_timesteps_per_batch, end='\r')
    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect ntraj rollouts using policy
    """
    paths = []
    for i in range(ntraj):
        # collect rollout
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
    return paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


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

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data

