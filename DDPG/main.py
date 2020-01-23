import numpy as np
import tensorflow as tf
import gym
tf.enable_eager_execution()
layers = tf.keras.layers


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size=int(1e6)):
        self.obs1_buf = np.zeros([max_size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([max_size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([max_size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class NetBlock(tf.keras.Model):
    def __init__(self, hidden_sizes=(32,), activation=tf.nn.tanh, output_activation=None):
        super(NetBlock, self).__init__()
        ns = len(hidden_sizes)-1
        self.hid_layers = []
        for n in range(ns):
            self.hid_layers.append(layers.Dense(hidden_sizes[n],
                        activation=activation, name="dense_"+str(n+1)))
        self.hid_layers.append(layers.Dense(hidden_sizes[-1],
                        activation=output_activation, name="dense_last"))

    def call(self, x):
        for i in range(len(self.hid_layers)):
            x = self.hid_layers[i](x)
        return x


class ActorCritic(tf.keras.Model):
    def __init__(self, hidden_sizes=(300,), activation=tf.nn.relu,
                 output_activation=tf.tanh, action_space=None):
        super(ActorCritic, self).__init__()
        # 根据 action space 确定动作空间和范围
        self.act_dim = action_space.sample().shape[-1]
        self.act_limit = action_space.high[0]
        # actor and critic
        self.net_pi = NetBlock(list(hidden_sizes)+[self.act_dim], activation, output_activation)
        self.net_q = NetBlock(list(hidden_sizes)+[1], activation, None)
        # target actor and critic
        self.tar_net_pi = NetBlock(list(hidden_sizes)+[self.act_dim], activation, output_activation)
        self.tar_net_q = NetBlock(list(hidden_sizes) + [1], activation, None)

    def get_action(self, s, noise_scale=0.1):
        a = self.net_pi(np.expand_dims(s, axis=0)) * self.act_limit
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def get_pi_q(self, s, a, net="main"):
        assert net in ['main', 'target']
        if net == 'main':
            pi_net_for_compute, q_net_for_compute = self.net_pi, self.net_q
        else:
            pi_net_for_compute, q_net_for_compute = self.tar_net_pi, self.tar_net_q

        pi = pi_net_for_compute(s) * self.act_limit
        q = tf.squeeze(q_net_for_compute(tf.concat([s, a], axis=-1))) if net == 'main' else None
        q_pi = tf.squeeze(q_net_for_compute(tf.concat([s, pi], axis=-1)))
        return pi, q, q_pi

    def get_target(self, r, s_next, done, gamma=0.99):
        pi_tar_next, _, q_pi_tar_next = self.get_pi_q(s_next, None, net='target')
        return r + gamma * (1-done) * q_pi_tar_next

    def cal_loss(self, s, a, r, s_next, done):
        # pi loss
        pi, q, q_pi = self.get_pi_q(s, a)
        pi_loss = -tf.reduce_mean(q_pi)
        # q loss
        target = self.get_target(r, s_next, done)
        q_loss = tf.reduce_mean((q - tf.stop_gradient(target)) ** 2)
        return pi_loss, q_loss

    def target_init(self):
        self.tar_net_pi.set_weights(self.net_pi.get_weights())
        self.tar_net_q.set_weights(self.net_q.get_weights())

    def target_update(self, polyak=0.995):
        self.tar_net_pi.set_weights([polyak * w1 + (1. - polyak) * w2 for w1, w2 in
                    zip(self.tar_net_pi.get_weights(), self.net_pi.get_weights())])
        self.tar_net_q.set_weights([polyak * w1 + (1. - polyak) * w2 for w1, w2 in
                    zip(self.tar_net_q.get_weights(), self.net_q.get_weights())])

    def call(self, x):
        pass


def ddpg(env_name, steps_per_epoch=5000, epochs=500, pi_lr=1e-3, q_lr=1e-3,
         batch_size=100, start_steps=10000, max_ep_len=1000, save_freq=10):
    """ Input:
        start_steps(int): Number of steps for uniform - random action selection, before
            running real policy. Helps exploration.
        act_noise(float): Stddev for Gaussian exploration noise added to policy at training time.
            (At test time, no noise is added.)
        max_ep_len(int): Maximum length of trajectory / episode / rollout.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """
    env, test_env = gym.make(env_name), gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    ac_space = env.action_space

    # actor-critic
    actor_critic = ActorCritic(action_space=ac_space)

    # replay buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim)

    # optimizer
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)

    # target network init
    actor_critic.target_init()

    # start
    o, r, d, ep_ret, ep_len = env.reset().astype(np.float32), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    for t in range(total_steps):
        # 在起初的 start_steps 个时间步都随机交互，先填充经验池
        if t > start_steps:
            a = actor_critic.get_action(o.astype(np.float32))
        else:
            a = env.action_space.sample()

        # 与环境交互, 存储样本
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        d = False if ep_len == max_ep_len else d    # 连续交互 1000 个时间步.
        replay_buffer.store(o, a, r, o2, d)
        o = o2

        # 周期结束, 训练
        if d or (ep_len == max_ep_len):
            print("Train, total steps=", t, ", episode steps=", ep_len, ", episode return=", ep_ret)
            for _ in range(ep_len):         # 训练 1000 个时间步
                batch = replay_buffer.sample_batch(batch_size)
                with tf.GradientTape(persistent=True) as tape:
                    pi_loss, q_loss = actor_critic.cal_loss(
                        s=batch['obs1'], a=batch['acts'], r=batch['rews'],
                        s_next=batch['obs2'], done=batch['done'])
                gradients_pi = tape.gradient(pi_loss, actor_critic.net_pi.trainable_variables)
                gradients_q = tape.gradient(q_loss, actor_critic.net_q.trainable_variables)
                pi_optimizer.apply_gradients(zip(gradients_pi, actor_critic.net_pi.trainable_variables))
                q_optimizer.apply_gradients(zip(gradients_q, actor_critic.net_q.trainable_variables))
                # target update
                actor_critic.target_update()
                if _ % 100 == 0:
                    print(".", end="", flush=True)
            print("Pi loss:", pi_loss.numpy(), "Q loss:", q_loss.numpy())
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # 进行测试, without exploration
        if t > 0 and t % steps_per_epoch == 0:
            print("start test: ", t)
            epoch = t // steps_per_epoch
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                actor_critic.save_weights("model/model_"+str(epoch)+".h5")

            for j in range(10):
                o_test, r_test, d_test, ep_ret_test, ep_len_test = test_env.reset().astype(np.float32), 0, False, 0, 0
                while not (d_test or (ep_len_test == max_ep_len)):
                    o_test, r_test, d_test, _ = test_env.step(actor_critic.get_action(o_test.astype(np.float32), noise_scale=0.))
                    ep_ret_test += r_test
                    ep_len_test += 1
                print("test episode return:", ep_ret_test, ", episode length:", ep_len_test)
            print()


def play(model_path, env_name="HalfCheetah-v2", max_ep_len=1000):
    env = gym.make(env_name)
    ac_space = env.action_space

    # actor-critic
    actor_critic = ActorCritic(action_space=ac_space)
    pi_loss, q_loss = actor_critic.cal_loss(
            s=np.expand_dims(env.reset().astype(np.float32), axis=0),
            a=np.expand_dims(env.action_space.sample(), axis=0),
            r=np.array([[0.]]).astype(np.float32),
            s_next=np.expand_dims(env.reset().astype(np.float32), axis=0),
            done=np.array([[0.]]).astype(np.float32))
    print(pi_loss, q_loss)
    print("load weights.")
    actor_critic.load_weights(model_path)
    print("done")

    # run
    for j in range(10):
        o, r, d, ep_ret, ep_len = env.reset().astype(np.float32), 0, False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            o, r, d, _ = env.step(
                actor_critic.get_action(o.astype(np.float32), noise_scale=0.))
            env.render()
            ep_ret += r
            ep_len += 1
        print("Test episode return:", ep_ret, ", episode length:", ep_len)
    print()


if __name__ == '__main__':
    # env
    # env = gym.make("HalfCheetah-v2")
    # ac_space = env.action_space

    # test NetBlock
    # s = tf.convert_to_tensor(np.random.random((10, 1)), tf.float32)
    # net_block = NetBlock()
    # out = net_block(s)
    # print(out.shape)
    # print(len(net_block.hid_layers), net_block.hid_layers[0].trainable_variables)
    # print("------")
    #
    # net_block_2 = NetBlock()
    # out2 = net_block_2(s)
    # print(len(net_block_2.hid_layers), net_block_2.hid_layers[0].trainable_variables)
    # print("------")
    #
    # net_block_2.set_weights([w1*1.+w2*2. for w1, w2 in zip(net_block.get_weights(), net_block_2.get_weights())])
    # print(len(net_block_2.hid_layers), net_block_2.hid_layers[0].trainable_variables)
    # print("------")

    # test actor critic
    # s = tf.convert_to_tensor(np.random.random((10, 5)), tf.float32)
    # a = tf.convert_to_tensor(np.random.random((10, 6)), tf.float32)
    # r = tf.convert_to_tensor(np.random.random((10,)) + 10., tf.float32)
    # s_next = tf.convert_to_tensor(np.random.random((10, 5)), tf.float32)
    # done = tf.zeros(shape=(10,), dtype=tf.float32)
    # s = np.random.random((10, 5)).astype(np.float32)
    # a = np.random.random((10, 6)).astype(np.float32)
    # r = (np.random.random((10,)) + 10.).astype(np.float32)
    # s_next = (np.random.random((10, 5))).astype(np.float32)
    # done = np.zeros((10,)).astype(np.float32)
    # actor_critic = ActorCritic(action_space=ac_space)
    # pi_loss, q_loss = actor_critic.cal_loss(s=s, a=a, r=r, s_next=s_next, done=done)
    # print(pi_loss, q_loss)

    #
    # ddpg(env_name="HalfCheetah-v2")
    play(model_path="model/model_60.h5")
