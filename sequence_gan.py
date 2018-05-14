import tensorflow as tf
import numpy as np
import random
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from keras.engine.training import _make_batches
import data_utils

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 128  # embedding dimension
HIDDEN_DIM = 128  # hidden state dimension of lstm cell
START_TOKEN = data_utils.GO_ID
PRE_EPOCH_NUM = 10  # supervise (maximum likelihood estimation) epochs
DIS_PRE_EPOCH_NUM = 10
SEED = 88
BATCH_SIZE = 128

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 128
dis_filter_sizes = [1, 2, 3, 4, 5]  # , 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200]  # , 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 128

model_dir = 'model_gan/'
#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200
generated_num = 50000


def target_loss(sess, model, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    """
    changed by zhoujifa
    """
    for (datax, keywords) in data_loader.next_batch():
        g_loss = sess.run(model.pretrain_loss, {model.x: datax, model.keywords: keywords})
        """
        end
        """
        nll.append(g_loss)
    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    """
    changed by zhoujifa
    """
    for index, (datax, keywords) in enumerate(data_loader.next_batch()):
        _, g_loss = trainable_model.pretrain_step(sess, datax, keywords)
        """
        end
        """
        supervised_g_losses.append(g_loss)
        if index % 1000 == 0:
            print('\tloss at {}: {}'.format(index, g_loss))
    return np.mean(supervised_g_losses)


class DataLoader(object):
    def __init__(self, data, keywords, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.keywords = keywords
        pass

    def next_batch(self):
        assert len(self.data) == len(self.keywords)

        self.batches = _make_batches(len(self.data), self.batch_size)
        index_array = np.arange(len(self.data))
        np.random.shuffle(index_array)
        for batch_index, (batch_start, batch_end) in enumerate(self.batches):
            if batch_end - batch_start != self.batch_size:
                # print('skip batch {} {}'.format(batch_start, batch_end))
                continue
            batch_ids = index_array[batch_start:batch_end]
            xdata = self.data[batch_ids]
            xkeywords = self.keywords[batch_ids]
            yield xdata, xkeywords


class DisDataLoader(object):
    def __init__(self, pos_data, neg_data, batch_size):

        self.sentences = np.concatenate([pos_data, neg_data], 0)
        positive_labels = [[0, 1] for _ in pos_data]
        negative_labels = [[1, 0] for _ in neg_data]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)
        # Shuffle the data
        # shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        # self.sentences = self.sentences[shuffle_indices]
        # self.labels = self.labels[shuffle_indices]
        self.batch_size = batch_size

    def next_batch(self):
        self.batches = _make_batches(len(self.labels), self.batch_size)
        index_array = np.arange(len(self.labels))
        np.random.shuffle(index_array)
        for batch_index, (batch_start, batch_end) in enumerate(self.batches):
            if batch_end - batch_start != self.batch_size:
                # print('skip batch {} {}'.format(batch_start, batch_end))
                continue
            batch_ids = index_array[batch_start:batch_end]
            xdata = self.sentences[batch_ids]
            y = self.labels[batch_ids]
            yield xdata, y


def generate_samples(sess, generator, BATCH_SIZE, generated_num, keywords):  # ---- changed by zhoujifa ------ #

    generated_samples = []
    kwd = select_keywords(keywords, generated_num)
    for _ in range(int(generated_num/BATCH_SIZE)):
        generated_samples.extend(generator.generate(sess, kwd[_*BATCH_SIZE:(_+1)*BATCH_SIZE]))
    # for _ in range(int(generated_num / BATCH_SIZE)):
    #     generated_samples.extend(generator.generate(sess))
    return np.array(generated_samples)


def load_model(sess, saver, ckpt_path='model_gan/'):
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
        print('resume from', latest_ckpt)
        saver.restore(sess, latest_ckpt)
        return int(latest_ckpt[latest_ckpt.rindex('-') + 1:])
    else:
        print('building model from scratch')
        sess.run(tf.global_variables_initializer())
        return -1


def select_keywords(keywords, batch=BATCH_SIZE):
    kwd = []
    index_array = np.arange(len(keywords))
    np.random.shuffle(index_array)

    i = 0
    j = 0
    while j < batch:
        while i < len(keywords)-1 and j < batch:
            i = i + 1
            j = j + 1
            kwd.append(keywords[index_array[i]])
            pass
        i = 0
        index_array = np.arange(len(keywords))
        np.random.shuffle(index_array)


    return kwd


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    vocab_dict, vocab_res = data_utils.load_vocab('./vocab.txt')
    data = data_utils.load_data('data.pkl')
    keywords = data_utils.load_data('kwd.pkl')

    print(len(keywords))
    print(len(data))
    # data = data[:1000]
    tn_size = int(len(data) * 0.8)
    tn_loader = DataLoader(data[:tn_size], keywords[:tn_size], BATCH_SIZE)
    ts_loader = DataLoader(data[tn_size:], keywords[tn_size:], BATCH_SIZE)
    print('data 个数: ', len(data))
    vocab_size = len(vocab_dict)
    SEQ_LENGTH = data.shape[1]
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, 1604, START_TOKEN)
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size,
                                  embedding_size=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                  l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    last_epoch = load_model(sess, saver, model_dir)

    if last_epoch <= 0:
        # pre-train generator
        print('Start pre-training...')
        for epoch in range(PRE_EPOCH_NUM):
            loss = pre_train_epoch(sess, generator, tn_loader)
            if epoch % 5 == 0:
                test_loss = target_loss(sess, generator, ts_loader)
                print('pre-train epoch ', epoch, 'train loss', loss, 'test_loss ', test_loss)

        print('Start pre-training discriminator...')
        # Train 3 epoch on the generated data and do this for 50 times
        for epoch_i in range(DIS_PRE_EPOCH_NUM):
            # --------- changed by zhoujifa -------------- #
            gen_data = generate_samples(sess, generator, BATCH_SIZE, generated_num, keywords)

            print(gen_data.shape)
            print(np.shape(data[:tn_size]))

            dis_loader = DisDataLoader(data[:tn_size], gen_data, BATCH_SIZE)

            for _ in range(3):
                losses = []
                for index, (x_batch, y_batch) in enumerate(dis_loader.next_batch()):
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _, dis_loss = sess.run([discriminator.train_op, discriminator.loss], feed)
                    losses.append(dis_loss)
                    if index % 1000 == 0:
                        print('\tepoch: {}, batch index : {}, loss: {}'.format(epoch_i, index, dis_loss))
                print('epoch: {}, loss: {}'.format(epoch_i, np.mean(losses)))
    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')
    for total_batch in range(last_epoch + 1, TOTAL_BATCH):
        # Train the generator for one step
        for it in range(10):
            """
            changed by zhoujifa
            """
            kwd = select_keywords(keywords)
            samples = generator.generate(sess, kwd)
            rewards = rollout.get_reward(sess, samples, kwd, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards, generator.keywords: kwd}
            _ = sess.run(generator.g_updates, feed_dict=feed)
            """
            end
            """

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            test_loss = target_loss(sess, generator, ts_loader)
            print('total_batch: ', total_batch, 'test_loss: ', test_loss)
        # Update roll-out parameters

        rollout.update_params()

        # Train the discriminator
        for _ in range(1):
            # ------ changed by zhoujifa ---------- #
            gen_data = generate_samples(sess, generator, BATCH_SIZE, generated_num, keywords)
            dis_loader = DisDataLoader(data[:tn_size], gen_data, BATCH_SIZE)

            for epoch in range(1):
                losses = []
                for index, (x_batch, y_batch) in enumerate(dis_loader.next_batch()):
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _, dis_loss = sess.run([discriminator.train_op, discriminator.loss], feed)
                    losses.append(dis_loss)
                    if index % 1000 == 0:
                        print('\tepoch: {}, batch index : {}, loss: {}'.format(epoch, index, dis_loss))
                print('\tepoch: {}, loss: {}'.format(epoch, np.mean(losses)))
        saver.save(sess, model_dir + 'poetry.module', global_step=total_batch)
    for i in range(int(5)):
        if i > len(samples):
            break
        arr = samples[i]
        poem = ''
        for index in arr:
            if index != data_utils.EOS_ID:
                poem += vocab_res[index]
        print(poem)
    sess.close()


def generate():
    random.seed(SEED)
    np.random.seed(SEED)
    vocab_dict, vocab_res = data_utils.load_vocab('./vocab.txt')
    data = data_utils.load_data('data.pkl')

    vocab_size = len(vocab_dict)
    SEQ_LENGTH = data.shape[1]

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    samples = generator.generate(sess)
    for i in range(int(1)):
        if i > len(samples):
            break
        arr = samples[i]
        poem = ''
        for index in arr:
            if index != data_utils.EOS_ID:
                poem += vocab_res[index]
        print(poem)


if __name__ == '__main__':
    # if len(sys.argv) > 1 and sys.argv[1] == 'train':
        main()
    # # else:
    #     generate()
