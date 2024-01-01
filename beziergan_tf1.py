import tensorflow as tf

EPSILON = 1e-7

def generator(c, z, latent_dim, noise_dim):
    depth_cpw = 32 * 8
    dim_cpw = int((31 + 1) / 8)
    kernel_size = (4, 3)
    if noise_dim == 0:
        cz = c
    else:
        cz = tf.concat([c, z], axis=-1)

    cpw = tf.layers.dense(cz, 1024)
    cpw = tf.layers.batch_normalization(cpw, momentum=0.9)  # , training=training)
    cpw = tf.nn.leaky_relu(cpw, alpha=0.2)

    cpw = tf.layers.dense(cpw, dim_cpw * 3 * depth_cpw)
    cpw = tf.layers.batch_normalization(cpw, momentum=0.9)  # , training=training)
    cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
    cpw = tf.reshape(cpw, (-1, dim_cpw, 3, depth_cpw))

    cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw / 2), kernel_size, strides=(2, 1), padding='same')
    cpw = tf.layers.batch_normalization(cpw, momentum=0.9)  # , training=training)
    cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
    #            cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)

    cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw / 4), kernel_size, strides=(2, 1), padding='same')
    cpw = tf.layers.batch_normalization(cpw, momentum=0.9)  # , training=training)
    cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
    #            cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)

    cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw / 8), kernel_size, strides=(2, 1), padding='same')
    cpw = tf.layers.batch_normalization(cpw, momentum=0.9)  # , training=training)
    cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
    #            cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)

    # Control points
    cp = tf.layers.conv2d(cpw, 1, (1, 2), padding='valid')  # batch_size x (bezier_degree+1) x 2 x 1
    cp = tf.nn.tanh(cp)
    cp = tf.squeeze(cp, axis=-1, name='control_point')  # batch_size x (bezier_degree+1) x 2
    # Weights
    w = tf.layers.conv2d(cpw, 1, (1, 3), padding='valid')
    w = tf.nn.sigmoid(w)  # batch_size x (bezier_degree+1) x 1 x 1
    w = tf.squeeze(w, axis=-1, name='weight')  # batch_size x (bezier_degree+1) x 1

    # Parameters at data points
    db = tf.layers.dense(cz, 1024)
    db = tf.layers.batch_normalization(db, momentum=0.9)  # , training=training)
    db = tf.nn.leaky_relu(db, alpha=0.2)

    db = tf.layers.dense(db, 256)
    db = tf.layers.batch_normalization(db, momentum=0.9)  # , training=training)
    db = tf.nn.leaky_relu(db, alpha=0.2)

    db = tf.layers.dense(db, 192 - 1)
    db = tf.nn.softmax(db)  # batch_size x (n_data_points-1)
    #            db = tf.random_gamma([tf.shape(cz)[0], self.X_shape[0]-1], alpha=100, beta=100)
    #            db = tf.nn.softmax(db) # batch_size x (n_data_points-1)

    ub = tf.pad(db, [[0, 0], [1, 0]], constant_values=0)  # batch_size x n_data_points
    ub = tf.cumsum(ub, axis=1)
    ub = tf.minimum(ub, 1)
    ub = tf.expand_dims(ub, axis=-1)  # 1 x n_data_points x 1

    # Bezier layer
    # Compute values of basis functions at data points
    num_control_points = 31 + 1
    lbs = tf.tile(ub, [1, 1, num_control_points])  # batch_size x n_data_points x n_control_points
    pw1 = tf.range(0, num_control_points, dtype=tf.float32)
    pw1 = tf.reshape(pw1, [1, 1, -1])  # 1 x 1 x n_control_points
    pw2 = tf.reverse(pw1, axis=[-1])

    lbs = tf.add(tf.multiply(pw1, tf.log(lbs + EPSILON)),
                 tf.multiply(pw2, tf.log(1 - lbs + EPSILON)))  # batch_size x n_data_points x n_control_points
    lc = tf.add(tf.lgamma(pw1 + 1), tf.lgamma(pw2 + 1))
    lc = tf.subtract(tf.lgamma(tf.cast(num_control_points, dtype=tf.float32)), lc)  # 1 x 1 x n_control_points
    lbs = tf.add(lbs, lc)  # batch_size x n_data_points x n_control_points
    bs = tf.exp(lbs)
    # Compute data points
    cp_w = tf.multiply(cp, w)
    dp = tf.matmul(bs, cp_w)  # batch_size x n_data_points x 2
    bs_w = tf.matmul(bs, w)  # batch_size x n_data_points x 1
    dp = tf.div(dp, bs_w)  # batch_size x n_data_points x 2
    dp = tf.expand_dims(dp, axis=-1, name='fake_image')  # batch_size x n_data_points x 2 x 1

    return dp, cp, w, ub, db

# input_c = tf.cast(tf.zeros(16, 3), dtype=tf.float32)float32
# input_n = tf.cast(tf.zeros(16, 10), dtype=tf.float32)
input_c = tf.placeholder(tf.float32, shape=[None, 3], name='latent_code')
input_n = tf.placeholder(tf.float32, shape=[None, 10], name='noise')
dp, cp, w, ub, db = generator(input_c, input_n, 3, 10)
# print(dp.shape, cp.shape, w.shape, ub.shape, db.shape)
# (?, 192, 2, 1) (?, 32, 2) (?, 32, 1) (?, 192, 1) (?, 191)


def discriminator(x, training = True):
    depth = 64
    dropout = 0.4
    kernel_size = (4, 2)
    # x: (?, 192, 2, 1)
    x = tf.layers.conv2d(x, depth * 1, kernel_size, strides=(2, 1), padding='same')
    x = tf.layers.batch_normalization(x, momentum=0.9)  # , training=training)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = tf.layers.dropout(x, dropout, training=training)

    x = tf.layers.conv2d(x, depth * 2, kernel_size, strides=(2, 1), padding='same')
    x = tf.layers.batch_normalization(x, momentum=0.9)  # , training=training)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = tf.layers.dropout(x, dropout, training=training)

    x = tf.layers.conv2d(x, depth * 4, kernel_size, strides=(2, 1), padding='same')
    x = tf.layers.batch_normalization(x, momentum=0.9)  # , training=training)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = tf.layers.dropout(x, dropout, training=training)

    x = tf.layers.conv2d(x, depth * 8, kernel_size, strides=(2, 1), padding='same')
    x = tf.layers.batch_normalization(x, momentum=0.9)  # , training=training)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = tf.layers.dropout(x, dropout, training=training)

    x = tf.layers.conv2d(x, depth * 16, kernel_size, strides=(2, 1), padding='same')
    x = tf.layers.batch_normalization(x, momentum=0.9)  # , training=training)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = tf.layers.dropout(x, dropout, training=training)

    x = tf.layers.conv2d(x, depth * 32, kernel_size, strides=(2, 1), padding='same')
    x = tf.layers.batch_normalization(x, momentum=0.9)  # , training=training)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = tf.layers.dropout(x, dropout, training=training)
    # (?, 3, 2, 2048)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 1024)
    x = tf.layers.batch_normalization(x, momentum=0.9)  # , training=training)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    print(x.shape)

    d = tf.layers.dense(x, 1)

    q = tf.layers.dense(x, 128)
    q = tf.layers.batch_normalization(q, momentum=0.9)  # , training=training)
    q = tf.nn.leaky_relu(q, alpha=0.2)
    q_mean = tf.layers.dense(q, 3)
    q_logstd = tf.layers.dense(q, 3)
    q_logstd = tf.maximum(q_logstd, -16)
    # Reshape to batch_size x 1 x latent_dim
    q_mean = tf.reshape(q_mean, (-1, 1, 3))
    q_logstd = tf.reshape(q_logstd, (-1, 1, 3))
    q = tf.concat([q_mean, q_logstd], axis=1, name='predicted_latent')  # batch_size x 2 x latent_dim
    return d, q

d, q = discriminator(dp)
print(d.shape, q.shape)