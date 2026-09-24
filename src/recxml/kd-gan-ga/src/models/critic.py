import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from src.feature_fusion import FeatureFusionBlock

class Critic(Model):
    """Wasserstein critic. Feature-fusion implementation is private."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.doc_dense1 = Dense(config.critic_hidden_dim, kernel_regularizer=l2(config.l2_weight))
        self.doc_act1 = LeakyReLU(negative_slope=0.2)
        self.doc_dropout1 = Dropout(config.critic_dropout)
        self.doc_embedding = Dense(config.critic_embedding_dim, kernel_regularizer=l2(config.l2_weight))
        self.doc_act2 = LeakyReLU(negative_slope=0.2)
        self.label_dense = Dense(config.critic_embedding_dim, kernel_regularizer=l2(config.l2_weight))
        self.label_act = LeakyReLU(negative_slope=0.2)
        self.fusion = FeatureFusionBlock(config.critic_hidden_dim // 2, name="private_critic_fusion")
        self.score_layer = Dense(1)

    def call(self, documents, labels, training=False):
        x_doc = self.doc_act1(self.doc_dense1(documents))
        x_doc = self.doc_dropout1(x_doc, training=training)
        x_doc = self.doc_act2(self.doc_embedding(x_doc))
        x_label = self.label_act(self.label_dense(labels))
        x = tf.concat([x_doc, x_label], axis=-1)
        x = self.fusion(x, training=training)
        return self.score_layer(x)
