import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from src.feature_fusion import FeatureFusionBlock

class Generator(Model):
    """Teacher generator. Feature-fusion implementation is private."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.doc_dense1 = Dense(config.hidden_dim, kernel_regularizer=l2(config.l2_weight))
        self.doc_act1 = LeakyReLU(negative_slope=0.2)
        self.doc_dropout1 = Dropout(config.dropout_rate)
        self.doc_embedding = Dense(config.embedding_dim, kernel_regularizer=l2(config.l2_weight))
        self.doc_act2 = LeakyReLU(negative_slope=0.2)
        self.noise_dense = Dense(config.embedding_dim, kernel_regularizer=l2(config.l2_weight))
        self.noise_act = LeakyReLU(negative_slope=0.2)
        self.fusion = FeatureFusionBlock(config.hidden_dim, name="private_teacher_fusion")
        self.output_layer = Dense(config.label_dim, activation="sigmoid")
        self.embedding = None

    def call(self, documents, noise, training=False):
        x_doc = self.doc_act1(self.doc_dense1(documents))
        x_doc = self.doc_dropout1(x_doc, training=training)
        x_doc = self.doc_act2(self.doc_embedding(x_doc))
        self.embedding = x_doc
        x_noise = self.noise_act(self.noise_dense(noise))
        x = tf.concat([x_doc, x_noise], axis=-1)
        x = self.fusion(x, training=training)
        return self.output_layer(x)
