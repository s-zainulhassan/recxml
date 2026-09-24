import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from src.feature_fusion import FeatureFusionBlock

class StudentGenerator(Model):
    """Student generator. Feature-fusion implementation is private."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.student_hidden_dim = config.hidden_dim // 2
        self.student_embedding_dim = config.embedding_dim // 2
        self.student_fusion_dim = self.student_hidden_dim // 2

        self.doc_dense1 = Dense(self.student_hidden_dim, kernel_regularizer=l2(config.l2_weight))
        self.doc_act1 = LeakyReLU(negative_slope=0.2)
        self.doc_dropout1 = Dropout(config.dropout_rate)
        self.doc_embedding = Dense(self.student_embedding_dim, kernel_regularizer=l2(config.l2_weight))
        self.doc_act2 = LeakyReLU(negative_slope=0.2)

        self.noise_dense = Dense(self.student_embedding_dim, kernel_regularizer=l2(config.l2_weight))
        self.noise_act = LeakyReLU(negative_slope=0.2)

        self.fusion = FeatureFusionBlock(self.student_fusion_dim, name="private_student_fusion")
        self.output_layer = Dense(config.label_dim, activation="sigmoid")

        self.feature_projection = Dense(
            config.embedding_dim, use_bias=False,
            kernel_regularizer=l2(config.l2_weight),
            name="feature_projection"
        )
        self.embedding = None
        self.projected_embedding = None

    def call(self, documents, noise, training=False):
        x_doc = self.doc_act1(self.doc_dense1(documents))
        x_doc = self.doc_dropout1(x_doc, training=training)
        x_doc = self.doc_act2(self.doc_embedding(x_doc))
        self.embedding = x_doc
        self.projected_embedding = self.feature_projection(x_doc)

        x_noise = self.noise_act(self.noise_dense(noise))
        x = tf.concat([x_doc, x_noise], axis=-1)
        x = self.fusion(x, training=training)
        return self.output_layer(x)
