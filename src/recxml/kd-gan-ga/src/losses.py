import tensorflow as tf

bce_loss = tf.keras.losses.BinaryCrossentropy()
mse_loss = tf.keras.losses.MeanSquaredError()
kl_loss = tf.keras.losses.KLDivergence()

def generator_classification_loss(real_labels, predicted_labels):
    return bce_loss(real_labels, predicted_labels)

def generator_adversarial_loss(fake_scores):
    return -tf.reduce_mean(fake_scores)

def critic_loss(real_scores, fake_scores):
    return tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)

def student_classification_loss(real_labels, student_predictions):
    return bce_loss(real_labels, student_predictions)

def output_distillation_loss(teacher_predictions, student_predictions, temperature):
    eps = 1e-7
    teacher_predictions = tf.clip_by_value(teacher_predictions, eps, 1.0 - eps)
    student_predictions = tf.clip_by_value(student_predictions, eps, 1.0 - eps)
    teacher_logits = tf.math.log(teacher_predictions / (1.0 - teacher_predictions))
    student_logits = tf.math.log(student_predictions / (1.0 - student_predictions))
    teacher_soft = tf.nn.sigmoid(teacher_logits / temperature)
    student_soft = tf.nn.sigmoid(student_logits / temperature)
    return (temperature ** 2) * kl_loss(teacher_soft, student_soft)

def feature_distillation_loss(teacher_embedding, student_embedding):
    return mse_loss(teacher_embedding, student_embedding)

def student_adversarial_loss(critic_scores):
    return -tf.reduce_mean(critic_scores)

def student_total_loss(
    real_labels, teacher_predictions, student_predictions,
    teacher_embedding, student_embedding, critic_scores, config
):
    hard_loss = student_classification_loss(real_labels, student_predictions)
    output_loss = output_distillation_loss(
        teacher_predictions, student_predictions, config.distillation_temperature
    )
    feature_loss = feature_distillation_loss(teacher_embedding, student_embedding)
    adv_loss = student_adversarial_loss(critic_scores)
    total_loss = (
        config.lambda_hard * hard_loss
        + config.lambda_output * output_loss
        + config.lambda_feature * feature_loss
        + config.lambda_adv * adv_loss
    )
    return total_loss, hard_loss, output_loss, feature_loss, adv_loss
