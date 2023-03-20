import tensorflow as tf
from modeling import MySequentialModel, CrossProductOutputModel
from global_variables import (
    date_range, 
    location_range, 
    training_size, 
    sigma, clip, delta,
)
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from data_utils import (
    create_multi_output_trainset,
    create_multi_input_trainset,
    create_simple_trainset,
    compute_privacy_budget,
)
from typings import *
from global_variables import date_range, location_range, batch_size, noise_multiplier, epochs
from sklearn.preprocessing import StandardScaler

def _train(
    X_train, 
    y_train,
    out_size: int,
    loss_func: str = 'CategoricalCrossentropy',
    num_microbatches: int = 1,
    is_privacy_preserve: bool = True,
    **kwargs
):
    learning_rate = kwargs['learning_rate']
    epochs = kwargs['epochs']
    l2_norm_clip = kwargs['l2_norm_clip']
    noise_multiplier = kwargs['noise_multiplier']
    batch_size = kwargs['batch_size']
    delta = kwargs['delta']
    if batch_size % num_microbatches != 0:
        raise ValueError('Batch size should be an integer multiple of the number of microbatches')
    
    if is_privacy_preserve:
        optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Define model and loss function
    if loss_func == 'CategoricalCrossentropy':
        model = MySequentialModel(out_size=out_size)
        loss = tf.keras.losses.CategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    elif loss_func == 'BinaryCrossentropy':
        model = CrossProductOutputModel(out_size=out_size)
        loss = tf.keras.losses.BinaryCrossentropy()
        model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy()])
    else:
        raise ValueError

    
    # Fit model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    print(f'n={X_train.shape[0]}, batch_size={batch_size}')
    eps = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=X_train.shape[0],
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=epochs,
        delta=delta)
    
    return (model, eps)


def _train_multi_output_model(embed_data, id_data, date_data, location_data, is_privacy_preserve, **kwargs):
    X_train, ids, y_train, sc = create_multi_output_trainset(embed_data, id_data, date_data, location_data)
    print(f'X_train: {X_train.shape}, labels: {y_train.shape}')
    model, eps = _train(
        X_train, y_train, 
        out_size=location_range*date_range, 
        loss_func='BinaryCrossentropy',
        is_privacy_preserve=is_privacy_preserve,
        **kwargs)
    return (model, eps, X_train, y_train, sc)

def _train_multi_input_model(embed_data, id_data, date_data, location_data, is_privacy_preserve, **kwargs):
    X_train, ids, y_train, sc =  create_multi_input_trainset(embed_data, id_data, date_data, location_data)
    print(f'X_train: {X_train.shape}, labels: {y_train.shape}')
    model, eps =  _train(
        X_train, y_train, 
        out_size=location_range, 
        loss_func='CategoricalCrossentropy',
        is_privacy_preserve=is_privacy_preserve,
        **kwargs)
    return (model, eps, X_train, y_train, sc)

def _train_simple(
    embed_data: Embeddings, 
    id_data: list[int], 
    location_data: list[int], 
    is_privacy_preserve: bool,
    out_size: int,
    **kwargs
) -> tuple[TFModel, tuple[int], Embeddings, np.array, StandardScaler]:
    X_train, ids, y_train, sc = create_simple_trainset(embed_data, id_data, location_data)
    print(f'X_train: {X_train.shape}, labels: {y_train.shape}')
    model, eps =  _train(
        X_train, y_train, 
        out_size=out_size,
        loss_func='CategoricalCrossentropy',
        is_privacy_preserve=is_privacy_preserve,
        **kwargs)
    return (model, eps, X_train, y_train, sc)

def train_model(embed_data, id_data, date_data, location_data, user_selection, is_privacy_preserve, out_size=location_range, **kwargs):
    if date_data is None: # simple_data
        return _train_simple(embed_data, id_data, location_data, is_privacy_preserve, out_size, **kwargs)
    else:
        if user_selection == 'multi-output':
            return _train_multi_output_model(embed_data, id_data, date_data, location_data, is_privacy_preserve, **kwargs)
        else:
            return _train_multi_input_model(embed_data, id_data, date_data, location_data,is_privacy_preserve, **kwargs)


def estimate_cost(embed_dim: int = 128) -> tuple[dict]:
    eps1 = compute_privacy_budget(embed_dim, clip, delta, sigma) 
    eps2 = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=training_size,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=epochs,
        delta=delta)
    
    estimated_acc1 = 0.94
    estimated_acc2 = 0.99
    return (
        {"eps": eps1, "acc": estimated_acc1},
        {"eps": eps2[0], "acc": estimated_acc2})
