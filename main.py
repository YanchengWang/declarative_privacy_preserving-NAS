
from time import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from data_utils import (
    get_embeddings, 
    synthesize_database, 
    synthesize_simple_database,
    get_passenger_database,
    check_passenger_exist,
    gaussian_noise_to_embeddings,
    create_simple_trainset
)
from train import train_model, estimate_cost
from test import evaluate
from global_variables import (
    date_range, 
    frequency_range, 
    sigma, 
    clip, 
    delta, 
    learning_rate,
    epochs,
    l2_norm_clip,
    noise_multiplier,
    batch_size,
    delta,
)


def main(
    is_simple_data: bool = False,
    plan_number: str = None,
    **kwargs,
) -> tuple[float, float]:
    if len(kwargs) == 0:
        kwargs['learning_rate'] = learning_rate
        kwargs['epochs']= epochs
        kwargs['l2_norm_clip'] = l2_norm_clip
        kwargs['noise_multiplier'] = noise_multiplier
        kwargs['batch_size'] = batch_size
        kwargs['delta'] = delta
    latency = 0.0
    message = "Elapsed time for query based on privacy preserving is {} seconds"

    # Query
    name_query = 'Alice Caine'
    datebirth = '1996-11-13'
    country = 'UK'
    picture_id = 14 # we can link picture to its id.

    # name_query = 'Peter Derr'
    # datebirth = '1982-06-05'
    # country = 'UK'
    # picture_id = 18 # we can link picture to its id.
 
    example_query = f"""
        SELECT img.location FROM virtual_surveillance_imgs img JOIN passengers ON match (passengers.pic, img) = True 
        WHERE passengers.name Like '\%{name_query}\%' AND datebirth='{datebirth}' AND country ='{country}'
    """
    print(f"Example query:\n{example_query}")
    
    # Airport dataset
    embed_original, indices = get_embeddings() # Considered the query pictures
    if is_simple_data:
        embed_data, id_data, location_data = synthesize_simple_database(embed_original)
        date_data = None
    else:
        embed_data, id_data, date_data, location_data = synthesize_database(embed_original)

    embed_data = np.stack(embed_data)
    noisy_embed_data, epsilon = gaussian_noise_to_embeddings(embed_data, sigma, clip, delta)

    # Passenger database
    passenger_data = get_passenger_database(embed_original)
    begin = time()
    query_image_index = check_passenger_exist(passenger_data, name_query, datebirth, country, picture_id)
    latency += time() - begin
    if query_image_index == -1:
        print(message.format(latency))
        raise ValueError("Passenger doesn't exist in the database")
    else:
        print(f"query_image_index: {query_image_index}")
    
    # Get query embedding and its ground truth label
    if query_image_index >= len(embed_original):
        raise ValueError("query image index exceed the total number of pictures") 
    print("Picture:") # Show picture
    embed_query = embed_original[query_image_index]
    print(f"Embedding:\n {embed_query}")

    if is_simple_data:
        truth_label = location_data[query_image_index]
    else:
        query_date_index = int(input(f'Select date (<{frequency_range}):')) # this is actually the index of the date for the same person
        if query_date_index >= frequency_range:
            raise ValueError(f"date exceed the total number of dates: {frequency_range}") 
        label_index = query_image_index * frequency_range + query_date_index
        truth_label = location_data[label_index]

    # Estimate cost
    costs = estimate_cost()

    print(f"Plan 1: eps: {costs[0]['eps']}, estimated accuracy: {costs[0]['acc']}")
    print(f"Plan 2: eps: {costs[1]['eps']}, estimated accuracy: {costs[1]['acc']}")

    # Select a plan and train model, hard coded
    plan_selection = input("Select a plan:") if plan_number is None else plan_number
    assert plan_selection == '1' or plan_selection == '2'
    embed_input = noisy_embed_data if is_simple_data and plan_selection == '1' else embed_data
    is_privacy_preserve = False if is_simple_data and plan_selection == '1' else True
    user_selection = 'multi-output' if plan_selection == '1' and not is_simple_data else 'single-output'

    model, eps, X_train, y_train, scaler = train_model(
        embed_input, id_data, date_data, location_data, 
        user_selection=user_selection, 
        is_privacy_preserve=is_privacy_preserve,
        **kwargs)

    # Evaluate model
    print('\n\nEvaluation:')
    
    if is_simple_data and plan_selection == '1':
        X_eval, ids, y_train, scaler = create_simple_trainset(embed_data, id_data, location_data)
        loss, acc = evaluate(model, X_eval, y_train)
        print(f"[Evaluation] eps: {epsilon:.2f}, acc: {acc:.2f}, loss: {loss:.2f}")
    else:
        loss, acc = evaluate(model, X_train, y_train)
        print(f"[Evaluation] eps: {eps[0]:.2f}, acc: {acc:.2f}, optimal RDP order: {eps[1]}, loss: {loss:.2f}")


    # Get prediction (location), hard coded
    if is_simple_data:
        begin = time()
        location_pred = model.predict(scaler.transform(np.expand_dims(embed_query, axis=0)))
        location_pred = np.argmax(location_pred, axis=1)
    else:
        if plan_selection == '1':
            input_ = scaler.transform(np.expand_dims(embed_original[query_image_index], axis=0))
            begin = time()
            location_pred = np.argmax(model.predict(input_), axis=1)
            location_pred %= location_pred
        elif plan_selection == '2':
            date_encode = np.zeros(date_range, dtype=np.int32)
            date_encode[date_data[label_index] - 1] = 1
            input_ = np.concatenate((embed_query, date_encode), axis=0)
            input_ = scaler.transform(np.expand_dims(input_, axis=0))
            begin = time()
            location_pred = np.argmax(model.predict(input_), axis=1)

    location_pred += 1
    latency += time() - begin
    print(message.format(latency))
    print(f'Predicted location is: {location_pred}')
    print(f'Ground truth location is: {truth_label}')
    return (epsilon, acc) if plan_selection == '1' else (eps[0], acc)
    
def noisy_data_experiment():
    save_dir = Path("./experiment_results/noisy_data")
    save_dir.mkdir(parents=True, exist_ok=True)
    sigmas = [0.1, 0.05, 0.01]
    clips = [0.6, 0.5, 0.4]
    deltas = [1e-4, 1e-5, 1e-6]
    global sigma, clip, delta
    
    results = []
    for s, c, d in product(sigmas, clips, deltas):
        result = {'sigma': s, 'clip': c, 'delta': d}
        sigma, clip, delta = s, c, d
        epsilon, acc = main(is_simple_data=True, plan_number='1')
        result['epsilon'] = epsilon
        result['acc'] = acc
        results.append(result)
    pd.DataFrame(results).to_csv(save_dir.joinpath("results.csv"), index=False)

    
def noisy_model_experiment():
    save_dir = Path("./experiment_results/noisy_model")
    save_dir.mkdir(parents=True, exist_ok=True)

    learning_rates = [0.1, 0.2, 0.5]
    batch_sizes = [50, 100, 200]
    epochss = [60, 70, 80]
    l2_norm_clips = [0.5, 1, 1.5]
    noise_multipliers = [0.1, 0.05, 0.03]
    deltas = [1e-4, 1e-5, 1e-6]
    # learning_rates = [0.1]
    # batch_sizes = [100, 200]
    # epochss = [60]
    # l2_norm_clips = [0.5, 1, 1.5]
    # noise_multipliers = [0.03]
    # deltas = [1e-5]

    results = []
    for lr, b, e, l, n, d in product(learning_rates, batch_sizes, epochss, l2_norm_clips, noise_multipliers, deltas):
        result = {
            'learning_rate': lr,
            'batch_size': b,
            'epochs' : e,
            'l2_norm_clip': l,
            'noise_multiplier': n,
            'delta': d,
        }
        epsilon, acc = main(is_simple_data=True, plan_number='2', **result)
        result['epsilon'] = epsilon
        result['acc'] = acc
        results.append(result)
    pd.DataFrame(results).to_csv(save_dir.joinpath("results.csv"), index=False)

def find_pareto_frontier(noisy_type: str):
    assert noisy_type == 'noisy_data' or noisy_type == 'noisy_model'
    df = pd.read_csv(f'./experiment_results/{noisy_type}/results.csv')
    frontier_indices = []
    for i, row in df.iterrows():
        if not any((row['epsilon'] >= df.iloc[j]["epsilon"] and row['acc'] < df.iloc[j]['acc'] or
                    row['epsilon'] > df.iloc[j]["epsilon"] and row['acc'] <= df.iloc[j]['acc'] for j in range(len(df)))):
            frontier_indices.append(i)
    df.iloc[frontier_indices].to_csv(f'./experiment_results/{noisy_type}/frontiers.csv')


if __name__ == '__main__':
    main(is_simple_data=True)

    # noisy_data_experiment()
    # find_pareto_frontier('noisy_data')

    # noisy_model_experiment()
    # find_pareto_frontier('noisy_model')
