import pickle
import random
from sklearn.preprocessing import StandardScaler
from typings import (Embedding, Embeddings)
from typing import Union
from global_variables import frequency_range, date_range, location_range, training_size, countries
import pandas as pd
import numpy as np
import os
from datetime import date, timedelta
import math

# Set random seed
SEED = 1433 
random.seed(SEED)


# Functions
def get_passenger_database(
    embed_original: Union[list[Embedding], Embeddings],
    data_path: str = 'dataset/passenger_data/passenger.csv',
    names_path: str = 'dataset/names/query-names.txt',
) -> pd.DataFrame:
    if os.path.exists(data_path):
        return pd.read_csv(data_path)

    with open(names_path, 'r') as f:
        names = [name.strip() for name in f.readlines()]

    test_date1, test_date2 = date(1980, 1, 1), date(2000, 1, 1)
    dates_bet = test_date2 - test_date1
    total_days = dates_bet.days

    rows = []
    # Synthesize passenger database
    for index in range(len(embed_original)):
        row = {"id": index}
        row["name"] = names[index] # assume names are distinct
        row["dob"] = str(test_date1 + timedelta(days=random.randrange(total_days)))  
        row["country"] = random.choice(countries)
        # row["embed"] = embed_original[index]
        rows.append(row)
    database = pd.DataFrame(rows)
    database.to_csv(data_path, index=False)
    print(database)
    return database

def check_passenger_exist(
    passenger_data,
    name_query,
    datebirth,
    country,
    picture_id,
):
    for i, row in passenger_data.iterrows():
        if row['id'] == picture_id and row["name"] == name_query and row['dob'] == datebirth and row['country'] == country:
            return i
    return -1

def get_embeddings(
    path: str = 'dataset/embeddings/10K_encodings.pkl', 
) -> tuple[list[Embedding], list[int]]:
    embeddings = pickle.load(open(path, 'rb'))
    size = training_size // frequency_range
    if size > len(embeddings):
        raise ValueError("Size exceed the embeddings size!")
    if size == len(embeddings):
        return embeddings

    indices = random.sample(range(len(embeddings)), size)
    embed_original = [embeddings[i] for i in indices]
    return (embed_original, indices)

def transform_embedding(embedding: Embedding) -> Embedding:
    return embedding

def synthesize_database(
    embed_original: list[Embedding],
    is_transform: bool = True,  
    is_fix_freq: bool = True,
) -> tuple[list[Embedding], list[int], list[int], list[int]]:
    
    if is_fix_freq:
        embed_data, id_data = [], []
        date_data, location_data = [], []
        for i, embedding in enumerate(embed_original):
            embed_data.extend([embedding] * frequency_range)
            id_data.extend([i] * frequency_range)
            dates = random.sample(range(date_range), frequency_range)
            date_data.extend([date + 1 for date in dates])
            locations = random.sample(range(location_range), frequency_range)
            location_data.extend([location + 1 for location in locations])
    else:
        if is_transform:
            embed_data, id_data = [], []
            date_data, location_data = [], []
            for i, embedding in enumerate(embed_original):
                for _ in range(random.randint(1, frequency_range)):
                    embed_data.append(transform_embedding(embedding))
                    id_data.append(i)
                    date_data.append(random.randint(1, date_range))
                    location_data.append(random.randint(1, location_range))
            
        else:
            embed_data = embed_original
            length = len(embed_data)
            id_data = [i for i in range(length)]
            date_data = [random.randint(1, date_range) for _ in range(length)]
            location_data = [random.randint(1, location_range) for _ in range(length)]

    return (embed_data, id_data, date_data, location_data)


def standardize_input(X_train: Embeddings) -> tuple[Embeddings, StandardScaler]:
    sc = StandardScaler()
    sc.fit(X_train)
    return (sc.transform(X_train), sc)

def create_multi_output_trainset(
    embed_data: Embeddings, 
    id_data: list[int], 
    date_data: list[int], 
    location_data: list[int],
) -> tuple[Embeddings, np.array, np.array]:
    # Need rewrite
    # labels = np.zeros((len(id_data), date_range * location_range), dtype=np.int32)
    # for id, date, location in zip(id_data, date_data, location_data):
    #     labels[id, date * location - 1] = 1
    # X_train, sc = standardize_input(embed_data)
    # return (X_train, np.array(id_data), labels, sc)
    length = id_data[-1] + 1
    labels = np.zeros((length, date_range * location_range), dtype=np.int32)

    prev_id = -1
    embed_data_dedup = []
    for id, date, location, embed in zip(id_data, date_data, location_data, embed_data):
        labels[id, date * location - 1] = 1
        if id != prev_id:
            embed_data_dedup.append(embed)
            prev_id = id
    assert len(embed_data_dedup) == length
    X_train, sc = standardize_input(np.stack(embed_data_dedup))
    return (X_train, np.arange(length), labels, sc)

def create_multi_input_trainset(    
    embed_data: Embeddings, 
    id_data: list[int], 
    date_data: list[int], 
    location_data: list[int],
    is_onehot_encode: bool = True,
):
    # Create labels
    location_data = np.array(location_data)
    labels = np.zeros((len(location_data), location_range), dtype=np.int32)
    labels[np.arange(len(location_data)), location_data - 1] = 1

    # Create X_train
    date_data = np.array(date_data)
    if is_onehot_encode:
        date_data_array = np.zeros((len(date_data), date_range), dtype=np.int32)
        date_data_array[np.arange(len(date_data)), date_data - 1] = 1
    else:
        date_data_array = np.expand_dims(date_data, axis=1)
    X_train = np.concatenate((embed_data, date_data_array), axis=1)
    X_train, sc = standardize_input(X_train)
    return (X_train, np.array(id_data), labels, sc)


def synthesize_simple_database(embed_original: list[Embedding]):
    assert frequency_range == 1
    location_data = [random.randint(1, location_range) for _ in range(len(embed_original))]
    id_data = list(range(len(location_data)))
    return (embed_original, id_data, location_data)

def create_simple_trainset(    
    embed_data: Embeddings, 
    id_data: list[int], 
    label_list: list[int],
) -> tuple[Embeddings, np.array, np.array]:
    label_array = np.array(label_list)
    labels = np.zeros((len(label_list), len(np.unique(label_array))), dtype=np.int32)
    labels[np.arange(len(label_list)), label_array - 1] = 1

    X_train, sc = standardize_input(embed_data)
    return (X_train, np.array(id_data), labels, sc)


def compute_privacy_budget(dims, clip, delta, sigma):
    l2_sensitivity = math.sqrt(dims * (2 * abs(clip)) ** 2)
    epsilon = math.sqrt(2 * math.log(1.25 / delta)) * l2_sensitivity / sigma
    return epsilon

def gaussian_noise_to_embeddings(
    embeddings: Embeddings,
    sigma: float,
    clip: float,
    delta: float,
) -> tuple[Embeddings, float]:
    # Add gaussian noise
    noisy_embeds = np.clip(embeddings, -abs(clip), abs(clip))
    gaussian_noise = sigma * np.random.randn(*embeddings.shape)
    noisy_embeds += gaussian_noise

    epsilon = compute_privacy_budget(embeddings.shape[1], clip, delta, sigma)
    return (noisy_embeds, epsilon)