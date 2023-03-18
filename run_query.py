import pandas as pd
import numpy as np
import csv
import time
from global_variables import sigma, clip, delta, training_size

from data_utils import (
    get_embeddings, 
    synthesize_database, 
    gaussian_noise_to_embeddings,
)

def create_image_dataset():
    embed_original, indices = get_embeddings()
    embed_data, id_data, date_data, location_data = synthesize_database(embed_original)
    list_columns = ["ImageID","Date","Location"] 

    with open('surveillence_img.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow (list_columns)
        for i in range(len(embed_original)):
            writer.writerow([id_data[i], date_data[i], location_data[i]])
        csvfile.close()
        

def query_location(name, datebirth, country, is_noisy):
    """
    Note: 
    For simplicity, we assume that the passenger embeddings are the same as the suveillence image embeddings.
    Not only so, we assume that the order is also the same.
    When using noisy date, the np.allclose hardly output the correct result.
    Also note that the query time depends on the position of the queried embedding.
    """
    # For fair comparison, we use size of the training size
    passenger = pd.read_csv("dataset/passenger_data/passenger.csv", nrows=training_size)
    image = pd.read_csv("dataset/surveillence_img.csv", nrows=training_size)
    embed_original, indices = get_embeddings()
    noisy_embed_data, epsilon = gaussian_noise_to_embeddings(np.stack(embed_original), sigma, clip, delta)
    
    example_query = f"""
        SELECT img.location FROM surveillence_img img JOIN passengers ON np.allclose(embeddings(photoes[passengers.id]),embeddings(faces[img.id])) == True 
        WHERE passengers.name Like '\%{name}\%' AND datebirth=='{datebirth}' AND country =='{country}'
    """
    print(f"Query is:\n{example_query}")

    begin = time.time()
    img_id = passenger.query('name==@name and dob==@datebirth and country==@country').iloc[0]['id']
    location_ground_truth = image.iloc[img_id]["Location"]
    #print(img_id)
    #print(len(embed_original))
    query_embedding = embed_original[img_id]
    #print (query_embedding)
    location = -1
    embed_to_scan = noisy_embed_data if is_noisy else embed_original
    for i in range (training_size):
        if (np.allclose(embed_to_scan[i], query_embedding)):
        # if (np.allclose(embed_to_scan[i], query_embedding, atol=0.3)):
            location = image.loc[i]["Location"]
            break
    end = time.time()
    print(f"Ground truth location: {location_ground_truth}, query output: {location}")
    print("Elapsed time for query based on embedding matching:", end-begin, " seconds")
    return location

def query_location_simple(name, datebirth, country):
    passenger = pd.read_csv("dataset/passenger_data/passenger.csv", nrows=training_size)
    image = pd.read_csv("dataset/surveillence_img.csv", nrows=training_size)
    example_query = f"""
        SELECT img.location FROM surveillence_img img JOIN passengers ON img.ImageID = passengers.ID
        WHERE passengers.name Like '\%{name}\%' AND datebirth=='{datebirth}' AND country =='{country}'
    """
    print(f"Query is:\n{example_query}")
    begin = time.time()
    location = pd.merge(image, passenger, left_on=['ImageID'], right_on=['id'], how='inner').query('name==@name and dob==@datebirth and country==@country').iloc[0]['Location']
    end = time.time()
    print("Elapsed time for query based on ID matching:", end-begin, " seconds")
    return location



if __name__ == '__main__':

    #create_image_dataset()

    name_query = 'Alice Caine'
    datebirth = '1996-11-13'
    country = 'UK'
    picture_id = 14 # we can link picture to its id.

    # name_query = 'Peter Derr'
    # datebirth = '1982-06-05'
    # country = 'UK'
    # picture_id = 18 # we can link picture to its id.

    location = query_location(name_query, datebirth, country, is_noisy=False)
    # print("Location queried through embedding matching:", location)
    print("\n\n")
    location1 = query_location_simple(name_query, datebirth, country)
    # print("Location queried through ID matching:", location1)

