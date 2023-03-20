from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd
import numpy as np
import torch

from time import time
from pathlib import Path
from PIL import Image

from data_utils import (
    get_passenger_database,
    check_passenger_exist,
    gaussian_noise_to_embeddings,
    create_simple_trainset
)
from train import train_model, estimate_cost
from test import evaluate
from global_variables import (
    sigma, 
    clip, 
    delta, 
    learning_rate,
    epochs,
    l2_norm_clip,
    noise_multiplier,
    batch_size,
    delta,
    training_size
)

def create_image_gender_dataset():
    save_dir = Path('face_database/processed_data/')
    save_dir.mkdir(exist_ok=True)

    # Get and save image-gender table
    df = pd.read_excel('face_database/demographic-others-labels.xlsx', sheet_name='Final Values', usecols=['Filename', 'Gender'])
    df.to_csv(save_dir.joinpath('image_gender.csv'), index=False)

    # Extract embeddings
    mtcnn = MTCNN()
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    imgs_cropped = []
    for img_path in Path('face_database/2222annotated_faces/').iterdir():
        img = Image.open(img_path)
        img_cropped = mtcnn(img)
        imgs_cropped.append(img_cropped)

    imgs_cropped = torch.stack(imgs_cropped)
    img_embeddings = resnet(imgs_cropped).detach().numpy()
    np.save(save_dir.joinpath('embeddings'), img_embeddings)
    print(f'Created image embeddings, shape is {img_embeddings.shape}')


def main(
    plan_number: str = None,
    target_dim: int = 2,
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
        SELECT img.gender FROM virtual_surveillance_imgs img JOIN passengers ON match (passengers.pic, img) = True 
        WHERE passengers.name Like '\%{name_query}\%' AND datebirth='{datebirth}' AND country ='{country}'
    """
    print(f"Example query:\n{example_query}")
    
    # Airport dataset
    embed_data  = np.load('face_database/processed_data/embeddings.npy')[:training_size]
    gender_data = pd.read_csv('face_database/processed_data/image_gender.csv')['Gender'].to_list()[:training_size]
    assert len(gender_data) == embed_data.shape[0]
    id_data = list(range(len(gender_data)))
    date_data = None
    noisy_embed_data, epsilon = gaussian_noise_to_embeddings(embed_data, sigma, clip, delta)

    # Passenger database
    passenger_data = get_passenger_database(embed_data)
    begin = time()
    query_image_index = check_passenger_exist(passenger_data, name_query, datebirth, country, picture_id)
    latency += time() - begin
    if query_image_index == -1:
        print(message.format(latency))
        raise ValueError("Passenger doesn't exist in the database")
    else:
        print(f"query_image_index: {query_image_index}")
    
    # Get query embedding and its ground truth label
    if query_image_index >= len(embed_data):
        raise ValueError("query image index exceed the total number of pictures") 
    print("Picture:") # Show picture
    embed_query = embed_data[query_image_index]
    print(f"Embedding:\n {embed_query}")

    truth_label = gender_data[query_image_index]

    # Estimate cost
    costs = estimate_cost(embed_dim=512)

    print(f"Plan 1: eps: {costs[0]['eps']}, estimated accuracy: {costs[0]['acc']}")
    print(f"Plan 2: eps: {costs[1]['eps']}, estimated accuracy: {costs[1]['acc']}")

    # Select a plan and train model, hard coded
    plan_selection = input("Select a plan:") if plan_number is None else plan_number
    assert plan_selection == '1' or plan_selection == '2'
    embed_input = noisy_embed_data if plan_selection == '1' else embed_data
    is_privacy_preserve = False if  plan_selection == '1' else True
    user_selection = 'single-output'

    model, eps, X_train, y_train, scaler = train_model(
        embed_input, id_data, date_data, gender_data, 
        user_selection=user_selection, 
        is_privacy_preserve=is_privacy_preserve,
        out_size=target_dim,
        **kwargs)

    # Evaluate model
    print('\n\nEvaluation:')
    
    if plan_selection == '1':
        X_eval, ids, y_train, scaler = create_simple_trainset(embed_data, id_data, gender_data)
        loss, acc = evaluate(model, X_eval, y_train)
        print(f"[Evaluation] eps: {epsilon:.2f}, acc: {acc:.2f}, loss: {loss:.2f}")
    else:
        loss, acc = evaluate(model, X_train, y_train)
        print(f"[Evaluation] eps: {eps[0]:.2f}, acc: {acc:.2f}, optimal RDP order: {eps[1]}, loss: {loss:.2f}")


    # Get prediction (location), hard coded

    begin = time()
    location_pred = model.predict(scaler.transform(np.expand_dims(embed_query, axis=0)))
    location_pred = np.argmax(location_pred, axis=1)
    location_pred += 1
    latency += time() - begin
    print(message.format(latency))
    print(f'Predicted location is: {location_pred}')
    print(f'Ground truth location is: {truth_label}')
    return (epsilon, acc) if plan_selection == '1' else (eps[0], acc)

if __name__ == '__main__':
    # create_image_gender_dataset()
    main()
    
    
