date_range = 365
location_range = 20
countries = ['UK', 'US', 'BRZ', 'CHN', 'AU']

hidden_units = 500

frequency_range = 1
training_size = 1000

learning_rate = 0.2
batch_size = 200
epochs = 70

l2_norm_clip = 1
noise_multiplier = 0.03

sigma = 0.1
clip = 0.5
delta = 1e-5

assert training_size % frequency_range == 0
assert frequency_range < location_range
assert frequency_range < date_range
