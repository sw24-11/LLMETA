import requests

# For Text Analysis
response = requests.post('http://127.0.0.1:5000/analyze', json={
    'data_type': 'text',
    'text': """
            Autoencoders
            Dor Bank, Noam Koenigstein, Raja Giryes
            Abstract An autoencoder is a specific type of a neural network, which is mainly
            designed to encode the input into a compressed and meaningful representation, and
            then decode it back such that the reconstructed input is similar as possible to the
            original one. This chapter surveys the different types of autoencoders that are mainly
            used today. It also describes various applications and use-cases of autoencoders.
    """
})
print(response.json())

# For Image Analysis
response = requests.post('http://127.0.0.1:5000/analyze', json={
    'data_type': 'image',
    'image_path': 'uploads/sample_image.jpg'
})
print(response.json())
