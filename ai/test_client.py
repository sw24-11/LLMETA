import requests

# For Text Analysis
response = requests.post('http://127.0.0.1:5000/analyze', json={
    'data_type': 'text',
    'text': """
        Autoencoders
        Dor Bank, Noam Koenigstein, Raja Giryes
        Abstract: An autoencoder is a specific type of neural network...
    """
})
print(response.json())

# For Image Analysis
response = requests.post('http://127.0.0.1:5000/analyze', json={
    'data_type': 'image',
    'image_path': 'uploads/sample_image.jpg'
})
print(response.json())
