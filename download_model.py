from chromadb.utils import embedding_functions

print("Starting the 79MB download. This might take a minute, please wait...")
# This forces the download to happen outside of a web server request
ef = embedding_functions.DefaultEmbeddingFunction()
ef(["This is a test sentence to trigger the download."])
print("Download complete! You are ready to go.")