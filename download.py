import kagglehub

path = kagglehub.dataset_download(
    "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"
)

print("Path to dataset files:", path)
