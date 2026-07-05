import kagglehub

# Download latest version
path = kagglehub.dataset_download("joebeachcapital/credit-card-fraud")

print("Path to dataset files:", path)
