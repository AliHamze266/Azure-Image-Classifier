# Step 1: Batch Inference in Azure Synapse

This guide labels all images in a Blob Storage container with the trained model and saves the result as a CSV file.

**Flow:** Blob Storage (images) -> Azure Synapse Analytics (Spark pool) -> Blob Storage (`predictions.csv`)

## Prerequisites

| Resource | Content |
|---|---|
| Storage account `<storage-account>`, container `images` | the Flickr8k images |
| Storage account `<storage-account>`, container `predictions` | target for the result CSV |
| Data Lake Storage Gen2 `<adls-account>`, file system `ml-assets` | `model_multilabel_5.pth` (created by `02_model_training.ipynb`) and `dataset/top_tokens_with_images.csv` |
| Synapse workspace with a Spark pool | runs the notebook below |

> **Security note:** The original project used the storage account key directly in the notebook. Do not do this outside of a short-lived prototype. Use a linked service with a managed identity or read the key from Azure Key Vault (`mssparkutils.credentials.getSecret`).

Run the following cells one after another in a Synapse notebook.

## 1. Install torchvision

```python
%pip install torchvision
```

Installs torchvision on the Spark pool. It contains EfficientNet and the image transforms the model needs. Synapse restarts the kernel after the installation.

## 2. Import PyTorch

```python
import torch
import torch.nn as nn
import torchvision.models as models
```

- `torch`: core functions for neural networks
- `nn`: layers and model building blocks
- `models`: pretrained backbones such as EfficientNet

## 3. Copy the model to the Spark node

```python
from notebookutils import mssparkutils

mssparkutils.fs.cp(
    "abfss://ml-assets@<adls-account>.dfs.core.windows.net/model_multilabel_5.pth",
    "file:/mnt/model_multilabel_5.pth"
)
print("Model copied to /mnt")
```

PyTorch cannot open `abfss://` paths directly, so the `.pth` file is copied to the local file system of the node first.

## 4. Load the label list

```python
import pandas as pd

csv_path = "abfss://ml-assets@<adls-account>.dfs.core.windows.net/top_tokens_with_images.csv"
tokens_df = pd.read_csv(csv_path)
tokens = tokens_df["token"].tolist()
print("Tokens loaded:", len(tokens))
```

The CSV contains the 100 tokens (labels) the model predicts. Their number defines the size of the output layer.

## 5. Define the model architecture

```python
class MultiLabelEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return torch.sigmoid(self.classifier(x))
```

This must be exactly the same architecture as in training. EfficientNet-B0 extracts the features, a linear layer produces one score per label, and the sigmoid turns each score into an independent probability.

## 6. Create the model and set the threshold

```python
num_classes = len(tokens)
model = MultiLabelEfficientNet(num_classes=num_classes)
threshold = 0.25
print("Model created, threshold:", threshold)
```

0.25 is the threshold with the best F1 score in `02_model_training.ipynb`.

## 7. Load the trained weights

```python
state_dict = torch.load("/mnt/model_multilabel_5.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
print("Model loaded")
```

`model.eval()` switches to inference mode (no dropout, no batch-norm updates), so predictions are stable.

## 8. Image preprocessing

```python
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
print("Transform defined")
```

Images are prepared exactly as in training: resized to 224×224, converted to a tensor and normalized with the ImageNet statistics.

## 9. Prediction for one image

```python
def predict_image(model, image_path, threshold=0.25):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    preds = (outputs[0] > threshold).nonzero(as_tuple=True)[0].tolist()
    return preds
```

Returns the indices of all labels with a probability above the threshold.

## 10. Give Spark access to the storage account

```python
spark.conf.set(
    "fs.azure.account.key.<storage-account>.blob.core.windows.net",
    "<storage-account-key>"
)
print("Access key set")
```

Without access, Synapse cannot read the `images` container. See the security note above for better options than a plain key.

## 11. List all images

```python
image_paths = mssparkutils.fs.ls("wasbs://images@<storage-account>.blob.core.windows.net/")
image_paths = [f.path for f in image_paths]
print("Images found:", len(image_paths))
print(image_paths[:5])
```

In the project, 8,092 files were found. This list is the input for the batch inference.

## 12. Batch inference function

```python
import pandas as pd

def run_batch_inference(model, image_paths, threshold=0.25):
    rows = []
    for img_path in image_paths:
        img_name = img_path.split("/")[-1]
        local_tmp = f"/tmp/{img_name}"
        image_url = f"https://<storage-account>.blob.core.windows.net/images/{img_name}"
        try:
            mssparkutils.fs.cp(img_path, f"file:{local_tmp}")
            pred_ids = predict_image(model, local_tmp, threshold)
            labels = [tokens[i] for i in pred_ids]
            rows.append({
                "image_name": img_name,
                "predicted_labels": ", ".join(labels),
                "image_url": image_url
            })
        except Exception as e:
            rows.append({
                "image_name": img_name,
                "predicted_labels": f"ERROR: {str(e)}",
                "image_url": image_url
            })
    df = pd.DataFrame(rows)
    df["Index"] = range(1, len(df) + 1)
    return df
```

For each image, the function copies the file to `/tmp/`, runs the prediction and maps the label indices to token names. The result has one row per image:

| Column | Content |
|---|---|
| `image_name` | file name of the image |
| `predicted_labels` | all predicted tokens as a comma-separated string |
| `image_url` | direct link to the image in Blob Storage |
| `Index` | running number |

## 13. Run the batch inference

```python
df_predictions = run_batch_inference(model, image_paths, threshold=0.25)
print(df_predictions.head())
print("Predictions created:", len(df_predictions))
```

## 14. Save the result to Blob Storage

```python
from notebookutils import mssparkutils

# Write a local file with a semicolon as separator
local_csv = "/tmp/predictions.csv"
df_predictions.to_csv(local_csv, index=False, sep=";", encoding="utf-8")

with open(local_csv, "r", encoding="utf-8") as f:
    csv_text = f.read()

# Target: container "predictions", no subfolder
output_path = "wasbs://predictions@<storage-account>.blob.core.windows.net/predictions.csv"

mssparkutils.fs.put(output_path, csv_text, overwrite=True)
print("predictions.csv saved to:", output_path)
```

The semicolon is used because the label column itself contains commas. `mssparkutils.fs.put()` expects raw text, so the CSV is read back as a string before the upload.

The first 20 predictions are included in the repository as [`predictions_sample.csv`](../predictions_sample.csv) (without the image URLs of the original storage account).

**Next:** [Step 2: From Synapse to Cosmos DB](02-synapse-to-cosmos-db.md)
