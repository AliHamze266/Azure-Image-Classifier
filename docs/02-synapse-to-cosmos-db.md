# Step 2: From Synapse to Cosmos DB (and Power BI)

This guide turns the batch inference result into one JSON document per user and stores the documents in Azure Cosmos DB, where the Power BI dashboard reads them.

**Flow:** Blob Storage (`predictions.csv`) -> Azure Synapse Analytics (Spark pool) -> Azure Cosmos DB (container `user-images`) -> Power BI

The whole process runs in a Synapse Spark pool:

1. Spark gets access to Blob Storage and loads `predictions.csv` into a DataFrame.
2. For the load test, 1,000 users are simulated, each with 3,000 randomly chosen images. That makes 3 million image entries.
3. The image fields are combined into nested objects and grouped by `userId`, so each user ends up with exactly one JSON document.
4. The documents are written to Cosmos DB with the Cosmos DB Spark connector. Existing documents with the same ID are overwritten.

## Prerequisites

| Resource | Content |
|---|---|
| Storage account `<storage-account>`, container `predictions` | `predictions.csv` from [Step 1](01-batch-inference-synapse.md) |
| Cosmos DB account `<cosmos-account>` (NoSQL API) | database `image-labels`, container `user-images` |
| Synapse workspace with a Spark pool | runs the notebook below |

> **Security note:** The original project put the storage key and the Cosmos DB key directly into the notebook. Use a linked service or Azure Key Vault (`mssparkutils.credentials.getSecret`) instead.

## 1. Access Blob Storage and load the CSV

```python
spark.conf.set(
    "fs.azure.account.key.<storage-account>.blob.core.windows.net",
    "<storage-account-key>"
)

df = spark.read.csv(
    "wasbs://predictions@<storage-account>.blob.core.windows.net/predictions.csv",
    header=True,
    inferSchema=True,
    sep=";"
)
df.show(5)
```

- `header=True`: the first row contains the column names
- `inferSchema=True`: Spark detects the data types
- `sep=";"`: the file is semicolon-separated

## 2. Simulate users

```python
import pyspark.sql.functions as F

num_users = 1000      # prototype: 1,000 users
sample_size = 3000    # images per user

df_users = None
for uid in range(1, num_users + 1):
    # 3,000 random images per user
    sampled = df.orderBy(F.rand()).limit(sample_size)
    # assign each image to this user
    sampled = sampled.withColumn("userId", F.lit(uid))
    # one nested object per image = one entry in the JSON document
    sampled = sampled.withColumn(
        "image_object",
        F.struct(
            F.col("image_name"),
            F.col("predicted_labels"),
            F.col("image_url"),
            F.col("index")
        )
    )
    df_users = sampled if df_users is None else df_users.union(sampled)
```

Each image row gets a fixed `userId`, and its columns are combined into the struct `image_object`.

## 3. One document per user

```python
df_grouped = (
    df_users
    .groupBy("userId")
    .agg(F.collect_list("image_object").alias("images"))
    .withColumn("id", F.col("userId").cast("string"))   # required by Cosmos DB
)
```

- `groupBy("userId")`: groups all images of a user
- `collect_list("image_object")`: collects them into one list, stored as `images`
- `withColumn("id", ...)`: Cosmos DB needs a string field `id` as the document ID

Result: a DataFrame with exactly one JSON document per user.

## 4. Write to Cosmos DB

```python
df_grouped.write \
  .format("cosmos.oltp") \
  .option("spark.cosmos.accountEndpoint", "https://<cosmos-account>.documents.azure.com:443/") \
  .option("spark.cosmos.accountKey", "<cosmos-account-key>") \
  .option("spark.cosmos.database", "image-labels") \
  .option("spark.cosmos.container", "user-images") \
  .option("spark.cosmos.write.strategy", "ItemOverwrite") \
  .mode("append") \
  .save()
```

- The Cosmos DB connector connects through the account endpoint and key.
- `ItemOverwrite` updates documents that already exist with the same `id`.
- `append` mode adds new user documents to the container.

## 5. Power BI

The Power BI dashboard connects to the Cosmos DB container `user-images`. Filtering by `userId` limits the search to one user's images, as the use case requires (users must not see other users' images). Within that user, images can be searched by label in `predicted_labels`, and the dashboard shows them through `image_url`.

## Possible improvements

- The loop in step 2 builds 1,000 separate unions. A single `crossJoin` with a user DataFrame plus random sampling would be much faster.
- One document with 3,000 images per user works for the prototype. For real users, one document per image with `userId` as the partition key scales better and stays below the Cosmos DB item size limit (2 MB).

**Back:** [Step 1: Batch Inference in Azure Synapse](01-batch-inference-synapse.md)
