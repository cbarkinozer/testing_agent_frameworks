from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Define the collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, "example collection")

# Create a collection
collection = Collection("example_collection", schema)

# Insert some data
import numpy as np

vectors = [np.random.random(128).tolist() for _ in range(10)]
ids = [i for i in range(10)]

data = [
    ids,
    vectors
]

collection.insert(data)
print(f"Inserted data: {data}")

# Create an index on the 'vector' field
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 100},
    "metric_type": "L2"
}

collection.create_index(field_name="vector", index_params=index_params)
print("Index created")

# Load the collection
collection.load()

# Search the collection
query_vectors = [np.random.random(128).tolist() for _ in range(1)]
results = collection.search(query_vectors, "vector", param={"metric_type": "L2", "params": {"nprobe": 10}}, limit=3)
print(f"Search results: {results}")

# Disconnect from Milvus
connections.disconnect("default")
