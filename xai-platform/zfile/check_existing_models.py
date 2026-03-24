#!/usr/bin/env python3
"""
Check all models in the database for incorrect feature schemas.
"""

import pymongo
from pprint import pprint

# Connect to local MongoDB (docker-compose maps port 27017)
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['xai']  # database name from config

print("Connected to MongoDB")
print(f"Database: {db.name}")
print(f"Collections: {db.list_collection_names()}\n")

models_coll = db.models

# Count total models
total = models_coll.count_documents({})
print(f"Total models in database: {total}\n")

if total == 0:
    print("No models found.")
else:
    # Fetch all models (limit to a few)
    models = list(models_coll.find({}))
    print(f"Showing all {len(models)} models:")
    print("=" * 100)

    for model in models:
        model_id = str(model.get('_id'))
        name = model.get('name')
        user_id = model.get('user_id')
        framework = model.get('framework')
        feature_schema = model.get('feature_schema', [])
        print(f"\nModel ID: {model_id}")
        print(f"Name: {name}")
        print(f"User ID: {user_id}")
        print(f"Framework: {framework}")
        print(f"Feature count: {len(feature_schema)}")

        # Check for categorical features
        cat_features = [fs for fs in feature_schema if fs.get('type') == 'categorical']
        num_features = [fs for fs in feature_schema if fs.get('type') == 'numeric']
        unknown_type = [fs for fs in feature_schema if fs.get('type') not in ('categorical', 'numeric')]

        print(f"Categorical: {len(cat_features)} | Numeric: {len(num_features)} | Unknown: {len(unknown_type)}")

        if cat_features:
            print("Categorical features (with options count):")
            for fs in cat_features:
                opts = fs.get('options', [])
                print(f"  - {fs['name']}: {len(opts)} options")
        else:
            print("** NO CATEGORICAL FEATURES DETECTED **")

        if unknown_type:
            print("Features with unknown type:")
            for fs in unknown_type:
                print(f"  - {fs['name']}: {fs.get('type')}")

        print("-" * 100)
