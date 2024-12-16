sweep_config = {
    'checkpoint_path': "/lustre/groups/ml01/projects/2023_nicheformer/pretrained_models/nicheformmer.ckpt",
    'freeze': False,
    'reinit_layers': None,
    'extractor': False,
    'batch_size': 12,
    'lr': 1e-4,
    'warmup': 1,
    'max_epochs': 50000,
    'pool': 'mean',
    'n_classes': 17,
    'dim_prediction': 1,
    'supervised_task': 'niche_classification',
    'regress_distribution': False,
    'predict_density': False,
    'ignore_zeros': False,
    'baseline': False,
    'organ': 'brain',
    'label': 'region',
    }

hp_config = {
    'freeze': False,  # Don't freeze the backbone during fine-tuning
    'reinit_layers': None,  # No layers to reinitialize
    'extractor': False,  # No extractor before the linear layer
    'batch_size': 8,  # Adjust based on memory constraints
    'lr': 1e-4,  # Learning rate
    'warmup': 1,  # Number of warmup steps
    'max_epochs': 10,  # Total epochs for fine-tuning
    'pool': 'mean',  # Use mean pooling for feature aggregation
    'n_classes': 11,  # Number of cell types to classify
    'dim_prediction': 1,  # Dimensionality of the prediction layer
    'supervised_task': 'niche_multiclass_classification',  # Task type
    'regress_distribution': False,  # Not performing regression over density distribution
    'predict_density': False,  # Not predicting cell density
    'ignore_zeros': False,  # Include zero labels in classification
    'baseline': False,  # Indicates a trained Transformer, not baseline
    'organ': 'pancreas',  # Dataset-specific information for logging
    'label': 'cell_type',  # Target label column
    'extract_layers': -1,  # Use the last hidden layer for linear head
    'function_layers': 'mean',  # Function to combine hidden layers
    'predict': False,  # Default behavior (adjust based on use-case)
    'without_context': True  # Whether context is included
}

