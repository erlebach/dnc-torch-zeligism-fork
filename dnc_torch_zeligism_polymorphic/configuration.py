# General configuraiton
training_config = {
    "random_seed": 10,
    "batch_size": 8,
    "epsilon": 1.e-6,
    "learning_rate": 1.e-3,
    "momentum": 0.9,
    "num_examples": 500,
}
training_config["checkpoint"] = 100 # training_config["num_examples"] // 200

# Controller configuration
controller_config = {
    "hidden_size": 64,
    "num_layers": 1,
}

# Memory Configuration
memory_config = {
    "memory_size": 32,
    "word_size": 8,
    "num_writes": 1,
    "num_reads": 4,
}
