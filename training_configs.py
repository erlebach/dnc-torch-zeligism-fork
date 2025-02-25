# Just a simple configs file with all the hyperparameters you need to set

# Note: 0 sets the seed to torch's initial seed
# RANDOM_SEED = 10

# Training-specific hyperparameters
# BATCH_SIZE = 8
# EPSILON = 1e-6
# LEARNING_RATE = 1e-4
# MOMENTUM = 0.9
# NUM_EXAMPLES = 20000
# CHECKPOINT = NUM_EXAMPLES // 200

# Controller configurations
# HIDDEN_SIZE = 64
# NUM_LAYERS = 1

# Memory configurations
# MEMORY_SIZE = 32
# WORD_SIZE = 8
# NUM_WRITES = 1
# NUM_READS = 4

# config = {}
# config["memory_size"] = MEMORY_SIZE
# config["word_size"] = WORD_SIZE
# config["num_writes"] = NUM_WRITES
# config["num_reads"] = NUM_READS
# config["hidden_size"] = HIDDEN_SIZE
# config["num_layers"] = NUM_LAYERS

# General configuraiton
config = {
    "random_seed": 10,
    "batch_size": 8,
    "epsilon": 1.e-6,
    "learning_rate": 1.e-4,
    "momentum": 0.9,
    "num_examples": 20000,
}
config["checkpoint"] = config["num_examples"] // 200

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
