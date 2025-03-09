def verify_memory_initialization():
    """Verify that memory_adapted is initialized correctly."""
    # Create memory implementation
    memory = MemoryAdapted()

    # Initialize memory
    memory.init_state()

    # Step 1: Check initial memory state
    print("\nInitial Memory State:")
    print("Memory data shape:", memory.memory_data.shape)
    print("Memory data mean:", memory.memory_data.mean().item())

    # Step 2: Verify all components are zero-initialized
    print("\nVerifying zero initialization:")
    print("Memory matrix is zero:", torch.all(memory.memory_data == 0))
    print("Usage vectors are zero:", torch.all(memory.usage == 0))
    print("Link matrices are zero:", torch.all(memory.link == 0))
    print("Precedence weights are zero:", torch.all(memory.precedence_weights == 0))
    print("Read weights are zero:", torch.all(memory.read_weights == 0))
    print("Write weights are zero:", torch.all(memory.write_weights == 0))


if __name__ == "__main__":
    verify_memory_initialization()
