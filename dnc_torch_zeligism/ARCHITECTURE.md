## DNC
### constructor
- `input_size, output_size, controller_config, controller=LSTM
   
## Controller: LSTM
### constructor
- `input_size, **controller_config`

   
### Inputs
```python
    def __init__(self, input_size, output_size,
        controller_config, memory_config, Controller=nn.LSTM):
```

### State
```python
   self.controller_state = (
          torch.zeros(num_layers, BATCH_SIZE, hidden_size),
          torch.zeros(num_layers, BATCH_SIZE, hidden_size)
   )
   # Initialize read_words state
  self.read_words = torch.zeros(BATCH_SIZE, self.memory.num_reads, self.memory.word_size)
```
### Forward
```
   inputs.shape = (seq_len, batch_size, input_size)
   output = []
   Loop over seq length
       output += output_layer(...)
```

## Memory
### Inputs
- `memory_size, word_size, num_writes, num_reads`
- `num_writes`: number of write heads
- `num_reads`: number of read heads

### State
- `memory_data.shape=(batch_size, memory_size, word_size)
- `read_weights.shape=(batch_size, num_reads, memory_size)
- `write_weights.shape=(batch_size, num_writes, memory_size)
- `precedence_weights.shape=(batch_size, num_writes, memory_size)
- `link.shape=(batch_size, num_writes, memory_size, memory_size)
- `usage.shape=(batch_size, memory_size)

### Update 
- update memory state
- Inputs: `interface` (dict)
- called from dnc.forward (memory layer)

### Outputs

## Interface (DNC_InterfaceLayer)
### Inputs
### State
### Outputs
### Forward
- returns a dict[str, Tensor] for "read_keys", "read_strengths", ...
