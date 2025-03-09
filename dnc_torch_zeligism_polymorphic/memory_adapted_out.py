
Memory Configuration:
Batch size: 8
Memory size: 32
Word size: 8
Num reads: 4
Num writes: 1

Memory Initialization Details:
Memory shape: torch.Size([8, 32, 8])
Memory sample values: tensor([0., 0., 0., 0., 0.])
Memory mean: 0.0

Memory Test Results:
Memory size: 32
Word size: 8
Number of read heads: 4
Number of write heads: 1

Initial Memory State:
Memory data shape: torch.Size([8, 32, 8])
Memory data mean: 0.000000
Usage vector mean: 0.000000

Content-based Addressing Test:
Content weights shape: torch.Size([8, 4, 32])
Content weights sum: 1.000000
=======> UPDATE, memory_adapted
UPDATE, memory_data

=== Memory Update Debugging (Adapted Implementation) ===
========> ADAPTED, update_memory_data

Input Values:
write_weights shape: torch.Size([8, 1, 32]), mean: 0.007023
erase_vector shape: torch.Size([8, 1, 8]), mean: 0.491605
write_vector shape: torch.Size([8, 1, 8]), mean: -0.006847

Before Update:
Memory shape: torch.Size([8, 32, 8]), mean: 0.000000
First row sample: [0.0, 0.0, 0.0, 0.0, 0.0]
Second row sample: [0.0, 0.0, 0.0, 0.0, 0.0]
adapted, expanded_erase_vector.shape=torch.Size([8, 1, 1, 8])
adapted, erase.shape=torch.Size([8, 1, 32, 8])

Erase Calculations:
erase shape: torch.Size([8, 1, 32, 8]), mean: 0.003465
weighted_erase shape: torch.Size([8, 32, 8]), mean: 0.003465
keep shape: torch.Size([8, 32, 8]), mean: 0.996535

Write Calculations:
write shape: torch.Size([8, 1, 32, 8]), mean: -0.000355
weighted_write shape: torch.Size([8, 32, 8]), mean: -0.000355
memory * keep mean: 0.000000
+ weighted_write mean: -0.000355

After Update:
Memory shape: torch.Size([8, 32, 8]), mean: -0.000355
First row sample: [-0.0034439307637512684, 0.007923930883407593, -0.0032197325490415096, 0.009899431839585304, 0.005915476940572262]
Second row sample: [-0.0034439307637512684, 0.007923930883407593, -0.0032197325490415096, 0.009899431839585304, 0.005915476940572262]

After Memory Update:
Read words shape: torch.Size([8, 4, 8])
Read words mean: -0.000084
Updated memory data mean: -0.000355
Updated usage vector mean: 0.000000

After State Detachment:
Memory data requires grad: False

Memory Test Completed Successfully!
