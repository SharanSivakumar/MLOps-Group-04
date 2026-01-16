import torch
torch.set_printoptions(threshold=float('inf'), linewidth=200, edgeitems=10)

r = torch.randn(1, 224, 224)

# Save to text file
with open('array_output.txt', 'w') as f:
    f.write(str(r))

print("Array saved to array_output.txt")

