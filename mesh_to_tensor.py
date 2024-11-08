import itertools
import torch
import trimesh
import os

# Function to load mesh and convert to tensors
def load_mesh_as_tensors(filepath):
    # Load the mesh
    mesh = trimesh.load(filepath, process=False)
    
    # Convert vertices and faces to PyTorch tensors
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)
    
    return vertices, faces

# List of mesh file paths (relative to the current working directory)
mesh_dir = '3D_Vocal_Tract_Meshes'
mesh_files = [os.path.join(mesh_dir, filename) for filename in ['segmentation_aa_mesh_300.ply', 'segmentation_ll_mesh_300.ply', 'segmentation_oo_mesh_300.ply']]

# Directory to save the output tensors
output_dir = 'tensors'

# Iterate through all permutations of source and target meshes
for source, target in itertools.permutations(mesh_files, 2):
    # Load source and target meshes as tensors
    VS, FS = load_mesh_as_tensors(source)
    VT, FT = load_mesh_as_tensors(target)
    
    # Generate output filename based on source and target mesh names
    source_name = os.path.basename(source).split('_')[1]  # Extract the identifier (e.g., 'aa')
    target_name = os.path.basename(target).split('_')[1]  # Extract the identifier (e.g., 'll')
    output_filename = f'{source_name}_{target_name}_mesh_300.pt'
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the tensors into a .pt file
    torch.save((VS, FS, VT, FT), output_path)
    
    print(f'Saved {output_path}')
