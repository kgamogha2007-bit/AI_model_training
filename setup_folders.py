import os

classes = ['out', 'six', 'wide']
for split in ['train', 'val']:
    for cls in classes:
        os.makedirs(os.path.join('data', split, cls), exist_ok=True)

print("Folder structure created.")