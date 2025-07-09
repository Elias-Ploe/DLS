import os

directory = '/home/elias/proj/_photon_correlation/10C_thym'
start_old = 1014
end_old = 4422
offset = 1

# First pass: Rename to temporary files
for i in range(start_old, end_old + 1):
    old_name = f"OUT{i}.DAT"
    temp_name = f"TEMP_OUT{i}.DAT"
    old_path = os.path.join(directory, old_name)
    temp_path = os.path.join(directory, temp_name)

    if os.path.exists(old_path):
        os.rename(old_path, temp_path)
        print(f"Temporarily renamed: {old_name} -> {temp_name}")
    else:
        print(f"File not found: {old_name}")

# Second pass: Rename temporary files to final target names
for i in range(start_old, end_old + 1):
    temp_name = f"TEMP_OUT{i}.DAT"
    new_name = f"OUT{i + offset}.DAT"
    temp_path = os.path.join(directory, temp_name)
    new_path = os.path.join(directory, new_name)

    if os.path.exists(temp_path):
        os.rename(temp_path, new_path)
        print(f"Renamed: {temp_name} -> {new_name}")
    else:
        print(f"Temporary file not found: {temp_name}")