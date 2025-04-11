import re

def extract_extensions(file_content):
    # Initialize a set to store unique extensions
    extensions = set()

    # Regular expression to match lines with extensions
    # Assumes extensions are listed in the second column (positions after the first `|`
    start_pattern = "     - ."
    for line in file_content.splitlines():
        
        if line.startswith(start_pattern):
            # Extract extensions and strip spaces
            ext_str = line.split("- ")[1]
            # Split multiple extensions separated by commas
            for ext in ext_str.split(','):
                ext = ext.strip()
                if ext:  # Check if extension is not empty
                    extensions.add(ext)
    
    # Convert the set to list and return
    return sorted(list(extensions))

# Assume 'content' is the string read from the file
with open(r"E:\Oyvind\OF_git\run_pipeline\develop\bioformats-supported-formats.rst.txt", "r") as file:
    content = file.read()

# Call the function with the read content
extension_list = extract_extensions(content)

# Print the list of extensions
print(extension_list)
