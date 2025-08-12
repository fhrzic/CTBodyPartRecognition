import os

def keep_only_matching_png_files(directory, target_words):
    """
    Keeps only PNG files in the specified directory and its subdirectories
    whose names contain any of the target words. Deletes all other PNG files.

    Args:
        directory (str): Path to the directory to process.
        target_words (list): List of words to search for in file names.
    """
    if not os.path.isdir(directory):
        print(f"The path {directory} is not a valid directory.")
        return

    for root, subdirs, files in os.walk(directory):
        print(f"Processing directory: {root}")
        for file_name in files:
            if file_name.lower().endswith(".png"):
                if any(word in file_name for word in target_words):
                    print(f"Keeping file: {os.path.join(root, file_name)}")
                else:
                    file_path = os.path.join(root, file_name)
                    try:
                        print(f"Deleting file: {file_path}")
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {e}")


def delete_files_with_target_words(directory, target_words):
    """
    Deletes files in the specified directory and its subdirectories
    if their names contain any of the target words.

    Args:
        directory (str): Path to the directory to process.
        target_words (list): List of words to search for in file names.
    """
    if not os.path.isdir(directory):
        print(f"The path {directory} is not a valid directory.")
        return

    for root, subdirs, files in os.walk(directory):
        print(f"Processing directory: {root}")
        print(f"Subdirectories: {subdirs}")
        print(f"Files: {files}")

        for file_name in files:
            if any(word in file_name for word in target_words):
                file_path = os.path.join(root, file_name)
               
                try:
                    print(f"Deleting file: {file_path}")
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")


if __name__ == "__main__":
    # Input directory from the user
    directory = "SAROS_working_data".strip()
    # List of target words to search for
    target_words = ["blobs", "TotalSegmentatior", "biggest", "filtered.nii", "original_image", "all.png", "coronal", "reduced_bounded"]
    delete_files_with_target_words(directory, target_words)
    target_words = ["main", "reducted_image"]
    keep_only_matching_png_files(directory, target_words)
