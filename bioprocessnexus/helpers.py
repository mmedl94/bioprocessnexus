import os
import zipfile
import webbrowser
import customtkinter as ctk


def normalize(array, mus, stds):
    """
    Normalizes the input array by subtracting the mean and dividing by the standard deviation.

    Args:
        array (numpy array): Array to be normalized.
        mus (numpy array): Array of means for each feature in "array".
        stds (numpy array): Array of standard deviations for each feature in "array".

    Returns:
        normalized_array: Normalized input array
    """
    normalized_array = (array-mus)/stds
    return normalized_array


def denormalize(array, mus, stds):
    """
    Reverts normalization by applying mean and standard deviation scaling.

    Args:
        array (numpy array): Array to be denormalized.
        mus (numpy array): Array of means used for normalization.
        stds (numpy array): Array of standard deviations used for normalization.

    Returns:
        denormalized_array: Denormalized input array
    """
    denormalized_array = array*stds+mus
    return denormalized_array


def open_help():
    """
    Opens a web browser to the help tutorial URL.
    """
    webbrowser.open(
        "https://bioprocessnexus.readthedocs.io/en/latest/")


def zip_dir(parent):
    """
    Compresses a directory selected by the user into a zip file.

    Args:
        parent: The main application instance

    This function prompts the user to select a directory, creates a zip file with the same name,
    and saves it in the same location.
    """
    parent.zip_dir = ctk.filedialog.askdirectory()
    if not parent.zip_dir:
        return
    zip_name = parent.zip_dir + ".zip"

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for folder_name, subfolders, filenames in os.walk(parent.zip_dir):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_ref.write(file_path, arcname=os.path.relpath(
                    file_path, parent.zip_dir))
    zip_ref.close()


def unzip_dir(parent):
    """
    Extracts a zip file selected by the user into a new directory.

    Args:
        parent: The main application instance

    This function prompts the user to select a zip file, creates a directory with the same name,
    and extracts the zip file contents into this directory.
    """
    parent.unzip_dir = ctk.filedialog.askopenfile(
        filetypes=[(".zip files", "*.zip")]).name
    if os.path.exists(str.rsplit(parent.unzip_dir, ".zip", 1)[0]) is False:
        os.mkdir(str.rsplit(parent.unzip_dir, ".zip", 1)[0])
    with zipfile.ZipFile(parent.unzip_dir, 'r') as zip_ref:
        zip_ref.extractall(str.rsplit(parent.unzip_dir, ".zip", 1)[0])


def nice_round(num):
    """
    Rounds a number based on its magnitude to provide a concise output.

    Args:
        num (float): Number to be rounded.

    Returns:
        float: Rounded number with appropriate precision.
    """
    if num > 10000:
        return round(num)
    elif num < 10000 and num > 1000:
        return round(num, 1)
    elif num < 1000 and num > 100:
        return round(num, 2)
    elif num < 100 and num > 10:
        return round(num, 3)
    elif num < 10 and num > 1:
        return round(num, 4)
    else:
        counter = 0
        for i in str(num).split(".")[1]:
            if i == "0":
                counter += 1
            else:
                break
        return round(num, 4+counter)


def check_dir(parent, y_dir, dir_type, central_log=0):
    """
    Verifies and creates the necessary directory structure for storing logs or images.

    Args:
        parent: The main application instance
        y_dir (str): Directory name for a specific response variable.
        dir_type (str): Type of directory to create (e.g., "logs", "images").
        central_log (int, optional): If set to 1, creates only the main directory without nested folders.

    Returns:
        str: Path to the created directory, if central_log is 0.
    """

    mother_dir = parent.model_dir.rsplit("/", 2)[0]
    model_name = parent.model_dir.rsplit("/")[-1]
    if os.path.exists(f"{mother_dir}/{dir_type}") is False:
        os.mkdir(f"{mother_dir}/{dir_type}")

    if os.path.exists(f"{mother_dir}/{dir_type}/{model_name}") is False:
        os.mkdir(f"{mother_dir}/{dir_type}/{model_name}")

    if central_log == 1:
        pass
    elif os.path.exists(f"{mother_dir}/{dir_type}/{model_name}/{y_dir}") is False:
        os.mkdir(f"{mother_dir}/{dir_type}/{model_name}/{y_dir}")

    if central_log == 0:
        return f"{mother_dir}/{dir_type}/{model_name}/{y_dir}"
