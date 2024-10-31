import os
import zipfile
import webbrowser
import customtkinter as ctk


def normalize(array, mus, stds):
    normalized_array = (array-mus)/stds
    return normalized_array


def denormalize(array, mus, stds):
    denormalized_array = array*stds+mus
    return denormalized_array


def open_help():
    webbrowser.open(
        "https://boku.ac.at/rali/stat/testseiten-stat/testseiten-stat-matthias/tutorials")


def zip_dir(parent):
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
    parent.unzip_dir = ctk.filedialog.askopenfile(
        filetypes=[(".zip files", "*.zip")]).name
    if os.path.exists(str.rsplit(parent.unzip_dir, ".zip", 1)[0]) is False:
        os.mkdir(str.rsplit(parent.unzip_dir, ".zip", 1)[0])
    with zipfile.ZipFile(parent.unzip_dir, 'r') as zip_ref:
        zip_ref.extractall(str.rsplit(parent.unzip_dir, ".zip", 1)[0])


def nice_round(num):
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

