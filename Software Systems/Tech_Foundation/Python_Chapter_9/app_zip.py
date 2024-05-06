# Mehnaz Afrose
from pathlib import Path
from zipfile import ZipFile

# Chapter 9.5- Working with Zip files
print("\nChapter 9.5- Working with Zip files " + "-" * 20)
# requires "from zipfile import ZipFile" see line 32
# I modified Mosh's code to make zipping more reusable
# you can change these three variables for your future needs
# If the zip file exists, this code will overwrite the existing zip file

zip_file_name = "files.zip"
directory_path = Path("ecommerce")
files_to_zip = "*.*"


with ZipFile(zip_file_name, "w") as zip:
    for path in Path(directory_path).rglob(files_to_zip):
        zip.write(path)


source_to_unzip = "files.zip"
target_directory = "extract"
with ZipFile(source_to_unzip) as zip:
    # return information about the zip file, if desired
    print(zip.namelist())
    info = zip.getinfo("ecommerce/__init__.py")
    print(info.file_size)
    print(info.compress_size)
# extract the contents of the zip file to the target_directory
# target_directory will be created if it doesn't exists
# extracting will overwrite existing files without warning
    zip.extractall(target_directory)
print("done extracting")


# How to extract multiple zip files
# create a list object, a list of files to unzip 
list_of_files =["files.zip","icons.zip"]
# extract each zip file, one at a time, in a for loop
target_directory = "extract"
for file in list_of_files:
    with ZipFile(file) as zip:
        zip.extractall(target_directory)
print("done extracting from multiple files")
