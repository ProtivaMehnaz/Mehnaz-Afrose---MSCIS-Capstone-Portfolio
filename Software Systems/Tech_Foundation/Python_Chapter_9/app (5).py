# Mehnaz Afrose
from pathlib import Path
from time import ctime
import shutil


# Chapter 9.1- Python Standard Library
# No coding. only an introduction of Python Standard Library


# Chapter 9.2- Working With Paths
print("Chapter 9.2- Working With Paths " + "-" * 20)
# Raw string allows single back slashes in path
path = Path(r"ecommerce/shipping.py")
print(f"Is the path object a file? {path.is_file()}")


# Alternative is to use forward slashes in path
path = Path("ecommerce/__init__.py")
print(f"Does the path object exist? {path.exists()}")
print(f"Is the path object a file? {path.is_file()}")
print(f"Is the path object a directory? {path.is_dir()}")
print(path.name)
print(path.stem)
print(path.suffix)
print(path.parent)
path = path.with_suffix(".txt")
print(path)


# Chapter 9.3- working with Directories
print("\nChapter 9.3- Working with Directories" + "-" * 20)
# create a path object that represents a directory
path = Path("blog")
if not path.exists():
    print("Director does not yet exists")
    path.mkdir()
else:
    print("Directory already exists.")


# create a path to the current directory . one period.
# two periods is the parrent directory .. if desired.
path = Path(".")
print("List of directories:")
path_of_directories = [p for p in path.iterdir() if p.is_dir()]

print(path_of_directories)

print("List of .py files:")
path_of_pyfiles = [p for p in path.rglob("*.py")]
print(path_of_pyfiles)


# How to access a path object inside a list
one_file = path_of_pyfiles[0]
print(one_file.name)
print(one_file.stem)
print(one_file.suffix)


# Chapter 9.4- Working with Files
print("\nChapter 9.4- Working with Files" + "-" * 20)
path = Path("ecommerce/__init__.py")
# Print creation time on Windows and time changed on Unix
# ctime requires "from time import ctime" see line 3
print(ctime(path.stat().st_ctime))

# Easy way to read the contents of a text file
print(path.read_text())

# Easy way to write to a file. Warning!
# This methode overwrites any existing content inside the file.
path_tax = Path("ecommerce/tax.py")
message = "# Add another comment to the file"
path_tax.write_text(message)

# Easy way to copy files.
# If the target file exists, it will be overwritten.
# shutil required "import shutil" see line 3
source = Path("ecommerce/__init__.py")
target = Path("ecommerce/__init__backup.py")
shutil.copy(source, target)
