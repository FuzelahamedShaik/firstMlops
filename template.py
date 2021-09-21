import os

dir = [
    os.path.join("data","raw"),
    os.path.join("data","processed"),
    "src",
    "notebooks",
    "saved_models"
]

for _dir_ in dir:
    #os.makedirs(_dir_,exist_ok=True)
    with open(os.path.join(_dir_,".gitkeep"),"w") as f:
        pass
    print(f"{_dir_} directory created with .gitkeep")

files = [
    ".gitignore",
    os.path.join("src","__init__.py"),
    "param.yaml",
    "dvc.yaml",
    "README.md"
]

for file in files:
    with open(file,"w") as f:
        pass
    print(f"{file} file created")