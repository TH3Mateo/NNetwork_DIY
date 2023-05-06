import os
def create_path(filename):return "\\".join(os.path.dirname(__file__).split("\\")[0:-1])+"\\"+filename

