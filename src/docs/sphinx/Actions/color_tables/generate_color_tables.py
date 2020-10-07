from os import listdir
from os.path import isfile, join
from shutil import copyfile

def repeat_char(times, char):
    res = ""
    count = 0
    while count < times:
        res += char
        count += 1
    return res


vtkm_files = [f for f in listdir("./vtkm") if isfile(join("./vtkm", f))]
#print(vtkm_files)
vtkm_png_files = [s for s in vtkm_files if ".png" in s]
vtkm_png_files.sort()
examples = ".. _vtkm_color_tables:\n\n"
examples += "VTK-m Color Tables\n"
examples +="===================\n"
examples +="\n"

for c in vtkm_png_files:
    print(c)
    filename = c
    ctable_name = c.split(".")[0]
    # docutils can't handle names with spaces
    if c.count(' ') != 0:
      nospace = "".join(c.split())
      copyfile("./vtkm/"+c, "./vtkm/nospace/"+nospace);
      filename = 'nospace/'+nospace
#    print(ctable_name)
    examples += ctable_name
    examples +="\n"
    examples += repeat_char(len(ctable_name), "-") + "\n"
    examples +="\n"
    examples += ".. image:: color_tables/vtkm/" + filename + "\n"
    examples +="\n"

#print(examples)
examples_file = open("../VTKmColorTables.rst", "w")
examples_file.write(examples)
examples_file.close()

# -------------------- DEVIL RAY --------------------------------
dray_files = [f for f in listdir("./devil_ray") if isfile(join("./devil_ray", f))]
#print files
dray_png_files = [s for s in dray_files if ".png" in s]
dray_png_files.sort()
examples = ".. _dray_color_tables:\n\n"
examples += "Devil Ray Color Tables\n"
examples +="=======================\n"
examples +="\n"

for c in dray_png_files:

    filename = c
    ctable_name = c.split(".")[0]
    # docutils can't handle names with spaces
    if c.count(' ') != 0:
      nospace = "".join(c.split())
      copyfile("./devil_ray/"+c, "./devil_ray/"+nospace);
      filename = nospace
#    print(ctable_name)
    examples += ctable_name
    examples +="\n"
    examples += repeat_char(len(ctable_name), "-") + "\n"
    examples +="\n"
    examples += ".. image:: color_tables/devil_ray/" + filename + "\n"
    examples +="\n"

examples_file = open("../DRayColorTables.rst", "w")
examples_file.write(examples)
examples_file.close()
