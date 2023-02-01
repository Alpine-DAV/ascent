#
# ascent_rc_compiler.py
#
#################
# creates base64 encoded conduit node that holds
# file system contents of a given directory that
# can be used as a resource to reconstruct a directory
################

import os
from conduit import *
from os.path import join as pjoin
import multiprocessing
import ctypes

def test():
    n = Node()
    n["tout_rc_a.txt"] = "here is a string for a.txt\n over\n"
    n["tout_rc_b.txt"] = "here is a string for b.txt\n over\n"
    n["tout/a/path/to/tout_rc_c.txt"] = "yikes!"
    r = gen_cpp_resource_header_source(n,"IMPORTANT_STUFF")
    print(r)
    expand_to_file_system(n,".")

def gen_cpp_resource_header_source(n,def_name):
    res_string = n.to_string(protocol="conduit_base64_json")
    res =  '// -----\n'
    res += '// NOTE: THIS FILE IS AUTO GENERATED\n'
    res += '// -----\n'
    res += '\n'
    res += '// -----\n'
    res += '// NOTE: Windows does not like large strings, so we provide two forms.\n'
    res += '// -----\n'
    res += '// -------------------- \n'
    res += '// (non windows case)\n'
    res += '#ifndef _WIN32\n'
    res += 'std::string RC_{0} =  R"xyzxyz(\n'.format(def_name)
    # TODO: split these across lines?
    res += res_string
    res += '\n)xyzxyz";\n'
    res += '// -------------------- \n'
    res += '// (windows case) \n'
    res += '#else\n'
    res += 'char [] RC_{0}_CHARS = \n'.format(def_name) + "{"
    res_bytes =  bytearray(res_string,'utf-8')
    res_hexs = [ hex(v) for v in res_bytes]
    res_bytes_code = ",".join(res_hexs)
    # TODO: split these across lines?
    res += res_bytes_code
    res += '};\n'
    res += 'std::string RC_{0} = std::string(RC_{0}_CHARS);\n'.format(def_name,def_name);
    res += '#endif\n'
    return res

def digest_filesystem_tree(path):
    n = Node()
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            fpath = os.path.join(root, name)
            dest  = fpath[len(path):]
            print("reading: {} into {}".format(fpath,dest))
            rcontents = open(fpath).read();
            # this work around is here b/c some file contents (bootstrap.js)
            # were causing strlen to fail during Node::operator=(char*)
            # this might be an encoding issue, but setting with explicit buffer
            # and length avoids the issue
            s = Schema()
            s.set(DataType.char8_str(len(rcontents)))
            sval = multiprocessing.RawArray(ctypes.c_ubyte,bytearray(rcontents,'utf8'))
            n[dest].set(s,sval)
        #for name in dirs:
        #    print(os.path.join(root, name))
    #print(n)
    return n;

def check_filesystem_tree(n,path):
    ok = True
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            fpath = os.path.join(root, name)
            dest  = fpath[len(path):]
            print("checking: {} vs {}".format(fpath,dest))
            curr_data = n.fetch(dest).value()
            print(curr_data)
            read_data = open(fpath).read()
            print("OK? = ")
            print(dir(curr_data))
            print(dir(read_data))
            print(curr_data == read_data)
            if curr_data != read_data:
                ok = False
    return ok

def expand_to_file_system(n,path):
    print(n)
    print(n.dtype())
    print(dir(n.dtype()))
    print(n.dtype().id())
    if n.dtype().id() != 1:
        print(path)
        open(path,"w").write(n.value())
    else:
        for c in n.children():
            cname = c.name()
            cn = c.node()
            if not os.path.isdir(path):
                os.mkdir(path)
            expand_to_file_system(cn,pjoin(path,cname))


n_cinema = digest_filesystem_tree("../src/libs/ascent/web_clients/cinema/")

r = gen_cpp_resource_header_source(n_cinema,"CINEMA_WEB")
print("[creating ascent_resources_cinema_web.hpp]")
open("ascent_resources_cinema_web.hpp","w").write(r)

n_ascent = digest_filesystem_tree("../src/libs/ascent/web_clients/ascent/")

r = gen_cpp_resource_header_source(n_ascent,"ASCENT_WEB")
print("[creating ascent_resources_ascent_web.hpp]")
open("ascent_resources_ascent_web.hpp","w").write(r)
