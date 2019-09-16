import sys
import subprocess

# parse hashes.txt file to check which BLT we should be using 
def sexe(cmd,ret_output=False,echo = False):
    """ Helper for executing shell commands. """
    if echo:
        print("[exe: {}]".format(cmd))
    if ret_output:
        p = subprocess.Popen(cmd,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        res = p.communicate()[0]
        res = res.decode('utf8')
        return p.returncode,res
    else:
        return subprocess.call(cmd,shell=True)

def read_blt_hash():
    with open("hashes.txt") as f:
       for txt in f.readlines():
           if txt.count("blt") > 0:
               key = "commit='"
               cmt_start = txt.find(key) + len(key)
               # find commit
               commit = txt[cmt_start:].strip()
               # remove trailing '
               return commit[:-1]
    # hard fail
    print("[ERROR: could not read blt info from hashes.txt]")
    sys.exit(-1)

def read_last_blt_hash():
    cmd = 'git log --patch develop -- src/blt/'
    rres, rout = sexe(cmd,ret_output=True)
    print rres, rout
    for txt in rout.split("\n"):
         print txt
         if txt.count("+Subproject") > 0:
             key = "commit "
             cmt_start = txt.find(key) + len(key)
             commit = txt[cmt_start:].strip()
             return commit
    print("[ERROR: could not read git ref log]")
    # hard fail
    sys.exit(-1)


def main():
    expected = read_blt_hash()
    current  = read_last_blt_hash()
    print("[blt sanity check]")
    print("[ Expected sha: {0}]".format(expected))
    print("[ Current  sha: {0}]".format(current))
    if expected != current:
        print("[ERROR: sha mismatch!]")
        print("[If you wanted to update blt - did you update hashes.txt?]")
        sys.exit(-1)
    else:
        print("[PASS: sanity is preserved]")
        sys.exit(0)


if __name__ == "__main__":
    main()
    
    

