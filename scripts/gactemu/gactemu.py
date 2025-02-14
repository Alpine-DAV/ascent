###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################
"""
gactemu.py (Generally Awesome Emu Wrangler)

Digests .github/workflows/ specs and generates scripts to run steps 
locally using docker.


"""

import yaml
import os
import stat
import glob
import subprocess
import sys


class CTX:
    def __init__(self,ctx=None):
        self.name = ""
        self.txt  = ""
        self.cwd  = ""
        self.container = ""
        if not ctx is None:
            self.name = ctx.name
            self.txt  = ctx.txt
            self.container = ctx.container
            self.cwd = ctx.cwd
    
    def set_name(self,name):
        self.name = name
    
    def set_container(self,container):
        self.container = container
    
    def set_cwd(self,cwd):
        self.cwd = cwd
    
    def print(self,txt):
        self.txt += txt + "\n"
    
    def print_esc(self,txt, tag = None):
        res = ""
        if not tag is None:
            res = "# [{0}: {1}]".format(tag,txt)
        else:
            res = "# [{0}]".format(txt)
        self.txt += res + "\n"

    def finish(self):
        print("[creating: {0}".format(self.script_file()))
        f = open(self.script_file(),"w",newline='\n')
        f.write(self.gen_script())
        os.chmod(self.script_file(), stat.S_IRWXU  | stat.S_IRWXG  | stat.S_IRWXO )

        print("[creating: {0}".format(self.launch_file()))
        f= open(self.launch_file(),"w",newline='\n')
        f.write(self.gen_launch())
        os.chmod(self.launch_file(), stat.S_IRWXU  | stat.S_IRWXG  | stat.S_IRWXO )

    def script_file(self):
        return "GACTEMU-SCRIPT-" + self.name + ".sh"

    def launch_file(self):
        return "GACTEMU-LAUNCH-" + self.name + ".sh"

    def gen_script(self):
        res  = "#!/bin/bash\n"
        res += "# CONTAINER: {0}\n".format(self.container)
        res += "set -e\n"
        res += "set -x\n"
        res += str(self)
        return res

    def gen_launch(self):
        res  = "#!/bin/bash\n"
        res += "set -x\n"
        res += "docker stop gactemu_exec\n"
        res += "docker rm  gactemu_exec\n"
        res += "docker run --name gactemu_exec -t -d {0}\n".format(self.container)
        res += 'docker exec gactemu_exec sh -c "apt-get update && env apt-get install -y sudo"\n'
        res += 'docker exec gactemu_exec sh -c "useradd -ms /bin/bash -G sudo user && echo \\"user:docker\\\" | chpasswd"\n'
        res += 'docker exec gactemu_exec sh -c "echo \\"user ALL=(root) NOPASSWD:ALL\\\" > /etc/sudoers.d/user &&  chmod 0440 /etc/sudoers.d/user"\n'
        res += 'docker cp {0} gactemu_exec://home/user/\n'.format(self.script_file())
        res += 'docker exec -u user gactemu_exec  sh -c "echo [GACT EXEC SCRIPT]!"\n'
        res += 'docker exec -u user --workdir=//home/user/ -i gactemu_exec ./{0}\n'.format(self.script_file())
        return res

    def __str__(self):
        res =""
        if self.name != "":
            res += "# [[{0}]]\n".format(self.name)
        res += self.txt
        return res


def shexe(cmd,ret_output=False,echo = True):
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

def sanitize_var(v):
    if type(v)==bool:
        if v:
            return "ON"
        else:
            return "OFF"
    return v

def proc_root(tree, config):
    for k,v in tree.items():
        if k == "name":
            config["root_name"] = v
        if k == "jobs":
            proc_jobs(v, config)

def map_gact_runners(runner,config):
    if runner in config["runners"]:
        return config["runners"][runner]
    else:
        print("# unsupported runner:" + runner)
        return "UNSUPPORTED"

def proc_jobs(tree, config):
    for job_name, job in tree.items():
        job_ctx = CTX()
        job_full_name = config["root_name"] + "-" + job_name
        job_ctx.print_esc(tag = "job", txt =job_full_name )
        if "runs-on" in job.keys():
             job_ctx.set_container(map_gact_runners(job["runs-on"],config))
        else:
            job_ctx.set_container(config["default_container"])
        job_ctx.set_name(job_full_name)
        if "env" in job.keys():
            job_ctx.print_esc("job env vars")
            for k,v in job["env"].items():
                job_ctx.print('export {0}="{1}"'.format(k,sanitize_var(v)))
        steps = job["steps"]
        proc_steps(steps,config, job_ctx)
        job_ctx.finish()
        ## fancier cases (matrix specs) not yet supported 

def proc_matrix_entry(steps, 
                      config,
                      matrix_entry_name,
                      env_vars,
                      ctx):
    ctx.print("#-------------------------------------")
    ctx.print_esc(tag = "matrix entry", txt = matrix_entry_name)
    ctx.print("#-------------------------------------")
    ctx.print_esc(tag = "azure global scope vars", txt = config["azure_vars"])
    ctx.print_esc("matrix env vars")
    for k,v in env_vars.items():
        ctx.print("export {0}={1}".format(k,sanitize_var(v)))
    ctx.print("")
    proc_steps(steps, config, ctx)

def proc_action_checkout(step, config, ctx):
    ctx.print("")
    ctx.print("#++++++++++++++++++++++++++++++++")
    ctx.print_esc("checkout")
    ctx.print_esc(step)
    ctx.print('echo ">start checkout"')
    ctx.print("date")
    ctx.print("git clone --recursive --depth=1 -b {0} {1} ".format(
                config["repo_branch"],
                config["repo_url"]))
    ctx.set_cwd(config["name"])
    ctx.print('echo ">end checkout"')
    ctx.print("date")
    ctx.print("#++++++++++++++++++++++++++++++++")

def proc_steps(steps, config, ctx):
    ctx.print("#-------------------------------------")
    ctx.print_esc("STEPS")
    ctx.print("#-------------------------------------")
    for s in steps:
        # we only process "uses:actions/checkout and run
        if "uses" in s.keys():
            # support checkout ...
            if s["uses"].count("actions/checkout") > 0:
                proc_action_checkout(s,config,ctx)
        elif "run" in s.keys():
            ctx.print("")
            ctx.print("#++++++++++++++++++++++++++++++++")
            ctx.print_esc(tag = "name", txt = s["name"])
            ctx.print_esc("script")
            ctx.print('echo ">start {0}"'.format(s["name"]))
            ctx.print("date")
            if not ctx.cwd is None:
                ctx.print("cd ~/{0}".format(ctx.cwd))
            lines = s["run"].strip().split("\n")
            ctx.print_esc("turn ON halt on error")
            ctx.print("set -e")
            for l in lines:
                ctx.print(l)
            ctx.print('echo ">end {0}"'.format(s["name"]))
            ctx.print("date")
            ctx.print("#++++++++++++++++++++++++++++++++")
        else:
            if "name" in s.keys():
                ctx.print_esc("STEP with name:{0} not SUPPORTED".format(s["name"]))
            else:
                ctx.print_esc("STEP not SUPPORTED")

def proc_config(config):
    if config["repo_branch"] == "<CURRENT>":
        rcode,rout = shexe("git rev-parse --abbrev-ref HEAD",ret_output=True,echo=True)
        if rcode == 0:
            config["repo_branch"] = rout.strip()
        else:
            print("[error finding current git branch]")
            sys.exit(-1)
    return config



def main():
    gactrep_yaml_files = glob.glob("../../.github/workflows/*yml")
    print(gactrep_yaml_files)
    config_yaml_file   = "gactemu-config.yaml"
    config = yaml.load(open(config_yaml_file), Loader=yaml.Loader)
    config = proc_config(config)
    for gactrep_yaml_file in gactrep_yaml_files:
        root   = yaml.load(open(gactrep_yaml_file), Loader=yaml.Loader)
        try:
            if os.path.isfile(root):
                root   = yaml.load(open(root), Loader=yaml.Loader)
        except:
            pass
        print(gactrep_yaml_file)
        #print(root)
        proc_root(root, config)

if __name__ == "__main__":
    main()
 
