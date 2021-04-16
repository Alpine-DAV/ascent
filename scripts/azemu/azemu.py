"""
azemu.py (Arizona Emu Wrangler)

Digests azure-pipelines.yaml specs and generates scripts to run steps 
locally using docker.


"""

import yaml

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

        print("[creating: {0}".format(self.launch_file()))
        f= open(self.launch_file(),"w",newline='\n')
        f.write(self.gen_launch())

    def script_file(self):
        return "AZEMU-SCRIPT-" + self.name + ".sh"

    def launch_file(self):
        return "AZEMU-LAUNCH-" + self.name + ".sh"

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
        res += "docker stop azemu_exec\n"
        res += "docker rm  azemu_exec\n"
        res += "docker run --name azemu_exec -t -d {0}\n".format(self.container)
        #res += 'docker exec azemu_exec sh -c "apt-get update && env DEBIAN_FRONTEND=\"noninteractive\" TZ=\"America/Los_Angeles\" apt-get install -y sudo software-properties-common git"\n'
        res += 'docker exec azemu_exec sh -c "apt-get update && env apt-get install -y sudo"\n'
        res += 'docker exec azemu_exec sh -c "useradd -ms /bin/bash -G sudo user && echo \\"user:docker\\\" | chpasswd"\n'
        res += 'docker exec azemu_exec sh -c "echo \\"user ALL=(root) NOPASSWD:ALL\\\" > /etc/sudoers.d/user &&  chmod 0440 /etc/sudoers.d/user"\n'
        res += 'docker cp {0} azemu_exec://home/user/\n'.format(self.script_file())
        res += 'docker exec -u user azemu_exec  sh -c "echo [AZEMU EXEC SCRIPT]!"\n'
        res += 'docker exec -u user --workdir=//home/user/ -i azemu_exec ./{0}\n'.format(self.script_file())
        return res

    def __str__(self):
        res =""
        if self.name != "":
            res += "# [[{0}]]\n".format(self.name)
        res += self.txt
        return res


def azure_var_sub(txt, azure_vars):
    pat ="${{ variables."
    if txt.find(pat) >= 0:
        st =txt.find(pat)
        l = len(pat)
        ed = txt.find("}}")
        key = txt[st+l:ed]
        key = key.strip()
        txt = txt.replace(pat + key + " }}",azure_vars[key])
    return txt

def proc_root(tree, config):
    azure_vars = {}
    for k,v in tree.items():
        if k == "variables":
            config["azure_vars"] = v
        if k == "stages":
            proc_stages(v, config)

def proc_stages(tree, config):
    for v in tree:
        proc_stage(v ,config, v["stage"])

def proc_stage(tree, config, stage_name):
    for job in tree["jobs"]:
        job_ctx = CTX()
        job_ctx.print_esc(tag = "stage", txt = stage_name)
        job_name = job["job"]
        if "container" in job.keys():
             job_ctx.set_container(azure_var_sub(job["container"],
                                   config["azure_vars"]))
        else:
            job_ctx.set_container(config["default_container"])
        job_ctx.print_esc(tag = "job", txt = job_name)
        job_ctx.set_name("{0}-{1}".format(job_name,job["job"]))
        if "variables" in job.keys():
            job_ctx.print_esc("job env vars")
            for k,v in job["variables"].items():
                job_ctx.print("export {0}={1}".format(k,v))
        steps = job["steps"]
        if "strategy" in job.keys():
            if "matrix" in job["strategy"].keys():
                for k,v in job["strategy"]["matrix"].items():
                    matrix_ctx = CTX(job_ctx)
                    # change container and name from base ctx
                    matrix_ctx.set_container(job_ctx.container)
                    matrix_entry_name = k
                    matrix_ctx.set_name("{0}-{1}-{2}".format(stage_name,
                                                          job_name,
                                                          matrix_entry_name))
                    env_vars = v
                    proc_matrix_entry(steps,
                                      config,
                                      matrix_entry_name,
                                      env_vars,
                                      matrix_ctx)
                    matrix_ctx.finish()
        else:
            proc_steps(steps,config, job_ctx)
            job_ctx.finish()

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
        ctx.print("export {0}={1}".format(k,v))
    ctx.print("")
    proc_steps(steps, config, ctx)

def proc_steps(steps, config, ctx):
    ctx.print("#-------------------------------------")
    ctx.print_esc("STEPS")
    ctx.print("#-------------------------------------")
    for s in steps:
        # we only process checkout and script
        if "checkout" in s.keys():
            ctx.print("")
            ctx.print("#++++++++++++++++++++++++++++++++")
            ctx.print_esc("checkout")
            ctx.print_esc(s)
            ctx.print('echo ">start checkout"')
            ctx.print("date")
            ctx.print("git clone --recursive --depth=1 -b {0} {1} ".format(
                      config["repo_branch"],
                      config["repo_url"]))
            ctx.set_cwd(config["name"])
            ctx.print('echo ">end checkout"')
            ctx.print("date")
            ctx.print("#++++++++++++++++++++++++++++++++")
        elif "script" in s.keys():
            ctx.print("")
            ctx.print("#++++++++++++++++++++++++++++++++")
            ctx.print_esc(tag = "displayName", txt = s["displayName"])
            ctx.print_esc("script")
            ctx.print('echo ">start {0}"'.format(s["displayName"]))
            ctx.print("date")
            if not ctx.cwd is None:
                ctx.print("cd ~/{0}".format(ctx.cwd))
            lines = s["script"].strip().split("\n")
            # turn off errors
            ctx.print_esc("turn OFF halt on error")
            ctx.print("set +e")
            for l in lines[:-1]:
                ctx.print(l)
            # azure reports errors on last command
            ctx.print_esc("turn ON halt on error")
            ctx.print("set -e")
            ctx.print(lines[-1])
            ctx.print('echo ">end {0}"'.format(s["displayName"]))
            ctx.print("date")
            ctx.print("#++++++++++++++++++++++++++++++++")
        else:
            if "displayName" in s.keys():
                ctx.print_esc("STEP with displayName:{0} not SUPPORTED".format(s["displayName"]))
            else:
                ctx.print_esc("STEP not SUPPORTED")

def main():
    azurep_yaml_file = "azure-pipelines.yml"
    config_yaml_file = "azemu-config.yaml"
    root   = yaml.load(open(azurep_yaml_file), Loader=yaml.Loader)
    config = yaml.load(open(config_yaml_file), Loader=yaml.Loader)
    proc_root(root, config)


if __name__ == "__main__":
    main()

