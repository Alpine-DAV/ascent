# Copyright 2019 Lawrence Livermore National Security, LLC and other
# Bridge Kernel Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

# parts of this are from nb_connect_existing
from __future__ import print_function
import socket
import os
import pexpect
from subprocess import Popen
import time

try:
    from subprocess import DEVNULL
except ImportError:
    DEVNULL = open("/dev/null", "w")

try:
    from zmq.ssh.tunnel import select_random_ports
except ImportError:
    def select_random_ports(num):
        ports = []
        socks = []
        for i in range(num):
            s = socket.socket()
            s.bind(('', 0))
            ports.append(s.getsockname()[1])
            socks.append(s)
        for s in socks:
            s.close()
        return ports


localhost = socket.gethostname()
connection_refused = "Connection refused"
unknown_host = "Are you sure you want to continue connecting"
auth_fail = "Authentication failed."
permission_denied = "Permission Denied."

host_check_opt = "-oStrictHostKeyChecking=no"
ssh_cmd = "ssh {host_check_opt} -S none -nT -{fwd_flg} {fwd_args} -p {ssh_port} {server} sleep {sleep}"
mrsh_cmd = "mrsh {server} {ssh_cmd}"
verify_ssh = "ssh {host_check_opt} -p {port} {server} true"
verify_mrsh = "mrsh {server} {ssh_cmd}"


# if use if False we use the host_check_opt which turns off
# strict host checking and stops "key cannot be verified"
# tunnel failures. would be good to make this option
# toggleable from request
def use_host_check(use):
    if not use:
        return host_check_opt
    return ""


class TunnelError(RuntimeError):
    pass


def ssh_tunnel(log, mode, ltransport, lport, rtransport, rport, server, user,
        ssh_port, check_hosts=False, sleep_duration=30, silent=True):
    """Open ssh tunnel from lport on localhost to rport on server

    ssh -{L|R} {<local port>|<local socket>}:{<host>:<remote port>|<remote socket>}

    <host> is always localhost for now.

    The local port is always local to the starting machine even though it is
    referred to as the remote in the ssh docs in the -R case.
    """
    if rtransport == "tcp":
        forwarding_args = "%s:%s:%s" % (lport, "localhost", rport)
    else:
        forwarding_args = "%s:%s" % (lport, rport)

    # remove the local ipc socket if it exists, might be
    # better to pick a unique name
    if ltransport == "ipc" and os.path.exists(lport):
        os.remove(lport)

    host_check_opt = use_host_check(check_hosts)

    if mode == "ssh":
        cmd = ssh_cmd.format(fwd_flg="L", fwd_args=forwarding_args, host_check_opt=host_check_opt,
                server=server, ssh_port=ssh_port, sleep=sleep_duration)
    elif mode == "mrsh":
        cmd = mrsh_cmd.format(server=server, ssh_cmd=ssh_cmd.format(fwd_flg="R", fwd_args=forwarding_args,
            host_check_opt=host_check_opt, server=localhost, ssh_port=ssh_port, sleep=sleep_duration))
    else:
        raise TunnelError("Unknown mode %s" % mode)

    if silent:
        args = dict(stdout=DEVNULL, stderr=DEVNULL, stdin=DEVNULL)
    else:
        args = {}

    log("starting ssh tunnel> %s" % cmd)
    # TODO: this can fail
    Popen(cmd.split(), close_fds=True, preexec_fn=os.setpgrp, **args)


def try_ssh(log, server, port, env, check_hosts=False, timeout=5):
    cmd = verify_ssh.format(port=port, server=server, host_check_opt=use_host_check(check_hosts))
    log("Testing ssh> %s" % cmd)
    with pexpect.spawn(cmd, env=env, timeout=timeout) as p:
        index = p.expect([pexpect.EOF, connection_refused, unknown_host, auth_fail, pexpect.TIMEOUT])
    #log("Got p: %s" % p)
    if index == 0:
        return True
        #if len(p.before) == 0:
        #    return True
        #else:
        #    raise TunnelError(p.before)
    elif index == 1:
        # try mrsh
        #raise TunnelError("Connection refused")
        return False
    elif index == 2:
        raise TunnelError("Host authenticity can't be established")
    elif index == 3:
        raise TunnelError("Authenication failed")
    elif index == 4:
        raise TunnelError("Timeout trying to tunnel to host")
    return False


def try_mrsh(log, server, port, env, check_hosts=False, timeout=5):
    cmd = verify_mrsh.format(server=server, ssh_cmd=verify_ssh.format(port=port, server=localhost,
        host_check_opt=use_host_check(check_hosts)))
    log("Testing mrsh> %s" % cmd)
    with pexpect.spawn(cmd, env=env, timeout=timeout) as p:
        index = p.expect([pexpect.EOF, connection_refused, permission_denied, pexpect.TIMEOUT])
    if index == 0:
        if len(p.before) == 0:
            return True
        else:
            raise TunnelError(p.before)
    elif index == 1:
        raise TunnelError("Premission Refused")
    elif index == 2:
        raise TunnelError("Unable to connect to localhost after mrsh to %s" % server)
    #elif index == 3:
    #    raise TunnelError("Authenication failed")
    elif index == 3:
        raise TunnelError("Timeout connecting back to localhost after mrsh to %s" % server)
    return False


def open_ssh_tunnel(log, config, server, ssh_port=622, timeout=5, ipc_wait_file=5):
    """Try to create ssh tunnels from localhost to server

    Try to create a a) passwordless ssh tunnel from localhost to server or b) munge rsh to server
    and create a passwordless reverse ssh tunnel from server to localhost.
    The transport protocol on localhost and server may differ if needed.
    """

    user = config["user"]

    # server must be visable for now
    try:
        server_info = socket.gethostbyaddr(server)
    except socket.herror:
        raise TunnelError("host %s is inaccessible" % server)
    except socket.gaierror as e:
        raise TunnelError(str(e))

    # make sure the kernel isn't on localhost
    if server_info[0] == "localhost":
        log("kernel on localhost - nothing to do")
        return

    # no gui password prompt
    env = os.environ.copy()
    env.pop("SSH_ASKPASS", None)

    if try_ssh(log, server, ssh_port, env):
        mode = "ssh"
    elif try_mrsh(log, server, ssh_port, env):
        mode = "mrsh"
    else:
        raise TunnelError("Unable to connect, tried ssh and mrsh")

    protocol = config["protocol"]

    # remote (r) ports are the ports for the machine hosting the kernel
    if protocol == "ipc":
        rport = config["uds"]
        lport = "%s-%s" % (rport, localhost)
        config["uds"] = lport
    elif protocol == "tcp":
        rport = config["port"]
        lport = select_random_ports(1)[0]
        config["port"] = lport
    else:
        raise TunnelError("Unsupported protocol %s" % protocol)

    log("attempting to create tunnels from %s@%s to %s@%s" % (protocol, localhost,
        protocol, server))

    ssh_tunnel(log, mode, ltransport=protocol, lport=lport,
            rtransport=protocol, rport=rport,
            server=server, user=user, ssh_port=ssh_port)

    if protocol == "ipc":
        while not os.path.exists(lport) and ipc_wait_file > 0:
            log("waiting for local ipc socket - %d" % ipc_wait_file)
            time.sleep(1)
            ipc_wait_file -= 1
        if not os.path.exists(lport):
            raise TunnelError("local ipc socket doesn't exist: %s" % lport)
    elif protocol == "tcp":
        time.sleep(2)
