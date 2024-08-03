# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

ARG BASE_IMAGE=ascent-ubuntu-22.04-x86_64_2024-08-03-sha2bec68
FROM alpinedav/ascent:${BASE_IMAGE}

# add our 'user' with password 'docker'
RUN useradd -ms /bin/bash -G sudo user && echo "user:docker" | chpasswd
# allow sudo w/o password
RUN echo "user ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/user && \
chmod 0440 /etc/sudoers.d/user
     
# run the rest as user
USER user
WORKDIR /home/user

# setup ssh keys for passwordless ssh to localhost
RUN mkdir -p ~/.ssh
RUN ssh-keygen -b 2048 -t rsa -f ~/.ssh/id_rsa -q -N ""
RUN cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
RUN chmod 0600 ~/.ssh/authorized_keys

# setup jupyter password
RUN mkdir /home/user/.jupyter/
# the password is:
#  learn
RUN echo "c.NotebookApp.password = 'sha1:9777986fd066:283f673e1a311e2d5ef58c174eaebf3e1cb536dd'" > /home/user/.jupyter/jupyter_notebook_config.py
# set bash as default shell
RUN echo "c.NotebookApp.terminado_settings = { 'shell_command': ['/bin/bash'] }" >> /home/user/.jupyter/jupyter_notebook_config.py

# disable announcements
RUN /uberenv_libs/spack_view/bin/python3 -m jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

# gen script that allows us to easily run jupyter
RUN cp /ascent_docker_setup_env.sh /home/user/ascent_docker_run_jupyter.sh
RUN echo "jupyter notebook --ip=\"0.0.0.0\" --no-browser" >> /home/user/ascent_docker_run_jupyter.sh

# gen script that allows us to easily run jupyter lab
RUN cp /ascent_docker_setup_env.sh /home/user/ascent_docker_run_jupyterlab.sh
RUN echo "jupyter lab --ip=\"0.0.0.0\" --no-browser" >> /home/user/ascent_docker_run_jupyterlab.sh

# shorten root path to tutorial and link proxies for easier acccess
# symlink proxies for easy access next to jupyter notebooks
RUN sudo ln -s /ascent/install/examples/ascent/proxies/ /ascent/install/examples/ascent/tutorial/ascent_intro/notebooks/
RUN sudo ln -s /ascent/install/examples/ascent/tutorial/ascent_intro/notebooks/ /tutorial

#make sure helpers are executable
RUN sudo chmod 777 /home/user/ascent_docker_run_jupyter.sh 
RUN sudo chmod 777 /home/user/ascent_docker_run_jupyterlab.sh
RUN sudo chmod 777 /ascent/install/examples/ascent/tutorial/ascent_intro/*/cleanup.sh

# open port 8888, for use by jupyter notebook http server
EXPOSE 8888

# launch jupyter lab
CMD sudo service ssh start && cd /tutorial/ && /home/user/ascent_docker_run_jupyterlab.sh
