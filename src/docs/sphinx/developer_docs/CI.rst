Docker Containers For CI Testing
================================

We use Azure Pipelines for CI testing Ascent's Pull Requests.

* `Ascent Azure DevOps Space <https://dev.azure.com/alpine-dav/Ascent/>`_


To speed up our CI testing we use Docker containers with pre-built third party libraries. These containers leverage our ``spack/uberenv`` third party build  process. The Docker files and build scripts used to create these containers are in ``scripts/ci/docker``. To update the containers (assuming you have Docker installed):

 * Run ``build_all.sh`` to build and tag new versions of the containers.

 The tags will include today's day and a short substring of the current git hash.
 Example Tag: ``alpinedav/ascent-ci:ubuntu-16-cuda-10.1-devel-tpls_2020-08-25-sha449ef8``


 * Run ``docker push <container-name>`` to push the container images to `Ascent's DockerHub Registry <https://hub.docker.com/orgs/alpinedav>`_.

  You will need to be logged into DockerHub to successfully push, the process may ask for your DockerHub username and password. Example Push Command: ``alpinedav/ascent-ci:ubuntu-16-cuda-10.1-devel-tpls_2020-08-25-sha449ef8``

 * To change which Docker Image is used by Azure, edit ``azure-pipelines.yml`` and change `container_tag` variable.

.. code-block:: yaml

  #####
  # TO USE A NEW CONTAINER, UPDATE TAG NAME HERE AS PART OF YOUR PR!
  #####

  variables:
    ubuntu_18_tag : alpinedav/ascent-ci:ubuntu-18-devel-tpls_2021-12-29-shadee81a
    ubuntu_18_cuda_tag : alpinedav/ascent-ci:ubuntu-18-cuda-10.1-devel-tpls_2021-12-29-shadee81a
    ubuntu_18_cuda_11_4_0_tag: alpinedav/ascent-ci:ubuntu-18.04-cuda-11.4.0-devel-tpls_2021-12-29-shadee81a
    ubuntu_20_10_tag: alpinedav/ascent-ci:ubuntu-20.10-devel-tpls_2021-12-29-shadee81a
    ubuntu_21_04_tag: alpinedav/ascent-ci:ubuntu-21.04-devel-tpls_2021-12-29-shadee81a
    ubuntu_21_10_tag: alpinedav/ascent-ci:ubuntu-21.10-devel-tpls_2021-12-29-shadee81a



When the PR is merged, the azure changes will be merged and PRs to develop will use now the new containers.

