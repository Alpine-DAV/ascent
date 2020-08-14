Docker Containers For CI Testing
=================================

We use Azure Pipelines for CI testing Ascent's Pull Requests.

* `Ascent Azure DevOps Space <https://dev.azure.com/alpine-dav/Ascent/>`_


To speed up our CI testing we use Docker containers with pre-built third party libraries. These containers leverage our ``spack/uberenv`` third party build  process. The Docker files and build scripts used to create these containers are in ``scripts/ci/docker``. To update the containers (assuming you have Docker installed):

 * Run ``build_all.sh`` to build and tag new versions of the containers.
 * Run ``push_all.sh`` to push the newer containers to `Ascent's DockerHub Registry <https://hub.docker.com/orgs/alpinedav>`_.

  You will need to be logged into DockerHub to successfully push, the process may ask for your DockerHub username and password.



