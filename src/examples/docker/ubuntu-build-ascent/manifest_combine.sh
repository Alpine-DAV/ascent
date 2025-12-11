 docker manifest rm alpinedav/ascent-jupyter:0.9.5
 docker manifest create \
 --amend alpinedav/ascent-jupyter:0.9.5 \
 --amend alpinedav/ascent:ascent-ubuntu-24.04-develop-x86_64_2025-09-15-sha1c32d8 \
 --amend alpinedav/ascent:ascent-ubuntu-24.04-develop-arm64_2025-09-15-sha1c32d8
 docker manifest push alpinedav/ascent-jupyter:0.9.5

docker manifest rm alpinedav/ascent-jupyter:latest
 docker manifest create \
--amend alpinedav/ascent-jupyter:latest \
--amend alpinedav/ascent:ascent-ubuntu-24.04-develop-x86_64_2025-09-15-sha1c32d8 \
--amend alpinedav/ascent:ascent-ubuntu-24.04-develop-arm64_2025-09-15-sha1c32d8
docker manifest push alpinedav/ascent-jupyter:latest
#
docker manifest rm alpinedav/ascent:0.9.5
docker manifest create \
--amend alpinedav/ascent:0.9.5 \
--amend alpinedav/ascent:ascent-ubuntu-24.04-develop-x86_64_2025-09-15-sha1c32d8 \
--amend alpinedav/ascent:ascent-ubuntu-24.04-develop-arm64_2025-09-15-sha1c32d8
docker manifest push alpinedav/ascent:0.9.5
#
docker manifest rm alpinedav/ascent:latest
 docker manifest create \
--amend alpinedav/ascent:latest \
--amend alpinedav/ascent:ascent-ubuntu-24.04-develop-x86_64_2025-09-15-sha1c32d8 \
--amend alpinedav/ascent:ascent-ubuntu-24.04-develop-arm64_2025-09-15-sha1c32d8
docker manifest push alpinedav/ascent:latest
