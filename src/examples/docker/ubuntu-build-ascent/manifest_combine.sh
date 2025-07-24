# docker manifest rm alpinedav/ascent-jupyter:0.9.4
# docker manifest create \
# --amend alpinedav/ascent-jupyter:0.9.4 \
# --amend alpinedav/ascent-jupyter:ascent-ubuntu-24.04-develop-arm64_2025-07-20-sha3e7678 \
# --amend alpinedav/ascent-jupyter:ascent-ubuntu-24.04-develop-x86_64_2025-07-20-sha3e7678
# docker manifest push alpinedav/ascent-jupyter:0.9.4

docker manifest rm alpinedav/ascent-jupyter:latest
 docker manifest create \
--amend alpinedav/ascent-jupyter:latest \
--amend alpinedav/ascent-jupyter:ascent-ubuntu-24.04-develop-arm64_2025-07-20-sha3e7678 \
--amend alpinedav/ascent-jupyter:ascent-ubuntu-24.04-develop-x86_64_2025-07-20-sha3e7678
docker manifest push alpinedav/ascent-jupyter:latest
#
docker manifest rm alpinedav/ascent:0.9.4
docker manifest create \
--amend alpinedav/ascent:0.9.4 \
--amend alpinedav/ascent:ascent-ubuntu-24.04-develop-arm64_2025-07-20-sha3e7678 \
--amend alpinedav/ascent:ascent-ubuntu-24.04-develop-x86_64_2025-07-20-sha3e7678
docker manifest push alpinedav/ascent:0.9.4
#
docker manifest rm alpinedav/ascent:latest
 docker manifest create \
--amend alpinedav/ascent:latest \
--amend alpinedav/ascent:ascent-ubuntu-24.04-develop-arm64_2025-07-20-sha3e7678 \
--amend alpinedav/ascent:ascent-ubuntu-24.04-develop-x86_64_2025-07-20-sha3e7678
docker manifest push alpinedav/ascent:latest