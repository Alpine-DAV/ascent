#docker manifest create \
#--amend alpinedav/ascent:0.9.3  \
#--amend alpinedav/ascent:ascent-ubuntu-22.04-x86_64_2024-05-23-sha988339 \
#--amend alpinedav/ascent:ascent-ubuntu-22.04-arm64_2024-05-23-sha988339
# docker manifest push alpinedav/ascent:0.9.3
#
docker manifest create \
--amend alpinedav/ascent:latest  \
--amend alpinedav/ascent:ascent-ubuntu-22.04-x86_64_2024-05-23-sha988339 \
--amend alpinedav/ascent:ascent-ubuntu-22.04-arm64_2024-05-23-sha988339
docker manifest push alpinedav/ascent:latest