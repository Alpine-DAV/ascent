#docker manifest create \
#--amend alpinedav/ascent-jupyter:0.9.3 \
#--amend alpinedav/ascent-jupyter:ascent-jupyter-ubuntu-22.04-x86_64_2024-05-23-sha988339 \
#--amend alpinedav/ascent-jupyter:ascent-jupyter-ubuntu-22.04-arm64_2024-05-23-sha988339
# docker manifest push alpinedav/ascent-jupyter:0.9.3
docker manifest create \
--amend alpinedav/ascent-jupyter:latest \
--amend alpinedav/ascent-jupyter:ascent-jupyter-ubuntu-22.04-x86_64_2024-05-23-sha988339 \
--amend alpinedav/ascent-jupyter:ascent-jupyter-ubuntu-22.04-arm64_2024-05-23-sha988339
docker manifest push alpinedav/ascent-jupyter:latest