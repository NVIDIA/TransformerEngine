# clean TransformerEngine
pip uninstall transformer_engine -y
rm -rf build
rm -rf transformer_engine.egg-info
rm -f transformer_engine/transformer_engine_torch.cpython-310-x86_64-linux-gnu.so
rm -f libtransformer_engine.so
rm -f transformer_engine_torch.cpython-310-x86_64-linux-gnu.so

# install TransformerEngine
export NVTE_FRAMEWORK=pytorch
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:/usr/local/musa/lib:/usr/local/musa/mudnn/lib

pip install --no-build-isolation -v . 2>&1 | tee install.log
