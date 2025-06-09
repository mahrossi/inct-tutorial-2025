DEEPHE3_PATH=/home/zekunlou/Projects/py_packages/deeph_install/DeepH-E3
python ${DEEPHE3_PATH}/deephe3-train.py train.ini

# performance on my laptop with 8 cores:
#     - irreps_embed = 64x0e; irreps_mid = 64x0e+32x1o+16x2e+8x3o+8x4e
#     - 40 s per epoch
#     - 128 MB RAM during training (batchsize 64)


