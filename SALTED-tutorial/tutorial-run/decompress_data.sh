cat data_part_* > data.tar.gz && tar -xzvf data.tar.gz
tar -xzvf data_trainset.tar.gz
# rm data_part_* data.tar.gz data_trainset.tar.gz
mkdir -p pred_data