Please refer to the following link for how to use SALTED in combination with FHI-AIMS:

https://gitlab.com/FHI-aims-club/tutorials/fhi-aims-with-salted

Compressing the Dataset
======================

To compress the dataset folder ``data`` with ``tar -czvf`` and split it into smaller chunks, use the following command:

.. code-block:: bash

    tar -czvf data.tar.gz data && split -b 90M data.tar.gz data_part_ && rm data.tar.gz
    tar -czvf data_trainset.tar.gz overlaps/ projections/ coefficients/

To reconstruct the dataset from the split files, use:

.. code-block:: bash

    cat data_part_* > data.tar.gz && tar -xzvf data.tar.gz
    tar -xzvf data_trainset.tar.gz
