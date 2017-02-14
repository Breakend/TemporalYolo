import sys, os
sys.path.append(os.path.abspath("../../"))

from shared_utils.data import BatchLoader

from nose.tools import assert_equals


def test_generating_batches():
    batch_loader = BatchLoader("tests/SAMPLE_DATA", seq_len=2, batch_size=2, folders_to_use=["TEST1", "TEST2", "TEST3"])

    assert_equals(len(batch_loader.batches), 30)
