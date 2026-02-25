"""
Simple test script to verify caching functionality in OpenCellDataset.
"""
import os
import time
from data.opencell.dataset import OpenCellDataset
from data.opencell.transforms import get_opencell_train_transforms


def test_caching():
    """Test the caching functionality with different cache rates."""

    # Path to CSV (adjust if needed)
    code_base_path = '/path/to/repository/'
    csv_path = os.path.join(code_base_path, "data/opencell_metadata/train.csv")

    print("=" * 80)
    print("Testing OpenCellDataset Caching Functionality")
    print("=" * 80)

    # Test 1: No caching (cache_rate=0.0)
    print("\n[Test 1] Creating dataset with cache_rate=0.0")
    start_time = time.time()
    dataset_no_cache = OpenCellDataset(
        csv_path=csv_path,
        split='train',
        transform=None,
        cache_rate=0.0,
        num_workers=4
    )
    init_time_no_cache = time.time() - start_time
    stats = dataset_no_cache.get_cache_stats()
    print(f"  Initialization time: {init_time_no_cache:.2f}s")
    print(f"  Cache stats: {stats}")
    assert stats['cached_images'] == 0, "Expected 0 cached images with cache_rate=0.0"
    print("  ✓ PASSED: No images cached")

    # Test 2: Partial caching (cache_rate=0.1)
    print("\n[Test 2] Creating dataset with cache_rate=0.1")
    start_time = time.time()
    dataset_partial_cache = OpenCellDataset(
        csv_path=csv_path,
        split='train',
        transform=None,
        cache_rate=0.1,
        num_workers=4
    )
    init_time_partial = time.time() - start_time
    stats = dataset_partial_cache.get_cache_stats()
    print(f"  Initialization time: {init_time_partial:.2f}s")
    print(f"  Cache stats: {stats}")
    expected_cached = int(stats['total_images'] * 0.1)
    assert stats['cached_images'] == expected_cached, f"Expected {expected_cached} cached images"
    print(f"  ✓ PASSED: {stats['cached_images']} images cached")

    # Test 3: Full caching (cache_rate=1.0) - only if dataset is small
    if dataset_partial_cache.get_cache_stats()['total_images'] <= 100:
        print("\n[Test 3] Creating dataset with cache_rate=1.0")
        start_time = time.time()
        dataset_full_cache = OpenCellDataset(
            csv_path=csv_path,
            split='train',
            transform=None,
            cache_rate=1.0,
            num_workers=4
        )
        init_time_full = time.time() - start_time
        stats = dataset_full_cache.get_cache_stats()
        print(f"  Initialization time: {init_time_full:.2f}s")
        print(f"  Cache stats: {stats}")
        assert stats['cached_images'] == stats['total_images'], "Expected all images cached"
        print("  ✓ PASSED: All images cached")
    else:
        print("\n[Test 3] Skipping full cache test (dataset too large)")

    # Test 4: Verify data loading works
    print("\n[Test 4] Testing data loading from cached dataset")
    try:
        # Load a cached item
        data_0 = dataset_partial_cache[0]
        print(f"  Loaded item 0 (cached): shape={data_0['image'].shape}")

        # Load a non-cached item
        last_idx = len(dataset_partial_cache) - 1
        data_last = dataset_partial_cache[last_idx]
        print(f"  Loaded item {last_idx} (not cached): shape={data_last['image'].shape}")

        print("  ✓ PASSED: Data loading works for both cached and non-cached items")
    except Exception as e:
        print(f"  ✗ FAILED: {str(e)}")
        raise

    # Test 5: Test with transforms
    print("\n[Test 5] Testing with transforms")
    transform = get_opencell_train_transforms(flip_prob=0.5, rotate_prob=0.5)
    dataset_with_transform = OpenCellDataset(
        csv_path=csv_path,
        split='train',
        transform=transform,
        cache_rate=0.1,
        num_workers=4
    )
    try:
        data = dataset_with_transform[0]
        print(f"  Loaded transformed item: shape={data['image'].shape}, dtype={data['image'].dtype}")
        print("  ✓ PASSED: Transforms work with caching")
    except Exception as e:
        print(f"  ✗ FAILED: {str(e)}")
        raise

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)


if __name__ == '__main__':
    test_caching()
