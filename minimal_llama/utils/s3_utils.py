import math
import os
import time
from pathlib import Path

import boto3
import hf_transfer
import torch
import torch.distributed


def print_rank_0(*args, **kwargs):
    """Print from rank 0 only"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def multiprocessing_starmap(func, args, num_processes=None):
    """Wrapper to allow for re-usable multiprocessing pools with `spawn` context handling
    Args:
        func (Callable): Function to call
        args (Iterable): Iterable of arguments to pass to `func`
        num_processes (int, optional): Number of processes to spawn. Defaults to `multiprocessing.cpu_count() - 1`
    """
    import multiprocessing
    num_processes = num_processes or (multiprocessing.cpu_count() - 1)
    with multiprocessing.get_context("spawn").Pool(processes=num_processes) as process_pool:
        process_pool.starmap(func, args)
        process_pool.terminate()
        process_pool.join()
        del process_pool


def _upload(
    file_path: str,
    s3_key: str,
    chunk_size: int = 104_857_600,
    max_files: int = 64,
    parallel_failures: int = 63,
    max_retries: int = 5,
):
    """Upload local file to S3 using `hf_transfer` library
    Args:
        file_path (str): Local filename to upload
        s3_key (str): S3 key to upload to. E.g. `s3://bucket-name/path/to/file`
        chunk_size (int, optional): Chunk size to use for multipart upload.
            Defaults to 100MiB = 104_857_600
        max_files (int, optional):  Number of open file handles, which determines
            the maximum number of parallel downloads. Defaults to 64
        parallel_failures (int, optional): Number of maximum failures of different
            chunks in parallel (cannot exceed max_files). Defaults to 63
        max_retries (int, optional): Number of retries for each chunk. Defaults to 5
    """
    s3 = boto3.client('s3')
    bucket = s3_key.split("s3://")[1].split("/")[0]
    key = s3_key.split(bucket)[1].lstrip("/")

    # 1. Init multipart upload and obtain unique upload identifier
    upload = s3.create_multipart_upload(
        ACL="bucket-owner-full-control",
        Bucket=bucket,
        Key=key,
    )
    upload_id = upload["UploadId"]

    # 2. Generate presigned URLs for each part
    file_size = os.stat(file_path).st_size
    urls = []
    nb_parts = math.ceil(file_size / chunk_size)
    for part_number in range(1, nb_parts + 1):
        params = {
            "Bucket": bucket,
            "Key": key,
            "PartNumber": part_number,
            "UploadId": upload_id,
        }
        urls.append(
            s3.generate_presigned_url(
                ClientMethod="upload_part", Params=params, ExpiresIn=86400
            )
        )

    # 3. Upload parts in parallel
    responses = hf_transfer.multipart_upload(
        file_path=file_path,
        parts_urls=urls,
        chunk_size=chunk_size,
        max_files=max_files,
        parallel_failures=parallel_failures,
        max_retries=max_retries,
    )

    # 4. Complete multipart upload request with ETag values
    etag_with_parts = []
    for part_number, header in enumerate(responses):
        etag = header.get("etag")
        etag_with_parts.append({"ETag": etag, "PartNumber": part_number + 1})
    parts = {"Parts": etag_with_parts}
    s3.complete_multipart_upload(
        Bucket=bucket, Key=key, MultipartUpload=parts, UploadId=upload_id
    )


def upload_checkpoint(local_fol_path, remote_fol_path):
    local_checkpoint_path = os.path.join(os.path.abspath(neox_args.save), get_checkpoint_tag(iteration))
    local_checkpoint_list = sorted(filter(
        lambda x: os.path.isfile(x),
        [str(p) for p in Path(local_checkpoint_path).rglob("*")],
    ))
    remote_checkpoint_path = os.path.join(
        neox_args.s3_path, os.path.basename(neox_args.save), get_checkpoint_tag(iteration))
    remote_checkpoint_list = [
        os.path.join(remote_checkpoint_path, os.path.relpath(local_checkpoint, local_checkpoint_path))
        for local_checkpoint in local_checkpoint_list
    ]
    inputs = zip(local_checkpoint_list, remote_checkpoint_list, [neox_args.s3_chunk_size] * len(local_checkpoint_list))

    print_rank_0(f"[RANK {torch.distributed.get_rank()}] Uploading checkpoint `{local_checkpoint_path}` to `{remote_checkpoint_path}`...")
    start = time.time()
    multiprocessing_starmap(_upload, inputs)
    total_time = time.time() - start
    print_rank_0(f"[RANK {torch.distributed.get_rank()}] Uploaded checkpoint `{local_checkpoint_path}` to `{remote_checkpoint_path}` in {total_time:.2f}s")