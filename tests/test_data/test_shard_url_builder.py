"""Tests for shard URL brace expansion and remote rewriting."""

from __future__ import annotations

import pytest

from stagerecon.data.streaming.shard_url_builder import (
    build_shard_urls,
    expand_brace_urls,
    rewrite_remote_url,
)


def test_expand_brace_urls_zero_padded():
    urls = expand_brace_urls("/data/shard-{000000..000002}.tar")
    assert urls == [
        "/data/shard-000000.tar",
        "/data/shard-000001.tar",
        "/data/shard-000002.tar",
    ]


def test_rewrite_s3_to_aws_pipe():
    url = rewrite_remote_url("s3://bucket/train/shard-000000.tar", s3_transport="pipe_aws")
    assert url.startswith("pipe:aws s3 cp ")
    assert "s3://bucket/train/shard-000000.tar" in url
    assert url.endswith(" -")


def test_rewrite_s3_rclone_and_raw():
    rclone = rewrite_remote_url("s3://b/x.tar", s3_transport="pipe_rclone")
    assert rclone.startswith("pipe:rclone cat ")
    assert rewrite_remote_url("s3://b/x.tar", s3_transport="raw") == "s3://b/x.tar"


def test_rewrite_gs_and_http():
    gs = rewrite_remote_url("gs://bucket/a.tar", gs_transport="pipe_gsutil")
    assert gs.startswith("pipe:gsutil cat ")
    assert rewrite_remote_url("https://ex.com/a.tar") == "https://ex.com/a.tar"
    curl = rewrite_remote_url("https://ex.com/a.tar", http_transport="pipe_curl")
    assert curl.startswith("pipe:curl -fsSL ")


def test_build_shard_urls_rewrites_s3_patterns():
    urls = build_shard_urls(
        {
            "shards": "s3://bucket/train/train-{000000..000001}.tar",
            "s3_transport": "pipe_aws",
        }
    )
    assert len(urls) == 2
    assert all(u.startswith("pipe:aws s3 cp ") for u in urls)
    assert "train-000000.tar" in urls[0]
    assert "train-000001.tar" in urls[1]


def test_build_shard_urls_split_specific_map():
    urls = build_shard_urls(
        {
            "shards": {
                "train": "/data/train/train-{000000..000001}.tar",
                "val": "/data/val/val-000000.tar",
            },
            "rewrite_remote": False,
        },
        split="train",
    )
    assert len(urls) == 2
    assert urls[0].endswith("train-000000.tar")
    assert urls[1].endswith("train-000001.tar")

    val_urls = build_shard_urls(
        {
            "shards": {
                "train": "/data/train/train-000000.tar",
                "val": "/data/val/val-000000.tar",
            },
            "rewrite_remote": False,
        },
        split="val",
    )
    assert len(val_urls) == 1
    assert val_urls[0].endswith("val-000000.tar")


def test_build_shard_urls_rejects_null_shards():
    with pytest.raises((ValueError, KeyError)):
        build_shard_urls({"shards": None})
