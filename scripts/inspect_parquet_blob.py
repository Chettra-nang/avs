#!/usr/bin/env python3
"""Inspect Parquet files produced by the collector to print schema and a few
sample records, focusing on binary blob columns and their associated shape/dtype
fields. Use this to debug messages like "Cannot handle this data type: (1, 1, 64), |u1".

Usage:
  python3 AVs/scripts/inspect_parquet_blob.py /path/to/transitions.parquet --rows 5
"""
import sys
import argparse
import json
import pyarrow.parquet as pq
import numpy as np


def safe_repr_blob(x):
    try:
        t = type(x)
        if hasattr(x, 'tobytes'):
            b = x.tobytes()
            return f"{t.__name__}, len={len(b)}"
        if isinstance(x, (bytes, bytearray, memoryview)):
            return f"{type(x).__name__}, len={len(x)}"
        return f"{t.__name__}"
    except Exception as e:
        return f"<repr-error {e}>"


def inspect(parquet_path, rows=5):
    print('Opening:', parquet_path)
    pf = pq.ParquetFile(parquet_path)
    print('Row groups:', pf.num_row_groups)
    print('Schema:')
    print(pf.schema)

    # read a small table
    table = pf.read_row_group(0) if pf.num_row_groups > 0 else pf.read()
    df = table.to_pandas()[:rows]

    print('\nColumns:')
    for c in df.columns:
        print('-', c, 'dtype:', df[c].dtype)

    print('\nSample rows:')
    for i, row in df.iterrows():
        print('\n--- row', i, '---')
        for k, v in row.items():
            if 'blob' in k or 'grayscale' in k or k.endswith('_blob'):
                print(k, '->', safe_repr_blob(v))
            elif 'shape' in k or 'dtype' in k:
                print(k, '->', repr(v))
            else:
                # keep the rest compact
                s = repr(v)
                if len(s) > 200:
                    s = s[:200] + '...'
                print(k, '->', s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parquet', help='Parquet file path to inspect')
    parser.add_argument('--rows', type=int, default=5)
    args = parser.parse_args()
    inspect(args.parquet, rows=args.rows)


if __name__ == '__main__':
    main()
