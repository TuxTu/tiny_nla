import pyarrow.parquet as pq
t = pq.read_table('/tmp/nla_test_explained.parquet')
print(f'rows: {t.num_rows}')
for row in t.to_pylist():
    doc = row['doc_id']
    n = row['n_raw_tokens']
    text = row['detokenized_text_truncated']
    expl = row['api_explanation']
    print(f'=== doc={doc}  tokens={n} ===')
    print(f'TEXT: {text[-120:]}...')  # last 120 chars before cut point
    print(f'EXPL: {expl}')
    print()
