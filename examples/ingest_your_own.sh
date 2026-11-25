# JSONL fields: source_id, doi, version, title, authors[{name,orcid}], date, abstract, body, references[{doi}], license
litmine ingest --input my_papers.jsonl --out work/snap
litmine index --snapshot work/snap --index work/idx --workers 4 --shards 8
litmine search "your question" --snapshot work/snap --k 5
