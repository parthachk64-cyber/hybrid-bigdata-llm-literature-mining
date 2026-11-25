# Evidence-linked walkthrough (synthetic demo corpus)

    litmine demo                      # offline, no token
    litmine demo --encoder ollama:nomic-embed-text --backend ollama   # full profile (local models)

Question: *Which methods have hydrology and manufacturing used to detect abrupt shifts in sensor streams?*
The report shows (1) ranked papers with BM25 / semantic raw scores, (2) the passage behind each rank, (3) an attributed
synthesis whose every sentence is a verbatim span with passage id and character offsets, (4) methods per field with two
evidence passages, (5) a **verified citation chain** S30 -> S22 -> S05 with the citing sentence at each hop, (6) trend counts.
An unsupported question (`litmine synthesize "effect of lunar gravity on yeast fermentation"`) prints "No supporting passage found".
