import logging, sys, multiprocessing
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

if __name__ == '__main__':
    multiprocessing.freeze_support()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from corpus.build_corpus import build_corpus
    df, stats = build_corpus(
        pcap_dir=r"X:\DATA\UNSW\pcap files\pcaps 22-1-2015",
        label_csv=r"X:\DATA\UNSW\CSV Files\NUSW-NB15_GT.csv",
        tshark_bin=None,
        out_parquet=r"X:\ADNS\outputs\corpus\unsw_day1_nfstream.parquet",
        extractor="nfstream",
    )
    print(f"Done. {len(df):,} rows")
    print(f"  n_attack={stats.n_attack}  n_benign={stats.n_benign}  n_dropped={stats.n_dropped_unprocessable}")
    print(f"  label_rows: matched={stats.label_rows_matched}  unmatched={stats.label_rows_unmatched}")
