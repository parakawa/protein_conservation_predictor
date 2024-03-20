# protein_conservation_predictor

We use the Kibby repository to calculate the conservation score per residue for the curated dataset (35871 sequences) Estimated time: 75min

`python3 conservation_from_fasta.py curated_dataset/sequences.fasta output_results_curated.csv -model esm2_t6_8M_UR50D`

Data:

[output_results_curated_esm2_t6_8M_UR50D](https://drive.google.com/file/d/1bnZlHD21NOVQEw0CAsH-1FpAZbLWdsHB/view?usp=sharing).csv

esm-extract esm2_t33_650M_UR50D curated_dataset/sequences.fasta curated_dataset/example_embeddings_esm2 --repr_layers 0 32 33 --include mean per_tok

esm-extract esm2_t6_8M_UR50D curated_dataset/sequences.fasta curated_dataset/example_embeddings_esm2 --repr_layers 0 5 6 --include mean per_tok

esm-extract esm2_t6_8M_UR50D sample_data/sequences.fasta sample_data/example_embeddings_esm2 --repr_layers 0 5 6 --include mean per_tok

esm-extract esm2_t6_8M_UR50D curated_dataset/reduced_input.fasta curated_dataset/example_embeddings_esm2_reduced_input --repr_layers 0 5 6 --include mean per_tok
