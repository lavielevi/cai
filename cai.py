import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# -------------------- Constants --------------------

AA_TO_CODONS: dict[str, list[str]] = {
    "Phe": ["TTT", "TTC"],
    "Leu": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "Ile": ["ATT", "ATC", "ATA"],
    "Met": ["ATG"],
    "Val": ["GTT", "GTC", "GTA", "GTG"],
    "Ser": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "Pro": ["CCT", "CCC", "CCA", "CCG"],
    "Thr": ["ACT", "ACC", "ACA", "ACG"],
    "Ala": ["GCT", "GCC", "GCA", "GCG"],
    "Tyr": ["TAT", "TAC"],
    "Stop": ["TAA", "TAG", "TGA"],
    "His": ["CAT", "CAC"],
    "Gln": ["CAA", "CAG"],
    "Asn": ["AAT", "AAC"],
    "Lys": ["AAA", "AAG"],
    "Asp": ["GAT", "GAC"],
    "Glu": ["GAA", "GAG"],
    "Cys": ["TGT", "TGC"],
    "Trp": ["TGG"],
    "Arg": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "Gly": ["GGT", "GGC", "GGA", "GGG"],
}

# Build the 64-codon lookup dictionary
STANDARD_GENETIC_CODE = {}
for aa, codons in AA_TO_CODONS.items():
    for codon in codons:
        STANDARD_GENETIC_CODE[codon] = aa

# Keep only the 18 amino acids that have >= 2 synonymous codons
# Exclude STOP, Met, Trp
DEGENERATE_AA_TO_CODONS = {}
for aa, codons in AA_TO_CODONS.items():
    if aa != "Stop" and len(codons) > 1:
        DEGENERATE_AA_TO_CODONS[aa] = codons

# Build flat, ordered list of the 59 codons (for numpy)
DEGENERATE_CODONS = []
for aa, codons in DEGENERATE_AA_TO_CODONS.items():
    for codon in codons:
        DEGENERATE_CODONS.append(codon)

# -------------------- End of constants --------------------

class CodonAdaptationIndex:
    def __init__(self, degenerate_aa_to_codons = DEGENERATE_AA_TO_CODONS):
        self.degenerate_aa_to_codons = degenerate_aa_to_codons
        self.codons = []
        self.codon_to_idx ={}

        idx = 0
        for aa, codons in self.degenerate_aa_to_codons.items():
            for codon in codons:
                self.codons.append(codon)
                self.codon_to_idx[codon] = idx
                idx += 1
        self.weights: np.ndarray | None = None
        self.log_weights: np.ndarray | None = None

    def _sequences_to_count_matrix(self, sequences) -> np.ndarray:
        seq_list = list(sequences)
        num_genes = len(seq_list)
        num_codons = len(self.codons)

        # Matrix C of shape (N_genes, 59)
        counts = np.zeros((num_genes, num_codons), dtype=np.float64)

        for row_idx, seq in enumerate(seq_list):
            for pos in range(0, len(seq), 3):
                codon = seq[pos : pos + 3]
                # Only count the degen. codons. Ignore Met, Trp, STOP
                if codon in self.codon_to_idx:
                    col_idx = self.codon_to_idx[codon]
                    counts[row_idx, col_idx] += 1.0
        return counts

    def fit(self, ref_sequences):
        # Matrix of shape (N, 59) : N is a subset of total genes
        counts = self._sequences_to_count_matrix(ref_sequences)
        
        # X_ij. Shape (59,) counts per codon
        total_counts = counts.sum(axis=0)
        for i in range(len(total_counts)):
            if total_counts[i] == 0:
                total_counts[i] = 0.5

        # w_ij. Shape (59,) normalized counts based on most popular codon count
        weights = np.zeros(len(self.codons), dtype=np.float64)

        for aa, codons in self.degenerate_aa_to_codons.items():
            indices = []
            for codon in codons:
                indices.append(self.codon_to_idx[codon])
            
            popular_codon_count = max(total_counts[i] for i in indices)
            for i in indices:
                weights[i] = total_counts[i] / popular_codon_count
        
        self.weights = weights
        self.log_weights = np.log(weights)
        return self

    def score(self, target_sequences):
        if self.log_weights is None:
            raise RuntimeError()
        
        # Matrix of shape (N, 59) counts per codon
        counts = self._sequences_to_count_matrix(target_sequences)
        cai_scores = []

        # Calculation: (1/L) \sum_{i=0}^58 (x_i * w_i)
        for gene_counts in counts:
            L = gene_counts.sum()

            if L == 0:
                cai_scores.append(0)
                continue
            
            weighted_log_sum = 0.0
            for i in range(len(self.codons)):
                weighted_log_sum += gene_counts[i] * self.log_weights[i]

            gene_cai = np.exp(weighted_log_sum / L)
            cai_scores.append(gene_cai)
        
        return np.array(cai_scores, dtype=np.float64)


def load_and_clean_yeast_xls(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(
        filepath,
        engine="calamine",
        na_values=["nan", "NaN", "NA", ""]
    )

    # Rename duplicate ORF label. Col A is gene ID, col G is DNA sequence (CDS)
    df = df.rename(columns={"ORF": "ORF_ID", "ORF.1": "CDS"})

    # Ensure expression cols are numeric floats
    numeric_cols = ["PA1", "PA2", "PA3", "mRNA"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean DNA seq. 
    df = df.dropna(subset=["CDS"]).copy()
    df["CDS"] = df["CDS"].astype(str).str.strip().str.upper()

    # Keep only valid coding seq. divisible by 3
    valid_rows = []
    valid_bases = {"A", "C", "T", "G"}
    for seq in df["CDS"]:
        is_triplet = len(seq) >= 3 and (len(seq) % 3 == 0)
        has_valid_chars = set(seq).issubset(valid_bases)
        valid_rows.append(is_triplet and has_valid_chars)
    df = df[valid_rows].reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = load_and_clean_yeast_xls("./yeast_parameters_table.xls")
    
    # Sort by "PC1" from highest to lowest and take the top 5% of genes
    top_5pc_PA1 = (
        df.dropna(subset=["PA1"])
        .sort_values(by="PA1", ascending=False)
        .head(294)
    )

    cai = CodonAdaptationIndex()
    cai.fit(top_5pc_PA1["CDS"])

    cai_scores = cai.score(df["CDS"])

    df["CAI"] = cai_scores
    
    train_orfs = set(top_5pc_PA1["ORF_ID"])

    # Correlation calculation (log Pearson's)
    print("Correlation between CAI scores and protein abundance measurements:")
    for pa in ["PA1", "PA2", "PA3"]:
        # Drop rows missing CAI or this PA measurement. Keep positive PA for log10
        valid = df.dropna(subset=["CAI", pa])
        valid = valid[valid[pa] > 0]

        r, p = pearsonr(valid["CAI"], np.log10(valid[pa]))
        print(f"{pa}:(n = {len(valid)}) r = {r:.4f}, p = {p:.2e}")

        if pa == "PA1":
            unseen = valid[~valid["ORF_ID"].isin(train_orfs)]
            r_u, p_u = pearsonr(unseen["CAI"], np.log10(unseen[pa]))
            print(f"{pa}: (excluding training data. n = {len(unseen)}), r = {r_u:.4f}, p = {p_u:.2e}")
        

    # Dot Plots
    fig, axes = plt.subplots(1, 3, figsize=(16,5))

    for ax, pa in zip(axes, ["PA1", "PA2", "PA3"]):
        valid = df.dropna(subset=["CAI", pa])
        valid = valid[valid[pa] > 0]

        x = valid["CAI"].to_numpy()
        y = np.log10(valid[pa].to_numpy())
        r, p = pearsonr(x, y)

        ax.scatter(x, y, s=12, alpha=0.35, color="blue", edgecolors="none", label="Genes")
        slope, intercept = np.polyfit(x, y, 1)
        x_line  = np.linspace(x.min(), x.max(), 200)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="red",
            linewidth=2,
            label=f"Fit: r = {r:.3f} (n={len(valid)})",
        )

        ax.set_title(f"CAI vs. log10({pa})")
        ax.set_xlabel("CAI")
        ax.set_ylabel(f"log10({pa})")
        ax.grid(True, linestyle="--", alpha=0.3)
    
    fig.suptitle("Yeast CAI vs. Log10 Protein Abundance (PA1, PA2, PA3)")
    plt.show()
