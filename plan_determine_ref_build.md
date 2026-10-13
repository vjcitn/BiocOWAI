# Plan: Determine Reference Genome Build for VCF File

## Goal

Determine the reference genome build (e.g., GRCh37/hg19 or GRCh38/hg38) used for the genotype calls in:

```         
1kGP_high_coverage_Illumina.chr17.filtered.SNV_INDEL_SV_phased_panel.vcf.gz
```

------------------------------------------------------------------------

## Step 1: Locate the File in Workspace Tables

Use the `anvil_file` table to find the file's storage details (GS URI, DRS URI, etc.).

### R (AnVIL)

``` r
library(AnVIL)
library(AnVILGCP)

avworkspace_namespace("landmarkanvil2")
avworkspace_name("BiocOWAI")

# Query the anvil_file table for the target file
file_info <- avtable("anvil_file") |>
  filter(str_detect(file_name, "1kGP_high_coverage_Illumina.chr17.filtered"))

print(file_info)
```

### Python (firecloud/fiss)

``` python
from firecloud.fiss import fapi

# List entity types to confirm 'anvil_file' exists
# Then query for the specific file
response = fapi.get_entities_with_type("landmarkanvil2", "BiocOWAI", "anvil_file")
# Filter for the target file by file_name
```

------------------------------------------------------------------------

## Step 2: Access the File in Google Cloud Storage

Once the GS URI is obtained, the file can be accessed via `gsutil` or directly in Python.

``` bash
# Example (replace with actual GS URI from Step 1)
gsutil ls gs://bucket-name/path/to/1kGP_high_coverage_Illumina.chr17.filtered.SNV_INDEL_SV_phased_panel.vcf.gz
```

------------------------------------------------------------------------

## Step 3: Examine the VCF Header for Reference Build Information

The VCF header contains metadata about the reference build. Key fields to inspect:

| Header Field | Description |
|----|----|
| `##reference=` | Directly states the reference genome (e.g., `GRCh37`, `GRCh38`, `hg19`, `hg38`) |
| `##contig=` lines | Contig lengths can be compared against known builds to infer the build |
| `##fileDate=` | Date of file creation (contextual) |
| `##reference` MD5 checksum | If present, can be used to verify the exact reference version |

### Method A: Inspect Header with `zgrep` (command line)

``` bash
# Decompress and grep the header (lines starting with ##)
zgrep '^##' gs://bucket-name/path/to/1kGP_high_coverage_Illumina.chr17.filtered.SNV_INDEL_SV_phased_panel.vcf.gz | head -50
```

Look for:

```         
##reference=GRCh37
# or
##reference=GRCh38
```

### Method B: Inspect Header with Python

``` python
import gzip

gs_uri = "gs://bucket-name/path/to/1kGP_high_coverage_Illumina.chr17.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"

# Download header using gsutil pipe, or use google-cloud-storage
from google.cloud import storage

client = storage.Client()
bucket_name, blob_name = gs_uri.replace("gs://", "").split("/", 1)
blob = client.bucket(bucket_name).blob(blob_name)

# Read only the header (first ~200KB)
header = blob.download_as_text(maximum_bytes=200_000)

# Parse header lines
for line in header.split("\n"):
    if line.startswith("##reference"):
        print(f"Reference: {line}")
    if line.startswith("##contig"):
        print(f"Contig: {line}")
    if line.startswith("#CHROM"):
        print(f"Column header: {line}")
        break
```

------------------------------------------------------------------------

## Step 4: Cross-Check with Contig Lengths (if `##reference=` is absent)

If the `##reference=` line is missing, compare contig lengths against known builds:

| Chromosome | GRCh37 (hg19) Length | GRCh38 (hg38) Length |
|------------|----------------------|----------------------|
| chr1       | 248,956,422          | 248,956,422          |
| chr17      | **83,257,441**       | **83,228,254**       |
| chrX       | 156,040,895          | 156,040,895          |

### Python: Extract and Compare chr17 Length

``` python
import re

# Parse contig lines from header
contig_lengths = {}
for line in header.split("\n"):
    if line.startswith("##contig"):
        match = re.search(r'id=([^,]+),length=(\d+)', line)
        if match:
            contig_lengths[match.group(1)] = int(match.group(2))

# Check chr17 length
chr17_len = contig_lengths.get("chr17", contig_lengths.get("17", "N/A"))
print(f"chr17 length in VCF: {chr17_len}")

if chr17_len == 83257441:
    print("Build: GRCh37 (hg19)")
elif chr17_len == 83228254:
    print("Build: GRCh38 (hg38)")
else:
    print("Build could not be determined from chr17 length alone.")
```

------------------------------------------------------------------------

## Step 5: Confirm with File Metadata (Optional)

Check the `anvil_file` table for any additional metadata:

``` r
# In R
file_info |> select(file_name, file_id, uri, data_modality)
```

Look for: - `data_modality` — should be `VCF` or `variant call format` - Any `pfb:reference_assembly` attribute in related tables (e.g., `sequencing_dataset`)

------------------------------------------------------------------------

## Summary of Evidence Sources

| Source | What It Tells You |
|----|----|
| `##reference=` in VCF header | Definitive answer (if present) |
| Contig lengths in VCF header | Strong evidence via comparison |
| `pfb:reference_assembly` attribute | Metadata from ingestion pipeline |
| `anvil_dataset` table | May contain title/description with build info |
