# Extraction checked against an expert table

`T1_extraction_vs_table1.csv`: for the 12 T1 gold studies included by the final T1 run
(`20261108-204319-7daa7f`), the values the pipeline extracted for three fields are compared
with Table 1 of the expert review (Nazarian 2021, read from its PMC XML, PMC8433941):
the smartwatch used, the number of subjects and the reference standard. 36 checks.

Verdicts (judged by the author, with the expert value as the answer):

- **match**: the same value (device named, same count, same reference test);
- **vague**: not wrong but less precise (for example "smartwatches" for "Samsung Simband");
- **differs**: a different number, backed by a verified quote from the paper (for example the
  total enrolled where the expert gives a count per device);
- **wrong**: the value contradicts the expert table;
- **not reported**: the pipeline found no value with a verified quote.

The T3 review (Arredondo-Velázquez 2026) has richer tables, but its publisher's site refuses
automated downloads, so it was not used.
