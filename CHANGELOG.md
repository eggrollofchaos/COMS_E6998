# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- Added an inline party-popper image macro for the HW3 XeLaTeX PDF build so the bonus heading renders without dropping the `🎉` markers.
- Removed the `enumitem` dependency from the HW3 XeLaTeX header so the report builds on leaner TeX installations.
- Restored the HW3 Triton notebook cell source formatting to standard line-by-line Jupyter JSON so the submitted `c4.ipynb` matches the template structure more closely.
- Repacked the HW3 submission archive with an explicit file list so `wax1.zip` excludes local notebook checkpoints and other non-required files.
- Added a sidecar HW3 Triton notebook backup artifact that preserves the earlier flattened giant-string cell representation for comparison/debugging.
- Added a blank line in the normalized HW3 Triton notebook around the `golden_out` setup cell to preserve readability after the formatting restore.
