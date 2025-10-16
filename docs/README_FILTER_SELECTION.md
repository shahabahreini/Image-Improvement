# Documentation Index: Filter Selection Feature

## Quick Start

If you're in a hurry, read these first:

1. **QUICK_REFERENCE.md** - One-line commands for each filter
2. **example_filter_usage.sh** - Copy-paste examples

## Detailed Guides

For comprehensive understanding:

1. **FILTER_SELECTION_GUIDE.md** - Complete feature guide with best practices
2. **COMMAND_REFERENCE.md** - All commands and options explained
3. **IMPLEMENTATION_SUMMARY.md** - What was changed and why

## Technical Details

For developers:

1. **ARCHITECTURE.md** - Implementation details and design

## Implementation Files

The actual code changes:

1. **LargeImage_Tiler.py** - Modified split command with filter selection

## Summary of What Was Added

### New Capabilities

- Choose which denoising filter to use (NLM, BM3D, Enhanced BM3D, or Custom)
- Automatically generate processing scripts with correct filter commands
- Pass additional arguments to filters
- Support for custom filters via command-line

### New Command-Line Arguments

```
--filter {nlm|bm3d|enhanced_bm3d|custom}    Select filter type
--filter-args ARGS                           Pass args to filter
--create-script                              Generate processing script
--custom-command CMD                         Custom filter command
```

### Supported Filters

| Filter        | Speed  | Quality   | Best For             |
| ------------- | ------ | --------- | -------------------- |
| NLM           | Fast   | Good      | Quick previews       |
| BM3D          | Medium | Excellent | Standard use         |
| Enhanced BM3D | Medium | Excellent | Production (default) |
| Custom        | Varies | Varies    | Special needs        |

## Using This Documentation

### I want to...

#### Process an image quickly

→ Read: QUICK_REFERENCE.md
→ Command: `python3 LargeImage_Tiler.py split image.tif tiles/ --filter nlm --create-script`

#### Process an image with best quality

→ Read: FILTER_SELECTION_GUIDE.md
→ Command: `python3 LargeImage_Tiler.py split image.tif tiles/ --create-script`

#### Use a specific filter

→ Read: COMMAND_REFERENCE.md
→ Find your filter in "All Available Options"

#### Understand how it works

→ Read: ARCHITECTURE.md
→ See: Data flow diagrams and class changes

#### Use a custom filter

→ Read: FILTER_SELECTION_GUIDE.md, section "Custom Filter"
→ Example: See example_filter_usage.sh

#### See working examples

→ Read: example_filter_usage.sh
→ Copy commands directly

## Document Contents Overview

### QUICK_REFERENCE.md (2.8 KB)

- Filter options in one table
- One-liner commands for each filter
- Command structure
- Default behavior

### FILTER_SELECTION_GUIDE.md (5.2 KB)

- Available filters explained
- When to use each filter
- Complete workflow examples
- Best practices and tips
- Troubleshooting guide

### COMMAND_REFERENCE.md (6.7 KB)

- Full command syntax
- All options explained
- 10+ detailed examples
- Merge command examples
- Quick decision tree
- Error messages and fixes

### IMPLEMENTATION_SUMMARY.md (3.7 KB)

- What was added
- Key changes to code
- Usage examples
- Files modified
- Benefits of the changes

### ARCHITECTURE.md (5.1 KB)

- Implementation overview diagram
- Function signatures before/after
- Generated script structure
- Filter command templates
- Data flow examples

### example_filter_usage.sh (1.3 KB)

- 5 quick examples
- Copy-paste ready
- Covers all main use cases

## File Relationships

```
LargeImage_Tiler.py (Modified)
    ↓
    Uses → NLM_Filter .py
    Uses → Basic_BM3D.py
    Uses → Enhanced_BM3D.py
    Or → Custom filter via --custom-command

Documentation:
    QUICK_REFERENCE.md ────→ Start here
    FILTER_SELECTION_GUIDE.md ──→ Detailed learning
    COMMAND_REFERENCE.md ──→ Command lookup
    ARCHITECTURE.md ────────→ Technical details
    IMPLEMENTATION_SUMMARY.md → What changed
    example_filter_usage.sh ────→ Copy-paste examples
```

## Feature Highlights

### Easy Filter Selection

```bash
# Just add --filter option
python3 LargeImage_Tiler.py split image.tif tiles/ --filter nlm --create-script
```

### Automatic Script Generation

```bash
# Generates process_tiles.sh with correct commands
cd tiles/
./process_tiles.sh
```

### Custom Filter Support

```bash
# Use your own denoising tool
python3 LargeImage_Tiler.py split image.tif tiles/ \
  --filter custom \
  --custom-command "my_tool {input} {output}" \
  --create-script
```

### Pass Filter Arguments

```bash
# Additional parameters for filters
--filter-args "--workers 4 --tile-size 2048"
```

## Getting Started

1. **Read**: QUICK_REFERENCE.md (2 minutes)
2. **Choose**: Your filter (NLM, BM3D, or Enhanced BM3D)
3. **Run**: One command to split and create script
4. **Execute**: The generated process_tiles.sh
5. **Merge**: Results back together

## Common Tasks

### Task: Process image with fastest speed

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ --filter nlm --create-script
```

See: QUICK_REFERENCE.md → NLM

### Task: Process image with best quality

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ --create-script
```

See: QUICK_REFERENCE.md → Default (Enhanced BM3D)

### Task: Process with custom parameters

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ \
  --filter enhanced_bm3d \
  --filter-args "--workers 2" \
  --create-script
```

See: COMMAND_REFERENCE.md → Examples

### Task: Use custom filter

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ \
  --filter custom \
  --custom-command "denoise {input} {output}" \
  --create-script
```

See: FILTER_SELECTION_GUIDE.md → Custom Filter

## Support Reference

| Question                       | Document                  | Section               |
| ------------------------------ | ------------------------- | --------------------- |
| What filters are available?    | QUICK_REFERENCE.md        | All Options           |
| How do I use NLM?              | example_filter_usage.sh   | Line 2                |
| What are the command options?  | COMMAND_REFERENCE.md      | All Available Options |
| How does it work internally?   | ARCHITECTURE.md           | Architecture Overview |
| What was changed in the code?  | IMPLEMENTATION_SUMMARY.md | Key Changes           |
| Can I use my own filter?       | FILTER_SELECTION_GUIDE.md | Custom Filter         |
| I have an error, what do I do? | COMMAND_REFERENCE.md      | Error Messages        |

## Version Information

- **Feature**: Filter Selection for Tile Processing
- **Component**: LargeImage_Tiler.py
- **Date**: October 16, 2025
- **Status**: Production Ready
- **Backward Compatibility**: Full (defaults to enhanced_bm3d)

## Quick Navigation

From anywhere:

- Want examples? → `cat example_filter_usage.sh`
- Want full commands? → `cat COMMAND_REFERENCE.md`
- Want to learn? → `cat FILTER_SELECTION_GUIDE.md`
- Want technical details? → `cat ARCHITECTURE.md`
- Just want a reminder? → `cat QUICK_REFERENCE.md`
