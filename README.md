# Search Service Processing Scripts

## Overview
This repository contains the local development scripts used for running and processing search service requests. Note that this is a development repository intended to showcase the current workflow and prompts, and does not represent the final production application.

## Quick Start

1. Clone the repository
2. Install dependencies
3. Begin with criteria generation (see Workflow section below)

## Project Structure
The project saves all generated files in subfolders following a `test-{number}` pattern, where the number increments based on existing items. Each script saves its output to these subfolders, which can then be referenced by defining a path/test number to load:
- Queries
- Criteria
- Results

## Input Configuration
All scripts require input configuration through variables:
- Target/input text for criteria and query generation is set via variables
- Scoring requires manual target text input
- Results and criteria are loaded from their respective subfolders

## Workflow

### 1. Generate Criteria
- Run the criteria generation script first
- Output saves automatically to a new test subfolder 

### 2. Generate Queries
- Uses the same input text as criteria generation
- Saves to designated subfolder

### 3. Perform Search
Two search paths available:

#### Patent Search
- Patent text extraction happens during search
- No additional extraction step needed
- Proceed directly to scoring

#### Scholar Search
- Requires separate extraction step after initial results
- Run extraction before scoring/analysis

### 4. Scoring & Analysis

#### For Patent Search
- Run scoring immediately after search
- Required inputs:
  - Result set path/test number
  - Input target text
  - Criteria subfolder test path

#### For Scholar Search
1. Run extraction script first:
   - Define result set via subfolder/test number
2. Run scoring script:
   - Define new result set containing extracted fields
   - Define criteria and target input text

## Output Format
All results are saved as JSON files containing:
- Query information
- Cost data
- Search metadata
- Result set of individual objects including:
  - Relevant texts
  - LLM analysis
  - Score
  - Patent/result metafields
