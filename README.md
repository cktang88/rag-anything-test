# RAG-Anything Test Setup

This project demonstrates how to use RAG-Anything with MinerU v2.5 for multimodal document processing.

## Installation

This project uses `uv` for dependency management. The project has been initialized with:

```bash
uv init --python 3.10
uv add 'raganything[all]'
```

## Files

- `raganything_example.py` - Main example script with full RAG-Anything functionality
- `test_mineru.py` - Test script to verify MinerU installation (no API key required)

## Usage

### 1. Test MinerU Installation

First, verify that MinerU v2.5 is properly installed:

```bash
# Test installation only
uv run python test_mineru.py --check-only

# Test with a document (parsing test)
uv run python test_mineru.py --file path/to/your/document.pdf
```

### 2. Run Full Example

To run the complete RAG-Anything example, you'll need an OpenAI API key:

```bash
# Using command line argument
uv run python raganything_example.py path/to/document.pdf --api-key YOUR_API_KEY

# Using environment variable
export OPENAI_API_KEY=your_api_key_here
uv run python raganything_example.py path/to/document.pdf

# With custom options
uv run python raganything_example.py document.pdf \
  --api-key YOUR_API_KEY \
  --parser mineru \
  --parse-method auto \
  --output-dir ./output \
  --working-dir ./rag_storage
```

### 3. Available Options

- `--parser`: Choose between "mineru" (default) or "docling"
- `--parse-method`: Choose between "auto" (default), "ocr", or "txt"
- `--output-dir`: Directory for processed documents (default: ./output)
- `--working-dir`: Directory for RAG storage (default: ./rag_storage)
- `--base-url`: Custom OpenAI base URL (optional)

## Features Demonstrated

1. **Document Processing**: Complete end-to-end processing of PDFs, Office documents, images
2. **Multimodal Support**: Processing of text, images, tables, and equations
3. **Query Types**:
   - Pure text queries
   - VLM-enhanced queries (automatic image analysis)
   - Multimodal queries with specific content
4. **MinerU v2.5 Integration**: Latest version with improved parsing capabilities

## Requirements

- Python 3.10+
- uv package manager
- OpenAI API key (for full functionality)
- LibreOffice (for Office document processing)

## Troubleshooting

If you encounter issues:

1. **MinerU not found**: Run `uv run python test_mineru.py --check-only`
2. **Office documents not processing**: Install LibreOffice
3. **API errors**: Verify your OpenAI API key and base URL
4. **Memory issues**: Try using `--parse-method txt` for text-only processing

## Example Output

The script will:
1. Process your document using MinerU v2.5
2. Extract text, images, tables, and equations
3. Create a knowledge graph
4. Run example queries to demonstrate capabilities
5. Save processed files to the output directory
