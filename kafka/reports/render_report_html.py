"""
Generates a standalone, publication-quality HTML document from MEMBER1_FINAL_COMPREHENSIVE_REPORT.md.
Includes:
- GitHub Markdown CSS styling
- KaTeX for LaTeX mathematical formulas
- Mermaid.js for architecture diagrams
- Clean print styles matching VS Code Markdown PDF export
"""
import sys
from pathlib import Path

def generate_html():
    root_dir = Path(__file__).resolve().parent.parent.parent
    md_file = root_dir / "MEMBER1_FINAL_COMPREHENSIVE_REPORT.md"
    if not md_file.exists():
        md_file = root_dir / "kafka" / "reports" / "MEMBER1_FINAL_COMPREHENSIVE_REPORT.md"
    
    output_html = root_dir / "kafka" / "reports" / "MEMBER1_FINAL_COMPREHENSIVE_REPORT.html"
    
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Escape backticks and backslashes for JS template literal
    escaped_md = md_content.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Member 1 — Apache Kafka Event Ingestion & Streaming Layer: Comprehensive Report</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <!-- GitHub Markdown CSS -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.1/github-markdown.min.css">
  <!-- Highlight.js for code syntax highlighting -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
  <!-- KaTeX for Math -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  
  <style>
    @page {{
      size: A4;
      margin: 18mm 16mm 20mm 16mm;
      @top-left {{
        content: "MEMBER1_FINAL_COMPREHENSIVE_REPORT.md";
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        font-size: 8.5pt;
        color: #57606a;
      }}
      @top-right {{
        content: "2026-10-17";
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        font-size: 8.5pt;
        color: #57606a;
      }}
      @bottom-center {{
        content: counter(page) " / " counter(pages);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        font-size: 9pt;
        color: #24292f;
      }}
    }}

    body {{
      background-color: #ffffff;
      color: #24292f;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 30px;
    }}

    .markdown-body {{
      box-sizing: border-box;
      min-width: 200px;
      max-width: 960px;
      margin: 0 auto;
      font-size: 14px;
      line-height: 1.6;
    }}

    .markdown-body table {{
      display: table;
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      margin-bottom: 16px;
    }}

    .markdown-body table th, .markdown-body table td {{
      padding: 8px 12px;
      border: 1px solid #d0d7de;
    }}

    .markdown-body table th {{
      background-color: #f6f8fa;
      font-weight: 600;
    }}

    .markdown-body pre {{
      background-color: #f6f8fa;
      border-radius: 6px;
      padding: 14px;
      border: 1px solid #d0d7de;
    }}

    .mermaid {{
      text-align: center;
      background: #fafbfc;
      padding: 16px;
      border: 1px solid #e1e4e8;
      border-radius: 6px;
      margin: 16px 0;
    }}

    @media print {{
      body {{
        padding: 0;
      }}
      .markdown-body {{
        max-width: 100%;
        font-size: 11pt;
      }}
      pre, blockquote, table, tr, img {{
        page-break-inside: avoid;
      }}
      h1, h2, h3, h4 {{
        page-break-after: avoid;
      }}
    }}
  </style>

  <!-- Marked.js -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <!-- Highlight.js -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <!-- KaTeX -->
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
  <!-- Mermaid.js -->
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head>
<body>
  <article class="markdown-body" id="content">
    Loading report...
  </article>

  <script>
    const rawMarkdown = `{escaped_md}`;

    // Configure marked
    marked.setOptions({{
      highlight: function(code, lang) {{
        if (lang === 'mermaid') {{
          return '<div class="mermaid">' + code + '</div>';
        }}
        const language = hljs.getLanguage(lang) ? lang : 'plaintext';
        return hljs.highlight(code, {{ language }}).value;
      }},
      breaks: false,
      gfm: true
    }});

    // Render markdown
    document.getElementById('content').innerHTML = marked.parse(rawMarkdown);

    // Render KaTeX math
    renderMathInElement(document.getElementById('content'), {{
      delimiters: [
        {{left: '$$', right: '$$', display: true}},
        {{left: '$', right: '$', display: false}},
        {{left: '\\\\(', right: '\\\\)', display: false}},
        {{left: '\\\\[', right: '\\\\]', display: true}}
      ],
      throwOnError: false
    }});

    // Initialize Mermaid
    mermaid.initialize({{
      startOnLoad: true,
      theme: 'default',
      securityLevel: 'loose'
    }});
  </script>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"Successfully generated HTML report at: {output_html}")

if __name__ == "__main__":
    generate_html()
