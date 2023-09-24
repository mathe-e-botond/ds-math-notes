import re

def convert_md_chapters(content):
    # Convert titles based on the number of hashes and ignore the numbering, matching entire lines
    content = re.sub(r'^# \**([\d.]+ )?(.+?)\**$', r'\\chapter{\2}', content, flags=re.MULTILINE)    
    content = re.sub(r'^## \**([\d.]+ )?(.+?)\**$', r'\\section{\2}', content, flags=re.MULTILINE)    
    content = re.sub(r'^### \**([\d.]+ )?(.+?)\**$', r'\\subsection{\2}', content, flags=re.MULTILINE)
    
    return content

def convert_md_text_formatting(content):
    # Convert **...** to \textbf{...} for bold text
    content = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1\\index{\1}}', content)
    
    # Convert *...* to \textit{...} for italic text
    content = re.sub(r'(?<![*$\d])\*(.+?)(?![*$\d])', r'\\textit{\1}', content)
    
    # Convert <br> to
    content = content.replace('<br>', '\\\\')
    
    return content

def escape_latex_special_chars(content):
    special_chars_map = {
        '_': r'\_',
        '#': r'\#',
        '$': r'\$',
        '%': r'\%',
        '&': r'\&',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde',
        '^': r'\textasciicircum'
    }

    for char, escape_seq in special_chars_map.items():
        content = content.replace(char, escape_seq)

    return content

def convert_html_image_to_latex(src, width, caption):
    if width:
        reduced_width = int(width) // 2  # This halves the width value
        latex_width = f'width={reduced_width}pt'
    else:
        latex_width = 'width=0.5\linewidth'  # This makes the image half the width of the text column

    latex_img = f"""
\\begin{{figure}}[htbp]
    \\begin{{center}}
        \\includegraphics[{latex_width}]{{{src}}}
        \\caption{{{caption}}}
    \\end{{center}}
\\end{{figure}}
"""

    return latex_img

